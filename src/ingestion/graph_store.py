"""
Graph Storage Abstraction & NetworkX Implementation.
Provides in-memory graph management, community detection (Louvain),
neighborhood traversal, and atomic JSON persistence.
"""

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Sequence

import networkx as nx

from src.ingestion.graph_extractor import Entity, Relationship


class BaseGraphStore(ABC):
    """Abstract interface for storing and querying Knowledge Graphs."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> None:
        """Adds or updates an entity node in the graph."""
        pass

    @abstractmethod
    def add_relationship(self, rel: Relationship) -> None:
        """Adds or updates an edge between entities in the graph."""
        pass

    @abstractmethod
    def get_entity(self, name: str) -> Optional[Entity]:
        """Retrieves an entity by name."""
        pass

    @abstractmethod
    def get_all_entities(self) -> List[Entity]:
        """Retrieves all entities in the graph."""
        pass

    @abstractmethod
    def get_all_relationships(self) -> List[Relationship]:
        """Retrieves all relationships in the graph."""
        pass

    @abstractmethod
    def get_neighbors(self, name: str, hops: int = 1) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Retrieves neighboring nodes and edges within N hops."""
        pass

    @abstractmethod
    def get_communities(self) -> Dict[int, List[str]]:
        """Returns detected communities as {community_id: [entity_names]}."""
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persists the graph to disk."""
        pass

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Loads the graph from disk."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of nodes in the graph."""
        pass


class NetworkXGraphStore(BaseGraphStore):
    """
    NetworkX in-memory graph backend with JSON persistence.
    Uses nx.MultiDiGraph to support multiple directed typed relationships between entities.
    """

    def __init__(self):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._communities_cache: Optional[Dict[int, List[str]]] = None

    def add_entity(self, entity: Entity) -> None:
        self._communities_cache = None
        if self.graph.has_node(entity.name):
            # Update attributes and union chunk_ids
            existing_chunks = self.graph.nodes[entity.name].get("chunk_ids", set())
            existing_docs = self.graph.nodes[entity.name].get("doc_names", set())
            existing_chunks.update(entity.chunk_ids)
            existing_docs.update(entity.doc_names)
            self.graph.nodes[entity.name]["chunk_ids"] = existing_chunks
            self.graph.nodes[entity.name]["doc_names"] = existing_docs
            if entity.description and not self.graph.nodes[entity.name].get("description"):
                self.graph.nodes[entity.name]["description"] = entity.description
        else:
            self.graph.add_node(
                entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                chunk_ids=set(entity.chunk_ids),
                doc_names=set(entity.doc_names),
                metadata=dict(entity.metadata),
            )

    def add_relationship(self, rel: Relationship) -> None:
        self._communities_cache = None
        if not self.graph.has_node(rel.source):
            self.add_entity(Entity(name=rel.source))
        if not self.graph.has_node(rel.target):
            self.add_entity(Entity(name=rel.target))

        # Check if edge with same relation_type already exists
        edge_exists = False
        if self.graph.has_edge(rel.source, rel.target):
            edge_dict = self.graph.get_edge_data(rel.source, rel.target)
            for k, attrs in edge_dict.items():
                if attrs.get("relation_type") == rel.relation_type:
                    attrs["weight"] = attrs.get("weight", 1.0) + rel.weight
                    attrs.get("chunk_ids", set()).update(rel.chunk_ids)
                    edge_exists = True
                    break

        if not edge_exists:
            self.graph.add_edge(
                rel.source,
                rel.target,
                relation_type=rel.relation_type,
                description=rel.description,
                weight=rel.weight,
                chunk_ids=set(rel.chunk_ids),
                metadata=dict(rel.metadata),
            )

    def get_entity(self, name: str) -> Optional[Entity]:
        if not self.graph.has_node(name):
            return None
        attrs = self.graph.nodes[name]
        return Entity(
            name=name,
            entity_type=attrs.get("entity_type", "CONCEPT"),
            description=attrs.get("description", ""),
            chunk_ids=set(attrs.get("chunk_ids", set())),
            doc_names=set(attrs.get("doc_names", set())),
            metadata=dict(attrs.get("metadata", {})),
        )

    def get_all_entities(self) -> List[Entity]:
        entities = []
        for node in self.graph.nodes():
            ent = self.get_entity(node)
            if ent:
                entities.append(ent)
        return entities

    def get_all_relationships(self) -> List[Relationship]:
        relationships = []
        for u, v, k, attrs in self.graph.edges(keys=True, data=True):
            relationships.append(Relationship(
                source=u,
                target=v,
                relation_type=attrs.get("relation_type", "RELATED_TO"),
                description=attrs.get("description", ""),
                weight=float(attrs.get("weight", 1.0)),
                chunk_ids=set(attrs.get("chunk_ids", set())),
                metadata=dict(attrs.get("metadata", {})),
            ))
        return relationships

    def get_neighbors(self, name: str, hops: int = 1) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Traverses the graph up to `hops` away from `name`.
        Returns a list of (source, target, edge_data) triples.
        """
        if not self.graph.has_node(name):
            return []

        visited_nodes: Set[str] = {name}
        frontier: Set[str] = {name}
        collected_edges: List[Tuple[str, str, Dict[str, Any]]] = []

        undirected = self.graph.to_undirected(as_view=True)

        for _ in range(hops):
            next_frontier: Set[str] = set()
            for current in frontier:
                for neighbor in undirected.neighbors(current):
                    # Get multi-edges between current and neighbor in directed graph
                    if self.graph.has_edge(current, neighbor):
                        for _, data in self.graph.get_edge_data(current, neighbor).items():
                            collected_edges.append((current, neighbor, dict(data)))
                    if self.graph.has_edge(neighbor, current):
                        for _, data in self.graph.get_edge_data(neighbor, current).items():
                            collected_edges.append((neighbor, current, dict(data)))

                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return collected_edges

    def get_communities(self) -> Dict[int, List[str]]:
        """
        Performs modularity-based Louvain community detection.
        Returns a mapping of community index to list of entity names.
        """
        if self._communities_cache is not None:
            return self._communities_cache

        if len(self.graph) == 0:
            return {}

        # Louvain runs on undirected graphs
        undirected = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            w = data.get("weight", 1.0)
            if undirected.has_edge(u, v):
                undirected[u][v]["weight"] += w
            else:
                undirected.add_edge(u, v, weight=w)

        # Include isolated nodes as individual nodes
        for node in self.graph.nodes():
            if node not in undirected:
                undirected.add_node(node)

        try:
            communities = nx.community.louvain_communities(undirected, weight="weight", seed=42)
            result = {i: sorted(list(comm)) for i, comm in enumerate(communities)}
        except Exception:
            # Fallback to connected components if Louvain encounters degeneracy
            components = list(nx.connected_components(undirected))
            result = {i: sorted(list(c)) for i, c in enumerate(components)}

        self._communities_cache = result
        return result

    def get_node_degree(self, name: str) -> int:
        if self.graph.has_node(name):
            return self.graph.degree(name)
        return 0

    def save(self, path: Path | str) -> None:
        """Atomically saves graph data to a JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".tmp")

        data = {
            "entities": [e.to_dict() for e in self.get_all_entities()],
            "relationships": [r.to_dict() for r in self.get_all_relationships()],
        }

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if os.path.exists(target_path):
            os.replace(tmp_path, target_path)
        else:
            tmp_path.rename(target_path)

    def load(self, path: Path | str) -> "NetworkXGraphStore":
        """Loads graph data from a JSON file."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Graph cache file not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph.clear()
        self._communities_cache = None

        for ent_dict in data.get("entities", []):
            self.add_entity(Entity.from_dict(ent_dict))

        for rel_dict in data.get("relationships", []):
            self.add_relationship(Relationship.from_dict(rel_dict))

        return self

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def number_of_edges(self) -> int:
        return self.graph.number_of_edges()
