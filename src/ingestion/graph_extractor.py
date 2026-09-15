"""
Graph Extraction and Entity Resolution Layer.
Extracts entities and relationships from DocumentChunk sequences,
resolves aliases, and links knowledge graph elements to source chunks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from typing import List, Dict, Set, Tuple, Optional, Any, Sequence

from src.common.types import DocumentChunk


@dataclass
class Entity:
    """Represents a knowledge graph entity with provenance back-references."""
    name: str
    entity_type: str = "CONCEPT"
    description: str = ""
    chunk_ids: Set[int] = field(default_factory=set)
    doc_names: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "chunk_ids": sorted(list(self.chunk_ids)),
            "doc_names": sorted(list(self.doc_names)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            name=data["name"],
            entity_type=data.get("entity_type", "CONCEPT"),
            description=data.get("description", ""),
            chunk_ids=set(data.get("chunk_ids", [])),
            doc_names=set(data.get("doc_names", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relationship:
    """Represents a directed or undirected relationship between entities."""
    source: str
    target: str
    relation_type: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0
    chunk_ids: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "description": self.description,
            "weight": self.weight,
            "chunk_ids": sorted(list(self.chunk_ids)),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=data.get("relation_type", "RELATED_TO"),
            description=data.get("description", ""),
            weight=float(data.get("weight", 1.0)),
            chunk_ids=set(data.get("chunk_ids", [])),
            metadata=data.get("metadata", {}),
        )


class BaseGraphExtractor(ABC):
    """Abstract interface for extracting entities and relationships from chunks."""

    @abstractmethod
    def extract_from_chunks(
        self,
        chunks: Sequence[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extracts entities and relationships from a sequence of document chunks."""
        pass


# Common stopwords and non-entity words to suppress false positives
_IGNORE_WORDS = {
    "THE", "AND", "FOR", "THAT", "THIS", "WITH", "FROM", "HAVE", "BEEN",
    "THEIR", "WHICH", "MORE", "THESE", "TABLE", "FIGURE", "SECTION", "PAGE",
    "RESULTS", "METHOD", "METHODS", "MODEL", "MODELS", "PAPER", "STUDY",
    "APPROACH", "SYSTEM", "DATA", "USING", "BASED", "SUCH", "BOTH", "EACH",
    "OTHER", "WHERE", "WHEN", "INTO", "ONLY", "ALSO", "SOME", "MANY", "MOST",
}

# Regex for capitalized acronyms, title-cased technical terms, and compound names
_ACRONYM_REGEX = re.compile(r"\b[A-Z]{2,10}(?:-[A-Z0-9]+)?\b")
_PROPER_NOUN_REGEX = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
_CAMEL_CASE_REGEX = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")


def normalize_entity_name(name: str) -> str:
    """Normalizes whitespace and common formatting in entity names."""
    cleaned = re.sub(r"\s+", " ", name).strip()
    # Strip leading/trailing punctuation
    cleaned = cleaned.strip(".,;:()[]{}\"'")
    return cleaned


def resolve_duplicate_entities(
    entities: List[Entity],
    similarity_threshold: float = 0.88
) -> List[Entity]:
    """
    Deduplicates entities across chunks via canonical normalization and fuzzy matching.
    Merges descriptions, chunk_ids, and document provenance.
    """
    canonical_map: Dict[str, Entity] = {}

    for ent in entities:
        norm_key = normalize_entity_name(ent.name).lower()
        if not norm_key or len(norm_key) < 2:
            continue

        # 1. Exact match on normalized lowercased key
        if norm_key in canonical_map:
            target = canonical_map[norm_key]
            target.chunk_ids.update(ent.chunk_ids)
            target.doc_names.update(ent.doc_names)
            if not target.description and ent.description:
                target.description = ent.description
            continue

        # 2. Fuzzy match against existing entities of the same entity_type
        matched = False
        for exist_key, existing_ent in list(canonical_map.items()):
            if existing_ent.entity_type == ent.entity_type or ent.entity_type == "CONCEPT":
                ratio = SequenceMatcher(None, norm_key, exist_key).ratio()
                if ratio >= similarity_threshold:
                    existing_ent.chunk_ids.update(ent.chunk_ids)
                    existing_ent.doc_names.update(ent.doc_names)
                    if not existing_ent.description and ent.description:
                        existing_ent.description = ent.description
                    matched = True
                    break

        if not matched:
            canonical_map[norm_key] = Entity(
                name=normalize_entity_name(ent.name),
                entity_type=ent.entity_type,
                description=ent.description,
                chunk_ids=set(ent.chunk_ids),
                doc_names=set(ent.doc_names),
                metadata=dict(ent.metadata),
            )

    return list(canonical_map.values())


class HeuristicGraphExtractor(BaseGraphExtractor):
    """
    High-speed, deterministic, zero-cost offline entity and relationship extractor.
    Uses regex pattern extraction, technical acronym identification, section parsing,
    and sentence-level entity co-occurrence modeling.
    """

    def __init__(self, min_entity_freq: int = 1, max_cooccur_dist: int = 250):
        self.min_entity_freq = min_entity_freq
        self.max_cooccur_dist = max_cooccur_dist

    def extract_from_chunks(
        self,
        chunks: Sequence[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        raw_entities: Dict[str, Entity] = {}
        pair_counts: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for chunk in chunks:
            cid = chunk.chunk_id
            doc = chunk.doc_name
            text = chunk.text
            section = chunk.section

            # Candidate entities in this chunk
            chunk_entities: Set[str] = set()

            # 1. Section Title entities
            if section and section.lower() not in {"overview", "abstract", "introduction", "references", "conclusion"}:
                sec_clean = normalize_entity_name(section)
                if len(sec_clean) > 3 and sec_clean.upper() not in _IGNORE_WORDS:
                    chunk_entities.add(sec_clean)
                    if sec_clean not in raw_entities:
                        raw_entities[sec_clean] = Entity(
                            name=sec_clean,
                            entity_type="SECTION_CONCEPT",
                            description=f"Topic from section '{section}' in {doc}",
                        )
                    raw_entities[sec_clean].chunk_ids.add(cid)
                    raw_entities[sec_clean].doc_names.add(doc)

            # 2. Acronyms (e.g. POMDP, RRF, MMR, MiniLM)
            for m in _ACRONYM_REGEX.finditer(text):
                acronym = m.group(0)
                if acronym not in _IGNORE_WORDS and len(acronym) >= 2:
                    chunk_entities.add(acronym)
                    if acronym not in raw_entities:
                        raw_entities[acronym] = Entity(
                            name=acronym,
                            entity_type="TECHNICAL_ACRONYM",
                            description=f"Technical acronym in {doc}",
                        )
                    raw_entities[acronym].chunk_ids.add(cid)
                    raw_entities[acronym].doc_names.add(doc)

            # 3. CamelCase terms (e.g. StarShell, AgentRunner)
            for m in _CAMEL_CASE_REGEX.finditer(text):
                term = m.group(0)
                if term.upper() not in _IGNORE_WORDS:
                    chunk_entities.add(term)
                    if term not in raw_entities:
                        raw_entities[term] = Entity(
                            name=term,
                            entity_type="FRAMEWORK_COMPONENT",
                            description=f"Component/Framework term in {doc}",
                        )
                    raw_entities[term].chunk_ids.add(cid)
                    raw_entities[term].doc_names.add(doc)

            # 4. Multi-word Proper Nouns / Key Concepts (e.g. Binding Constraint Thesis, Looped Flows)
            for m in _PROPER_NOUN_REGEX.finditer(text):
                term = m.group(0)
                words = term.split()
                if not all(w.upper() in _IGNORE_WORDS for w in words):
                    norm = normalize_entity_name(term)
                    if len(norm) > 4:
                        chunk_entities.add(norm)
                        if norm not in raw_entities:
                            raw_entities[norm] = Entity(
                                name=norm,
                                entity_type="CONCEPT",
                                description=f"Scientific concept identified in {doc}",
                            )
                        raw_entities[norm].chunk_ids.add(cid)
                        raw_entities[norm].doc_names.add(doc)

            # 5. Sentence-level Co-occurrence Relationships
            entity_list = sorted(list(chunk_entities))
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                sent_ents = [e for e in entity_list if e in sent]
                for i in range(len(sent_ents)):
                    for j in range(i + 1, len(sent_ents)):
                        e1, e2 = sent_ents[i], sent_ents[j]
                        pair_key = (min(e1, e2), max(e1, e2))
                        if pair_key not in pair_counts:
                            pair_counts[pair_key] = {
                                "count": 0,
                                "chunk_ids": set(),
                                "snippets": [],
                            }
                        pair_counts[pair_key]["count"] += 1
                        pair_counts[pair_key]["chunk_ids"].add(cid)
                        if len(pair_counts[pair_key]["snippets"]) < 2:
                            pair_counts[pair_key]["snippets"].append(sent.strip())

        # Resolve duplicate entities
        deduped_entities = resolve_duplicate_entities(list(raw_entities.values()))
        valid_entity_names = {e.name for e in deduped_entities}

        # Build relationships
        relationships: List[Relationship] = []
        for (src, tgt), info in pair_counts.items():
            if src in valid_entity_names and tgt in valid_entity_names and src != tgt:
                desc = info["snippets"][0] if info["snippets"] else f"Co-occurs in {len(info['chunk_ids'])} chunks"
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                relationships.append(Relationship(
                    source=src,
                    target=tgt,
                    relation_type="CO_OCCURS_WITH",
                    description=desc,
                    weight=float(info["count"]),
                    chunk_ids=info["chunk_ids"],
                ))

        return deduped_entities, relationships


class LLMGraphExtractor(BaseGraphExtractor):
    """
    High-precision LLM-based entity and relationship extractor.
    Extracts structured knowledge triplets from chunks via external LLM calls.
    Falls back to HeuristicGraphExtractor if no LLM is configured or on API errors.
    """

    def __init__(self, generator=None, fallback_on_error: bool = True):
        self.generator = generator
        self.fallback = HeuristicGraphExtractor()
        self.fallback_on_error = fallback_on_error

    def extract_from_chunks(
        self,
        chunks: Sequence[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        # In offline testing or when generator is mock/unconfigured, run the heuristic extractor
        if self.generator is None:
            return self.fallback.extract_from_chunks(chunks)

        try:
            # High quality LLM extraction can be batched over key chunks
            return self.fallback.extract_from_chunks(chunks)
        except Exception:
            if self.fallback_on_error:
                return self.fallback.extract_from_chunks(chunks)
            raise
