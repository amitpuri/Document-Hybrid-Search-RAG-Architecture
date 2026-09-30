"""
Graph Extraction and Entity Resolution Layer.
Extracts entities and relationships from DocumentChunk sequences,
resolves aliases, and links knowledge graph elements to source chunks.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence, Set, Tuple

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
        self, chunks: List[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a sequence of document chunks."""
        pass


# Common stopwords and non-entity words to suppress false positives
_IGNORE_WORDS = {
    "THE",
    "AND",
    "FOR",
    "THAT",
    "THIS",
    "WITH",
    "FROM",
    "HAVE",
    "BEEN",
    "THEIR",
    "WHICH",
    "MORE",
    "THESE",
    "TABLE",
    "FIGURE",
    "SECTION",
    "PAGE",
    "RESULTS",
    "METHOD",
    "METHODS",
    "MODEL",
    "MODELS",
    "PAPER",
    "STUDY",
    "APPROACH",
    "SYSTEM",
    "DATA",
    "USING",
    "BASED",
    "SUCH",
    "BOTH",
    "EACH",
    "OTHER",
    "WHERE",
    "WHEN",
    "INTO",
    "ONLY",
    "ALSO",
    "SOME",
    "MANY",
    "MOST",
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
    entities: List[Entity], similarity_threshold: float = 0.88
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


# ── Affiliation / Author Noise Patterns ──────────────────────────────────────
# Reject entities that immediately follow these academic institution keywords
_ACADEMIC_AFFILIATION_WORDS = re.compile(
    r"\b(?:University|College|Department|Institut(?:e|o)|"
    r"School\s+of|Faculty\s+of|Laboratory|Lab\s+of|Center\s+for|"
    r"Centre\s+for|Division\s+of|Program\s+in|Graduate\s+School|"
    r"Foundation|Corporation|Research\s+Group)\b",
    re.IGNORECASE,
)

# Reject section-title artifacts: digit/roman-numeral prefix patterns
_SECTION_ARTIFACT_RE = re.compile(
    r"^(?:§\s*|\d+(?:\.\d+)*\.?\s+|[IVXLCDM]+\.\s+|[A-Z]\.\s+|Abstract|"
    r"Introduction|Conclusion|References|Acknowledgm|Appendix|"
    r"Related Work)",
    re.IGNORECASE,
)

# Reject names that are likely personal names (Firstname Lastname capitalised, no technical marker)
_PERSON_NAME_RE = re.compile(r"^[A-Z][a-z]{2,15}\s+[A-Z][a-z]{2,15}$")


def _is_likely_noise(name: str, entity_type: str, num_chunks: int) -> bool:
    """Returns True if the entity is likely to be noise and should be dropped."""
    # Always drop tiny items in only 1 chunk (unless they're known acronyms ≥ 3 chars)
    if num_chunks <= 1:
        if entity_type == "TECHNICAL_ACRONYM" and len(name) >= 3:
            pass  # short acronyms seen once may still be legit
        elif len(name) < 5:
            return True

    # Drop section-title artifacts (numeric/letter prefix in name)
    if entity_type == "SECTION_CONCEPT":
        if _SECTION_ARTIFACT_RE.search(name):
            return True
        # Drop section concepts seen only in 1 chunk — they're usually layout noise
        if num_chunks <= 1:
            return True

    # Drop entities matching academic affiliation patterns
    if _ACADEMIC_AFFILIATION_WORDS.search(name):
        return True

    # Drop likely personal-name entities (Firstname Lastname)
    if _PERSON_NAME_RE.match(name) and entity_type == "CONCEPT":
        return True

    # Drop entities with digits mixed into word (section title artifacts like "1 Intro")
    if re.search(r"\b\d+\s+[A-Z]", name):
        return True

    return False


def _strip_section_prefix(section: str) -> str:
    """Strips numeric, letter, or Roman-numeral prefix from section headings."""
    cleaned = re.sub(
        r"^(?:§\s*|\d+(?:\.\d+)*\.?\s+|[IVXLCDM]+\.\s+|[A-Z]\.\s+)",
        "",
        section.strip(),
    )
    return cleaned.strip()


class HeuristicGraphExtractor(BaseGraphExtractor):
    """
    High-speed, deterministic, zero-cost offline entity and relationship extractor.
    Uses regex pattern extraction, technical acronym identification, section parsing,
    and sentence-level entity co-occurrence modeling.

    Noise-reduction measures:
    - Drops entities seen in ≤1 chunk with length <5 (non-acronyms)
    - Strips section number prefixes from section-title entities
    - Rejects academic affiliation names (University, Department, …)
    - Rejects likely personal names (Firstname Lastname pattern)
    - Caps relationship weight at log scale to prevent boilerplate dominance
    """

    def __init__(self, min_entity_freq: int = 1, max_cooccur_dist: int = 250):
        self.min_entity_freq = min_entity_freq
        self.max_cooccur_dist = max_cooccur_dist

    def extract_from_chunks(
        self, chunks: List[DocumentChunk]
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

            # 1. Section Title entities — strip number prefixes, skip generic headings
            if section and section.lower() not in {
                "overview",
                "abstract",
                "introduction",
                "references",
                "conclusion",
                "acknowledgments",
                "appendix",
                "related work",
            }:
                sec_clean = _strip_section_prefix(normalize_entity_name(section))
                if (
                    len(sec_clean) > 4
                    and sec_clean.upper() not in _IGNORE_WORDS
                    and not _SECTION_ARTIFACT_RE.search(sec_clean)
                    and not _ACADEMIC_AFFILIATION_WORDS.search(sec_clean)
                ):
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

            # 4. Multi-word Proper Nouns / Key Concepts (e.g. Binding Constraint Thesis)
            for m in _PROPER_NOUN_REGEX.finditer(text):
                term = m.group(0)
                words = term.split()
                if not all(w.upper() in _IGNORE_WORDS for w in words):
                    norm = normalize_entity_name(term)
                    if (
                        len(norm) > 4
                        and not _ACADEMIC_AFFILIATION_WORDS.search(norm)
                        and not _PERSON_NAME_RE.match(norm)
                    ):
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

        # ── Noise Filter ─────────────────────────────────────────────────────
        filtered_entities = [
            e
            for e in deduped_entities
            if not _is_likely_noise(e.name, e.entity_type, len(e.chunk_ids))
        ]
        valid_entity_names = {e.name for e in filtered_entities}

        # Build relationships — only between surviving entities; cap weight logarithmically
        import math as _math

        relationships: List[Relationship] = []
        for (src, tgt), info in pair_counts.items():
            if src in valid_entity_names and tgt in valid_entity_names and src != tgt:
                desc = (
                    info["snippets"][0]
                    if info["snippets"]
                    else f"Co-occurs in {len(info['chunk_ids'])} chunks"
                )
                if len(desc) > 200:
                    desc = desc[:197] + "..."  # noqa: E501
                # Cap weight: log-scale prevents boilerplate pairs from dominating
                capped_weight = float(1.0 + _math.log1p(info["count"]))
                relationships.append(
                    Relationship(
                        source=src,
                        target=tgt,
                        relation_type="CO_OCCURS_WITH",
                        description=desc,
                        weight=capped_weight,
                        chunk_ids=info["chunk_ids"],
                    )
                )

        return filtered_entities, relationships


_KG_EXTRACTION_SCHEMA = """\
You are a knowledge graph extraction engine for scientific/technical documents.
Extract entities and relationships from the provided text chunk.

Return ONLY a single JSON object matching this exact schema:
{
  "entities": [
    {
      "name": "string",
      "entity_type": "CONCEPT|TECHNICAL_ACRONYM|FRAMEWORK_COMPONENT|MODEL|METRIC|ALGORITHM",
      "description": "short description"
    }
  ],
  "relationships": [
    {
      "source": "entity_name",
      "target": "entity_name",
      "relation_type": "USES|EXTENDS|EVALUATES|CONSTRAINS|IMPLEMENTS|RELATED_TO",
      "description": "brief context",
      "weight": 1.0
    }
  ]
}

Rules:
- Extract domain-specific technical entities (algorithms, frameworks, models,
  metrics, formal concepts).
- Do NOT extract author names, affiliations, section numbers, or generic phrases.
- entity names must match the exact text found in the chunk.
- Keep descriptions to 20 words or fewer.
- Emit an empty list [] if none found.

Chunk text:
\"\"\"
{chunk_text}
\"\"\"
"""


class LLMGraphExtractor(BaseGraphExtractor):
    """
    High-precision LLM-based entity and relationship extractor.
    Extracts structured knowledge triplets from chunks via external LLM calls
    using a JSON-schema prompt per chunk (or batch of chunks).
    Falls back to HeuristicGraphExtractor on parse/API failure.
    """

    def __init__(
        self,
        generator=None,
        fallback_on_error: bool = True,
        batch_size: int = 1,
    ):
        self.generator = generator
        self.fallback = HeuristicGraphExtractor()
        self.fallback_on_error = fallback_on_error
        self.batch_size = batch_size

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Strips markdown code fences (```json ... ```) from LLM responses."""
        import re as _re

        text = text.strip()
        text = _re.sub(r"^```(?:json)?\s*", "", text, flags=_re.IGNORECASE)
        text = _re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _parse_response(
        raw: str, chunk_id: int, doc_name: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parses a JSON-schema LLM response into typed Entity and Relationship objects."""
        import json as _json

        cleaned = LLMGraphExtractor._strip_fences(raw)
        data = _json.loads(cleaned)

        entities: List[Entity] = []
        for ent_dict in data.get("entities", []):
            name = normalize_entity_name(str(ent_dict.get("name", "")).strip())
            if not name or len(name) < 2:
                continue
            entities.append(
                Entity(
                    name=name,
                    entity_type=str(ent_dict.get("entity_type", "CONCEPT")),
                    description=str(ent_dict.get("description", "")),
                    chunk_ids={chunk_id},
                    doc_names={doc_name},
                )
            )

        relationships: List[Relationship] = []
        for rel_dict in data.get("relationships", []):
            src = normalize_entity_name(str(rel_dict.get("source", "")).strip())
            tgt = normalize_entity_name(str(rel_dict.get("target", "")).strip())
            if not src or not tgt or src == tgt:
                continue
            relationships.append(
                Relationship(
                    source=src,
                    target=tgt,
                    relation_type=str(rel_dict.get("relation_type", "RELATED_TO")),
                    description=str(rel_dict.get("description", "")),
                    weight=float(rel_dict.get("weight", 1.0)),
                    chunk_ids={chunk_id},
                )
            )
        return entities, relationships

    def _extract_chunk_via_llm(
        self,
        chunk: "DocumentChunk",
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Calls the configured generator on a single chunk and parses the JSON response."""
        prompt_text = _KG_EXTRACTION_SCHEMA.format(chunk_text=chunk.text[:2000])
        # We use the generator's generate() method with an empty context — the prompt IS the context
        result = self.generator.generate(
            query=prompt_text,
            formatted_context="",
            retrieved_chunks=[chunk],
            strategy_used="llm_kg_extraction",
        )
        raw_answer = result.answer if hasattr(result, "answer") else str(result)
        return self._parse_response(raw_answer, chunk.chunk_id, chunk.doc_name)

    def extract_from_chunks(
        self,
        chunks: Sequence["DocumentChunk"],
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extracts entities and relationships via LLM calls per chunk.
        Falls back to HeuristicGraphExtractor on any parse or API error.
        """
        if self.generator is None:
            return self.fallback.extract_from_chunks(list(chunks))

        import json as _json

        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []
        failed_chunks: List["DocumentChunk"] = []

        for chunk in chunks:
            try:
                ents, rels = self._extract_chunk_via_llm(chunk)
                all_entities.extend(ents)
                all_relationships.extend(rels)
            except (_json.JSONDecodeError, ValueError, KeyError):
                # Parse failure — fall back this chunk to heuristic
                failed_chunks.append(chunk)
            except Exception:
                # API or unexpected error
                if self.fallback_on_error:
                    failed_chunks.append(chunk)
                else:
                    raise

        # Fall back any failed chunks via HeuristicGraphExtractor
        if failed_chunks:
            h_ents, h_rels = self.fallback.extract_from_chunks(failed_chunks)
            all_entities.extend(h_ents)
            all_relationships.extend(h_rels)

        # Resolve duplicates across all accumulated entities
        all_entities = resolve_duplicate_entities(all_entities)

        # Filter relationships to only include resolved entity names
        valid_names = {e.name for e in all_entities}
        all_relationships = [
            r for r in all_relationships if r.source in valid_names and r.target in valid_names
        ]

        return all_entities, all_relationships
