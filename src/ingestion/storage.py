"""
Storage Abstractions for Document Chunks and Metadata.
Enables decoupling retrieval from storage, supporting big-data persistence,
partitioned Parquet datasets, or in-memory stores.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    _PYARROW_AVAILABLE = True
except ImportError:
    pa = None
    pc = None
    pq = None
    _PYARROW_AVAILABLE = False

from src.common.types import DocumentChunk

try:
    from src.config import PARQUET_COMPRESSION
except (ImportError, AttributeError):
    PARQUET_COMPRESSION = "zstd"

# PyArrow schema for ParquetChunkStore.
#
# NOTE: `formatted_text` is intentionally NOT a persisted column. It is a
# pure function of (doc_name, page_num, section, text) via
# DocumentChunk.to_formatted_text(), and recomputing it for a few thousand
# rows costs microseconds -- not worth an ~10% file size increase on every
# write for a derived value.
#
# NOTE: `metadata` is stored as a JSON-encoded string rather than a typed
# PyArrow struct/map. DocumentChunk.metadata is a free-form Dict[str, Any]
# populated by heterogeneous extractors/chunkers; a typed column would
# require reconciling their schemas, whereas JSON keeps this store agnostic
# to whatever shape metadata happens to take.
if _PYARROW_AVAILABLE:
    CHUNK_PYARROW_SCHEMA = pa.schema(
        [
            pa.field("chunk_id", pa.int64(), nullable=False),
            pa.field("doc_name", pa.string(), nullable=False),
            pa.field("page_num", pa.int32(), nullable=False),
            pa.field("section", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("metadata_json", pa.string(), nullable=True),
        ]
    )
else:
    CHUNK_PYARROW_SCHEMA = None


def chunks_to_pyarrow_table(chunks: Sequence[DocumentChunk]) -> pa.Table:
    """Convert a sequence of DocumentChunk objects into a PyArrow Table.

    Uses the 6-column CHUNK_PYARROW_SCHEMA (no ``formatted_text`` column;
    that column is derivable via DocumentChunk.to_formatted_text() and is
    intentionally kept out of persisted storage to avoid redundancy).
    """
    if not _PYARROW_AVAILABLE:
        raise RuntimeError(
            "pyarrow is required for Parquet table operations. Install with: pip install pyarrow"
        )

    if not chunks:
        return CHUNK_PYARROW_SCHEMA.empty_table()

    chunk_ids: list = []
    doc_names: list = []
    page_nums: list = []
    sections: list = []
    texts: list = []
    metadata_jsons: list = []
    for c in chunks:
        chunk_ids.append(int(c.chunk_id))
        doc_names.append(str(c.doc_name))
        page_nums.append(int(c.page_num))
        sections.append(str(c.section))
        texts.append(str(c.text))
        metadata_jsons.append(json.dumps(c.metadata, ensure_ascii=False) if c.metadata else None)

    return pa.Table.from_pydict(
        {
            "chunk_id": chunk_ids,
            "doc_name": doc_names,
            "page_num": page_nums,
            "section": sections,
            "text": texts,
            "metadata_json": metadata_jsons,
        },
        schema=CHUNK_PYARROW_SCHEMA,
    )


def pyarrow_row_to_chunk(row: dict) -> DocumentChunk:
    """Reconstruct a DocumentChunk from a plain dict representing one PyArrow row.

    ``row`` is expected to have keys matching CHUNK_PYARROW_SCHEMA column
    names.  ``formatted_text`` is not a persisted column and is therefore
    not expected in ``row``.
    """
    meta_raw = row.get("metadata_json")
    metadata: dict = {}
    if meta_raw:
        try:
            metadata = json.loads(meta_raw)
        except Exception:
            metadata = {}
    return DocumentChunk(
        chunk_id=int(row["chunk_id"]),
        doc_name=str(row["doc_name"]),
        page_num=int(row["page_num"]),
        section=str(row["section"]),
        text=str(row["text"]),
        metadata=metadata,
    )


class BaseChunkStore(ABC):
    """Abstract storage interface for document chunks."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Adds a list of chunks to the store."""
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: int) -> Optional[DocumentChunk]:
        """Retrieves a chunk by its integer ID."""
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Retrieves all chunks in index order."""
        pass

    @abstractmethod
    def get_texts(self) -> List[str]:
        """Returns the formatted text representation of all chunks."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def get_chunks(self, chunk_ids: Sequence[int]) -> List[DocumentChunk]:
        """Batch lookup with default fallback to sequential get_chunk."""
        return [c for cid in chunk_ids if (c := self.get_chunk(cid)) is not None]

    def filter(
        self,
        doc_name: Optional[str] = None,
        page_num: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Predicate filtering with default fallback to in-memory scan."""
        results = []
        for c in self.get_all_chunks():
            if doc_name is not None and c.doc_name != doc_name:
                continue
            if page_num is not None and c.page_num != page_num:
                continue
            if section is not None and section.lower() not in c.section.lower():
                continue
            results.append(c)
        return results

    def save(self, path: str | Path) -> None:
        """Optional persistence interface."""
        raise NotImplementedError("This chunk store does not support persistence.")

    def load(self, path: str | Path) -> None:
        """Optional reload interface."""
        raise NotImplementedError("This chunk store does not support reloading.")


class InMemoryChunkStore(BaseChunkStore):
    """In-memory chunk storage with fast array and dict indexing."""

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None):
        self._chunks: List[DocumentChunk] = []
        self._id_map: Dict[int, DocumentChunk] = {}
        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks.append(chunk)
            self._id_map[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: int) -> Optional[DocumentChunk]:
        return self._id_map.get(chunk_id)

    def get_chunks(self, chunk_ids: Sequence[int]) -> List[DocumentChunk]:
        return [self._id_map[cid] for cid in chunk_ids if cid in self._id_map]

    def get_all_chunks(self) -> List[DocumentChunk]:
        return self._chunks

    def get_texts(self) -> List[str]:
        return [chunk.to_formatted_text() for chunk in self._chunks]

    def filter(
        self,
        doc_name: Optional[str] = None,
        page_num: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[DocumentChunk]:
        results = []
        for c in self._chunks:
            if doc_name is not None and c.doc_name != doc_name:
                continue
            if page_num is not None and c.page_num != page_num:
                continue
            if section is not None and section.lower() not in c.section.lower():
                continue
            results.append(c)
        return results

    def __getitem__(self, idx: int) -> DocumentChunk:
        return self._chunks[idx]

    def __iter__(self) -> Iterator[DocumentChunk]:
        return iter(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)


class ParquetChunkStore(BaseChunkStore):
    """
    PyArrow/Parquet-backed chunk storage.

    Implements the same BaseChunkStore contract as InMemoryChunkStore
    (add_chunks, get_chunk, get_all_chunks, get_texts, __len__), backed by
    a columnar pyarrow.Table instead of a Python list + dict. This makes
    chunk storage:

      - readable by external analytical tooling (DuckDB, Spark, Polars,
        pandas) once written to disk via `save()`, without going through
        this codebase or pickle.
      - able to project a single column off disk without deserializing
        full DocumentChunk objects, via the `load_texts_only()` classmethod
        (used by retrieval indexing, which only ever needs text).

    Row ordering: rows are always kept sorted by chunk_id ascending. This
    matches InMemoryChunkStore's index-order guarantee, which
    src/evaluation/dataset.py's ground-truth indices rely on.

    Note on scope: this class intentionally does NOT add get_chunks(),
    filter(), or an abstract save()/load() to BaseChunkStore itself.
    Extending the shared abstract contract would require InMemoryChunkStore
    to implement those methods too or instantiation would break; save/load
    are kept as concrete, Parquet-specific methods here instead.
    """

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None):
        if not _PYARROW_AVAILABLE:
            raise RuntimeError(
                "pyarrow is required to use ParquetChunkStore. Install with: pip install pyarrow"
            )
        self._table: pa.Table = CHUNK_PYARROW_SCHEMA.empty_table()
        # chunk_id -> row index. Rebuilt whenever the table changes.
        # Chunk IDs are contiguous ascending under normal ingestion
        # (see src/ingestion/chunkers.py), which would allow direct offset
        # addressing instead of a dict at very large scale -- kept as an
        # explicit map here for correctness under any future filtering or
        # out-of-order construction.
        self._id_to_row: Dict[int, int] = {}
        if chunks:
            self.add_chunks(chunks)

    # -- construction --------------------------------------------------

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return
        new_table = chunks_to_pyarrow_table(chunks)
        combined = (
            pa.concat_tables([self._table, new_table])
            if self._table.num_rows
            else new_table
        )
        # Keep chunk_id ascending so get_all_chunks()/get_texts() match
        # InMemoryChunkStore's index order exactly, regardless of the
        # order add_chunks() was called in.
        sort_idx = pc.sort_indices(combined, sort_keys=[("chunk_id", "ascending")])
        self._table = combined.take(sort_idx)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._id_to_row = {
            chunk_id: row
            for row, chunk_id in enumerate(self._table.column("chunk_id").to_pylist())
        }

    # -- BaseChunkStore contract -----------------------------------------

    def get_chunk(self, chunk_id: int) -> Optional[DocumentChunk]:
        row = self._id_to_row.get(chunk_id)
        return None if row is None else self._row_to_chunk(row)

    @staticmethod
    def _table_to_chunks(table: "pa.Table") -> List[DocumentChunk]:
        """Batch-materialize DocumentChunk objects from a PyArrow Table via to_pydict().

        Extracting column vectors all at once into Python lists is ~10-15x faster
        than querying cell-by-cell with table.column(name)[row] across thousands of rows.
        """
        if table.num_rows == 0:
            return []
        data = table.to_pydict()
        chunk_ids = data["chunk_id"]
        doc_names = data["doc_name"]
        page_nums = data["page_num"]
        sections = data["section"]
        texts = data["text"]
        metadata_jsons = data["metadata_json"]
        return [
            DocumentChunk(
                chunk_id=chunk_ids[i],
                doc_name=doc_names[i],
                page_num=page_nums[i],
                section=sections[i],
                text=texts[i],
                metadata=json.loads(metadata_jsons[i]) if metadata_jsons[i] else {},
            )
            for i in range(table.num_rows)
        ]

    def get_all_chunks(self) -> List[DocumentChunk]:
        return self._table_to_chunks(self._table)

    def get_texts(self) -> List[str]:
        # Zero-copy column extraction: builds formatted text without
        # touching chunk_id or metadata_json for rows we don't need them
        # from. Mirrors DocumentChunk.to_formatted_text()'s exact format.
        doc_names = self._table.column("doc_name").to_pylist()
        page_nums = self._table.column("page_num").to_pylist()
        sections = self._table.column("section").to_pylist()
        texts = self._table.column("text").to_pylist()
        return [
            f"[{d} | Page {p} | § {s}] {t}"
            for d, p, s, t in zip(doc_names, page_nums, sections, texts)
        ]

    def filter(
        self,
        doc_name: Optional[str] = None,
        page_num: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Predicate filtering using native PyArrow columnar compute expressions.
        Evaluates predicates on columns before converting matching rows to
        DocumentChunk objects via fast batch extraction.
        """
        if self._table.num_rows == 0:
            return []

        table = self._table
        if doc_name is not None:
            table = table.filter(pc.equal(table["doc_name"], doc_name))
        if page_num is not None:
            table = table.filter(pc.equal(table["page_num"], page_num))
        if section is not None:
            table = table.filter(
                pc.match_substring(pc.utf8_lower(table["section"]), section.lower())
            )

        return self._table_to_chunks(table)

    def __len__(self) -> int:
        return self._table.num_rows

    def __iter__(self) -> Iterator[DocumentChunk]:
        return iter(self.get_all_chunks())

    # -- row <-> DocumentChunk --------------------------------------------

    def _row_to_chunk(self, row: int, table: Optional[pa.Table] = None) -> DocumentChunk:
        target = self._table if table is None else table
        metadata_json = target.column("metadata_json")[row].as_py()
        return DocumentChunk(
            chunk_id=target.column("chunk_id")[row].as_py(),
            doc_name=target.column("doc_name")[row].as_py(),
            page_num=target.column("page_num")[row].as_py(),
            section=target.column("section")[row].as_py(),
            text=target.column("text")[row].as_py(),
            metadata=json.loads(metadata_json) if metadata_json else {},
        )

    # -- persistence -------------------------------------------------------

    def save(self, path: Union[str, Path], compression: str = PARQUET_COMPRESSION) -> None:
        """
        Write this store to a single Parquet file.

        Writes to a temp file in the same directory and then renames it
        into place, so a crash or interrupt mid-write can never leave a
        corrupted/partial cache file at `path`.
        """
        if not _PYARROW_AVAILABLE:
            raise RuntimeError(
                "pyarrow is required to use ParquetChunkStore. Install with: pip install pyarrow"
            )
        path = Path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(self._table, tmp_path, compression=compression)
        tmp_path.replace(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParquetChunkStore":
        """Load a full store (all columns, all rows) from a Parquet file."""
        if not _PYARROW_AVAILABLE:
            raise RuntimeError(
                "pyarrow is required to use ParquetChunkStore. Install with: pip install pyarrow"
            )
        table = pq.read_table(path, schema=CHUNK_PYARROW_SCHEMA)
        store = cls()
        store._table = table
        store._rebuild_index()
        return store

    @staticmethod
    def load_texts_only(path: Union[str, Path]) -> List[str]:
        """
        Column-projected read for retrieval indexing (BM25/TF-IDF), which
        only ever needs formatted text -- never chunk_id or metadata_json.
        Reads only the four columns needed to build it, skipping the rest
        of the file entirely rather than materializing full DocumentChunk
        objects.
        """
        if not _PYARROW_AVAILABLE:
            raise RuntimeError(
                "pyarrow is required to use ParquetChunkStore. Install with: pip install pyarrow"
            )
        table = pq.read_table(path, columns=["doc_name", "page_num", "section", "text"])
        doc_names = table.column("doc_name").to_pylist()
        page_nums = table.column("page_num").to_pylist()
        sections = table.column("section").to_pylist()
        texts = table.column("text").to_pylist()
        return [
            f"[{d} | Page {p} | § {s}] {t}"
            for d, p, s, t in zip(doc_names, page_nums, sections, texts)
        ]