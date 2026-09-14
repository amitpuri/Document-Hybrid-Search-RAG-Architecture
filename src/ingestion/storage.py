"""
Storage Abstractions for Document Chunks and Metadata.
Enables decoupling retrieval from storage, supporting big-data persistence,
partitioned Parquet datasets, or in-memory stores.
"""

import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Sequence, Iterator, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc

from src.common.types import DocumentChunk
from src.config import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_DEFAULT_ROW_GROUP_SIZE,
    PARQUET_IN_MEMORY_THRESHOLD,
)

CHUNK_PYARROW_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.int64(), nullable=False),
    pa.field("doc_name", pa.string(), nullable=False),
    pa.field("page_num", pa.int32(), nullable=False),
    pa.field("section", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("formatted_text", pa.string(), nullable=False),
    pa.field("metadata_json", pa.string(), nullable=True),
])


def chunks_to_pyarrow_table(chunks: Sequence[DocumentChunk]) -> pa.Table:
    """Converts a sequence of DocumentChunk objects into a PyArrow Table adhering to CHUNK_PYARROW_SCHEMA."""
    if not chunks:
        return pa.Table.from_batches([], schema=CHUNK_PYARROW_SCHEMA)

    chunk_ids = [int(c.chunk_id) for c in chunks]
    doc_names = [str(c.doc_name) for c in chunks]
    page_nums = [int(c.page_num) for c in chunks]
    sections = [str(c.section) for c in chunks]
    texts = [str(c.text) for c in chunks]
    formatted_texts = [c.to_formatted_text() for c in chunks]
    metadata_jsons = [
        json.dumps(c.metadata, ensure_ascii=False) if c.metadata else None
        for c in chunks
    ]

    return pa.Table.from_arrays(
        [
            pa.array(chunk_ids, type=pa.int64()),
            pa.array(doc_names, type=pa.string()),
            pa.array(page_nums, type=pa.int32()),
            pa.array(sections, type=pa.string()),
            pa.array(texts, type=pa.string()),
            pa.array(formatted_texts, type=pa.string()),
            pa.array(metadata_jsons, type=pa.string()),
        ],
        schema=CHUNK_PYARROW_SCHEMA,
    )


def pyarrow_row_to_chunk(row: Dict[str, Any]) -> DocumentChunk:
    """Reconstructs a DocumentChunk from a dictionary representing a PyArrow row."""
    meta_raw = row.get("metadata_json")
    metadata: Dict[str, Any] = {}
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
    Big-data ready Apache Parquet Chunk Store.
    Supports in-memory Arrow tables, memory-mapped out-of-core dataset scanning,
    projection pushdown (loading only formatted text columns), and partition pruning.
    """

    def __init__(
        self,
        table: Optional[pa.Table] = None,
        dataset: Optional[ds.Dataset] = None,
        dataset_dir: Optional[str | Path] = None,
        in_memory_threshold: int = PARQUET_IN_MEMORY_THRESHOLD,
    ):
        self._table: Optional[pa.Table] = table
        self._dataset: Optional[ds.Dataset] = dataset
        self.dataset_dir: Optional[Path] = Path(dataset_dir) if dataset_dir else None
        self.in_memory_threshold = in_memory_threshold

        # Initialize dataset if directory provided
        if self.dataset_dir and self.dataset_dir.exists() and self._dataset is None and self._table is None:
            self._init_from_path(self.dataset_dir)

    def _init_from_path(self, path: Path) -> None:
        """Initializes store from a parquet file or partitioned dataset directory."""
        if path.is_dir():
            self._dataset = ds.dataset(
                str(path),
                format="parquet",
                schema=CHUNK_PYARROW_SCHEMA,
                ignore_prefixes=["_", "."],
            )
            num_rows = self._dataset.count_rows()
            if num_rows <= self.in_memory_threshold:
                # Load in-memory table for zero-copy high-throughput access
                self._table = self._dataset.to_table()
        else:
            self._table = pq.read_table(str(path), schema=CHUNK_PYARROW_SCHEMA)

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[DocumentChunk],
        in_memory_threshold: int = PARQUET_IN_MEMORY_THRESHOLD,
    ) -> "ParquetChunkStore":
        """Constructs a ParquetChunkStore directly from a list of DocumentChunks."""
        table = chunks_to_pyarrow_table(chunks)
        return cls(table=table, in_memory_threshold=in_memory_threshold)

    @classmethod
    def from_dataset(
        cls,
        dataset_dir: str | Path,
        in_memory_threshold: int = PARQUET_IN_MEMORY_THRESHOLD,
    ) -> "ParquetChunkStore":
        """Loads a ParquetChunkStore from a partitioned dataset directory."""
        return cls(dataset_dir=dataset_dir, in_memory_threshold=in_memory_threshold)

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Appends new chunks to the in-memory Arrow table."""
        new_table = chunks_to_pyarrow_table(chunks)
        if self._table is None:
            self._table = new_table
        else:
            self._table = pa.concat_tables([self._table, new_table])

    def get_chunk(self, chunk_id: int) -> Optional[DocumentChunk]:
        """Retrieves a single chunk by chunk_id using direct offset lookup or compute filter."""
        if self._table is not None:
            total_rows = len(self._table)
            if 0 <= chunk_id < total_rows:
                cid_val = self._table.column("chunk_id")[chunk_id].as_py()
                if cid_val == chunk_id:
                    row_dict = {
                        name: self._table.column(name)[chunk_id].as_py()
                        for name in self._table.column_names
                    }
                    return pyarrow_row_to_chunk(row_dict)

            # Fallback for non-contiguous offset within table
            filtered = self._table.filter(pc.equal(self._table["chunk_id"], chunk_id))
            if len(filtered) > 0:
                row_dict = {
                    name: filtered.column(name)[0].as_py()
                    for name in filtered.column_names
                }
                return pyarrow_row_to_chunk(row_dict)
            return None

        elif self._dataset is not None:
            # Dataset scanner with partition pruning via chunk_id min/max statistics
            scanner = self._dataset.scanner(filter=pc.equal(pc.field("chunk_id"), chunk_id))
            res_table = scanner.to_table()
            if len(res_table) > 0:
                row_dict = {
                    name: res_table.column(name)[0].as_py()
                    for name in res_table.column_names
                }
                return pyarrow_row_to_chunk(row_dict)
            return None

        return None

    def get_chunks(self, chunk_ids: Sequence[int]) -> List[DocumentChunk]:
        """Batch lookup of multiple chunks in the specified sequence order."""
        if not chunk_ids:
            return []

        if self._table is not None:
            total_rows = len(self._table)
            can_direct_index = all(0 <= cid < total_rows for cid in chunk_ids)
            if can_direct_index:
                cid_col = self._table.column("chunk_id")
                if all(cid_col[cid].as_py() == cid for cid in chunk_ids):
                    results = []
                    for cid in chunk_ids:
                        row_dict = {
                            name: self._table.column(name)[cid].as_py()
                            for name in self._table.column_names
                        }
                        results.append(pyarrow_row_to_chunk(row_dict))
                    return results

            filtered = self._table.filter(pc.is_in(self._table["chunk_id"], value_set=pa.array(chunk_ids, type=pa.int64())))
            chunk_dict: Dict[int, DocumentChunk] = {}
            for i in range(len(filtered)):
                row = {name: filtered.column(name)[i].as_py() for name in filtered.column_names}
                chunk_dict[row["chunk_id"]] = pyarrow_row_to_chunk(row)
            return [chunk_dict[cid] for cid in chunk_ids if cid in chunk_dict]

        elif self._dataset is not None:
            scanner = self._dataset.scanner(
                filter=pc.is_in(pc.field("chunk_id"), value_set=pa.array(chunk_ids, type=pa.int64()))
            )
            table = scanner.to_table()
            chunk_dict = {}
            for i in range(len(table)):
                row = {name: table.column(name)[i].as_py() for name in table.column_names}
                chunk_dict[row["chunk_id"]] = pyarrow_row_to_chunk(row)
            return [chunk_dict[cid] for cid in chunk_ids if cid in chunk_dict]

        return []

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Returns all chunks sorted by chunk_id."""
        table = self._table
        if table is None and self._dataset is not None:
            table = self._dataset.to_table()

        if table is None or len(table) == 0:
            return []

        sort_indices = pc.sort_indices(table["chunk_id"])
        sorted_table = table.take(sort_indices)

        chunks: List[DocumentChunk] = []
        for i in range(len(sorted_table)):
            row = {name: sorted_table.column(name)[i].as_py() for name in sorted_table.column_names}
            chunks.append(pyarrow_row_to_chunk(row))
        return chunks

    def get_texts(self) -> List[str]:
        """
        Extracts formatted chunk text representation.
        Uses column projection pushdown to load ONLY the formatted_text column,
        avoiding deserializing full chunk text and metadata.
        """
        if self._table is not None:
            return [str(x) for x in self._table.column("formatted_text").to_pylist()]

        if self._dataset is not None:
            text_table = self._dataset.scanner(columns=["chunk_id", "formatted_text"]).to_table()
            sort_indices = pc.sort_indices(text_table["chunk_id"])
            sorted_texts = text_table.column("formatted_text").take(sort_indices)
            return [str(x) for x in sorted_texts.to_pylist()]

        return []

    def filter(
        self,
        doc_name: Optional[str] = None,
        page_num: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Filters chunks using PyArrow compute expressions and dataset partition pruning.
        """
        expr = None
        if doc_name is not None:
            expr = (pc.field("doc_name") == doc_name)
        if page_num is not None:
            cond = (pc.field("page_num") == page_num)
            expr = cond if expr is None else (expr & cond)
        if section is not None:
            cond = pc.match_substring(pc.utf8_lower(pc.field("section")), section.lower())
            expr = cond if expr is None else (expr & cond)

        if expr is None:
            return self.get_all_chunks()

        if self._table is not None:
            filtered_table = self._table.filter(expr)
        elif self._dataset is not None:
            filtered_table = self._dataset.scanner(filter=expr).to_table()
        else:
            return []

        sort_indices = pc.sort_indices(filtered_table["chunk_id"])
        sorted_filtered = filtered_table.take(sort_indices)

        results = []
        for i in range(len(sorted_filtered)):
            row = {name: sorted_filtered.column(name)[i].as_py() for name in sorted_filtered.column_names}
            results.append(pyarrow_row_to_chunk(row))
        return results

    def save(
        self,
        path: str | Path,
        compression: str = PARQUET_COMPRESSION,
        compression_level: int = PARQUET_COMPRESSION_LEVEL,
        row_group_size: int = PARQUET_DEFAULT_ROW_GROUP_SIZE,
    ) -> None:
        """Saves current table to a single Parquet file."""
        if self._table is None:
            raise ValueError("No in-memory table to save.")
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            self._table,
            dest_path,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
        )

    def load(self, path: str | Path) -> None:
        """Loads chunk store from a Parquet file or dataset directory."""
        self._init_from_path(Path(path))

    def __len__(self) -> int:
        if self._table is not None:
            return len(self._table)
        if self._dataset is not None:
            return self._dataset.count_rows()
        return 0

    def __iter__(self) -> Iterator[DocumentChunk]:
        return iter(self.get_all_chunks())
