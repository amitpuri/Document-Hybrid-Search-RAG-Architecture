"""
Ingestion Pipeline orchestrator.
Handles PDF discovery, batch extraction, structured chunking, incremental
partitioned Parquet dataset management, caching, and storage abstractions.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.common.types import DocumentChunk
from src.config import (
    CACHE_DIR,
    CORPUS_DIR,
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP_SENTENCES,
    DEFAULT_STORAGE_BACKEND,
)
from src.ingestion.cache import (
    atomic_write_parquet_table,
    compute_cache_key,
    compute_chunking_regime_hash,
    load_cached_chunks,
    load_dataset_manifest,
    save_cached_chunks,
    save_dataset_manifest_atomic,
)
from src.ingestion.chunkers import chunk_document_structured
from src.ingestion.extractors import extract_text_from_pdf
from src.ingestion.storage import (
    BaseChunkStore,
    InMemoryChunkStore,
    ParquetChunkStore,
    QdrantChunkStore,
    chunks_to_pyarrow_table,
)


class IngestionRegimeMismatchError(ValueError):
    """Raised when an incremental ingestion batch has conflicting chunking parameters."""

    pass


_CHUNK_PREFIX_REGEX = re.compile(
    r"^\[(?P<doc>[^|]+)\s*\|\s*Page\s*(?P<page>\d+)\s*\|\s*"
    r"§\s*(?P<section>[^\]]+)\]\s*(?P<text>.*)$",
    re.DOTALL,
)


def parse_formatted_chunk_string(formatted_str: str, chunk_id: int) -> DocumentChunk:
    """Parses a legacy or serialized formatted chunk string back into a DocumentChunk."""
    match = _CHUNK_PREFIX_REGEX.match(formatted_str.strip())
    if match:
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_name=match.group("doc").strip(),
            page_num=int(match.group("page").strip()),
            section=match.group("section").strip(),
            text=match.group("text").strip(),
        )
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_name="Unknown",
        page_num=1,
        section="Overview",
        text=formatted_str,
    )


class IngestionPipeline:
    """
    Orchestrates the document ingestion pipeline:
    1. Discovers PDF files in the corpus directory.
    2. Enforces parameter regime consistency across incremental batches.
    3. Manages partitioned Apache Parquet dataset storage and caching.
    4. Extracts text page-by-page with fallback support.
    5. Segments text into sentence-aware, section-preserving chunks.
    6. Populates and returns a decoupled chunk store (ParquetChunkStore or InMemoryChunkStore).
    """

    def __init__(
        self,
        corpus_dir: Union[str, Path] = CORPUS_DIR,
        cache_dir: Union[str, Path] = CACHE_DIR,
        max_words: int = DEFAULT_MAX_WORDS,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
        storage_backend: str = DEFAULT_STORAGE_BACKEND,
        dataset_dir: Optional[Union[str, Path]] = None,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.cache_dir = Path(cache_dir)
        self.max_words = max_words
        self.overlap_sentences = overlap_sentences
        self.storage_backend = storage_backend.lower()
        self.dataset_dir = Path(dataset_dir) if dataset_dir else (self.cache_dir / "chunks_dataset")

    def run(self, force_rebuild: bool = False) -> Tuple[BaseChunkStore, int, List[str]]:
        """
        Executes the ingestion pipeline.

        Returns:
            Tuple: (chunk_store, total_pages, list_of_pdf_paths)
        """
        if self.storage_backend == "parquet":
            return self._run_parquet(force_rebuild=force_rebuild)
        elif self.storage_backend == "qdrant":
            return self._run_qdrant(force_rebuild=force_rebuild)
        else:
            return self._run_memory(force_rebuild=force_rebuild)

    def _run_parquet(self, force_rebuild: bool = False) -> Tuple[ParquetChunkStore, int, List[str]]:
        """Executes partitioned Parquet dataset ingestion with incremental appending."""
        current_regime_hash = compute_chunking_regime_hash(self.max_words, self.overlap_sentences)
        manifest = load_dataset_manifest(self.dataset_dir) if not force_rebuild else None

        pdf_paths = sorted([str(p) for p in self.corpus_dir.glob("*.pdf")])
        current_pdf_map = {os.path.basename(p): p for p in pdf_paths}

        if manifest is not None:
            # 1. Chunking regime check
            saved_regime = manifest.get("chunking_regime", {})
            saved_hash = saved_regime.get("regime_hash")
            if saved_hash and saved_hash != current_regime_hash:
                msg = (
                    "Cannot incrementally ingest into dataset with mismatched "
                    "chunking parameters. "
                    f"Dataset regime: {saved_regime}; "
                    f"Requested: max_words={self.max_words}, "
                    f"overlap_sentences={self.overlap_sentences}. "
                    "Specify force_rebuild=True to re-chunk the entire corpus "
                    "uniformly, or use matching parameters."
                )
                raise IngestionRegimeMismatchError(msg)

            # 2. Incremental diffing against manifest
            partitions = manifest.get("partitions", {})
            unchanged_docs: List[str] = []
            new_or_modified_docs: List[str] = []

            for doc_name, path_str in current_pdf_map.items():
                if doc_name in partitions:
                    try:
                        stat = os.stat(path_str)
                        part_info = partitions[doc_name]
                        if (
                            stat.st_mtime == part_info.get("mtime")
                            and stat.st_size == part_info.get("size")
                            and (self.dataset_dir / part_info.get("partition_path", "")).exists()
                        ):
                            unchanged_docs.append(doc_name)
                            continue
                    except OSError:
                        pass
                new_or_modified_docs.append(doc_name)

            # If all PDFs are intact and unchanged, return instantly from dataset
            if len(new_or_modified_docs) == 0 and len(unchanged_docs) == len(current_pdf_map):
                store = ParquetChunkStore.from_dataset(self.dataset_dir)
                total_pages = manifest.get("total_pages", 0)
                return store, total_pages, pdf_paths

            # 3. Incremental update: ingest only new or modified PDFs
            next_chunk_id = manifest.get("next_chunk_id", manifest.get("total_chunks", 0))
            total_pages = manifest.get("total_pages", 0)

            for doc_name in new_or_modified_docs:
                path = current_pdf_map[doc_name]
                pages = extract_text_from_pdf(path)
                total_pages += len(pages)

                doc_chunks = chunk_document_structured(
                    doc_name=doc_name,
                    pages=pages,
                    start_chunk_id=next_chunk_id,
                    max_words=self.max_words,
                    overlap_sentences=self.overlap_sentences,
                )

                if doc_chunks:
                    table = chunks_to_pyarrow_table(doc_chunks)
                    part_rel_path = f"doc={doc_name}/data_0000.parquet"
                    part_dest = self.dataset_dir / part_rel_path
                    atomic_write_parquet_table(table, part_dest)

                    stat = os.stat(path)
                    partitions[doc_name] = {
                        "partition_path": part_rel_path,
                        "chunk_id_min": doc_chunks[0].chunk_id,
                        "chunk_id_max": doc_chunks[-1].chunk_id,
                        "chunk_count": len(doc_chunks),
                        "total_pages": len(pages),
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                    }
                    next_chunk_id += len(doc_chunks)

            manifest["next_chunk_id"] = next_chunk_id
            manifest["total_chunks"] = next_chunk_id
            manifest["total_pages"] = total_pages
            manifest["partitions"] = partitions
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_dataset_manifest_atomic(self.dataset_dir, manifest)

            store = ParquetChunkStore.from_dataset(self.dataset_dir)
            return store, total_pages, pdf_paths

        # 4. Clean Rebuild
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        partitions_map: Dict[str, Any] = {}
        next_chunk_id = 0
        total_pages = 0

        for path in pdf_paths:
            doc_name = os.path.basename(path)
            pages = extract_text_from_pdf(path)
            total_pages += len(pages)

            doc_chunks = chunk_document_structured(
                doc_name=doc_name,
                pages=pages,
                start_chunk_id=next_chunk_id,
                max_words=self.max_words,
                overlap_sentences=self.overlap_sentences,
            )

            if doc_chunks:
                table = chunks_to_pyarrow_table(doc_chunks)
                part_rel_path = f"doc={doc_name}/data_0000.parquet"
                part_dest = self.dataset_dir / part_rel_path
                atomic_write_parquet_table(table, part_dest)

                stat = os.stat(path)
                partitions_map[doc_name] = {
                    "partition_path": part_rel_path,
                    "chunk_id_min": doc_chunks[0].chunk_id,
                    "chunk_id_max": doc_chunks[-1].chunk_id,
                    "chunk_count": len(doc_chunks),
                    "total_pages": len(pages),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
                next_chunk_id += len(doc_chunks)

        manifest_data = {
            "version": "2.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "next_chunk_id": next_chunk_id,
            "total_chunks": next_chunk_id,
            "total_pages": total_pages,
            "chunking_regime": {
                "max_words": self.max_words,
                "overlap_sentences": self.overlap_sentences,
                "regime_hash": current_regime_hash,
            },
            "partitions": partitions_map,
        }
        save_dataset_manifest_atomic(self.dataset_dir, manifest_data)

        store = ParquetChunkStore.from_dataset(self.dataset_dir)
        return store, total_pages, pdf_paths

    def _run_memory(self, force_rebuild: bool = False) -> Tuple[InMemoryChunkStore, int, List[str]]:
        """Legacy in-memory pipeline path for fast testing or backward compatibility."""
        # If Parquet dataset already exists, load chunks directly into InMemoryChunkStore
        manifest = load_dataset_manifest(self.dataset_dir)
        if not force_rebuild and manifest is not None and self.dataset_dir.exists():
            pq_store = ParquetChunkStore.from_dataset(self.dataset_dir)
            parquet_chunks = pq_store.get_all_chunks()
            pdf_paths = sorted([str(p) for p in self.corpus_dir.glob("*.pdf")])
            return (
                InMemoryChunkStore(parquet_chunks),
                manifest.get("total_pages", 0),
                pdf_paths,
            )

        # Check legacy pickle cache
        cache_key = compute_cache_key(self.corpus_dir, self.max_words, self.overlap_sentences)
        cache_file = self.cache_dir / f"structured_cache_{cache_key}.pkl"

        if not force_rebuild and cache_file.exists():
            cached_data = load_cached_chunks(cache_file)
            if cached_data is not None:
                raw_chunks = cached_data.get("chunks", [])
                total_pages = cached_data.get("total_pages", 0)
                pdf_paths = cached_data.get("pdf_paths", [])

                cached_chunks: List[DocumentChunk] = []
                for idx, item in enumerate(raw_chunks):
                    if isinstance(item, DocumentChunk):
                        cached_chunks.append(item)
                    else:
                        cached_chunks.append(parse_formatted_chunk_string(str(item), idx))

                return InMemoryChunkStore(cached_chunks), total_pages, pdf_paths

        # Full extraction
        pdf_paths = sorted([str(p) for p in self.corpus_dir.glob("*.pdf")])
        extracted_chunks: List[DocumentChunk] = []
        total_pages = 0
        current_id = 0

        for path in pdf_paths:
            doc_name = os.path.basename(path)
            pages = extract_text_from_pdf(path)
            total_pages += len(pages)
            doc_chunks = chunk_document_structured(
                doc_name=doc_name,
                pages=pages,
                start_chunk_id=current_id,
                max_words=self.max_words,
                overlap_sentences=self.overlap_sentences,
            )
            extracted_chunks.extend(doc_chunks)
            current_id += len(doc_chunks)

        store = InMemoryChunkStore(extracted_chunks)
        save_data = {
            "chunks": store.get_texts(),
            "total_pages": total_pages,
            "pdf_paths": pdf_paths,
        }
        save_cached_chunks(cache_file, save_data)
        return store, total_pages, pdf_paths

    def _run_qdrant(self, force_rebuild: bool = False) -> Tuple[QdrantChunkStore, int, List[str]]:
        """Qdrant-backed ingestion pipeline storing chunks and metadata in local Qdrant."""
        store = QdrantChunkStore(recreate=force_rebuild)
        pdf_paths = sorted([str(p) for p in self.corpus_dir.glob("*.pdf")])

        # If Qdrant collection is already populated and not force_rebuild, return immediately
        if not force_rebuild and len(store) > 0:
            manifest = load_dataset_manifest(self.dataset_dir)
            total_pages = manifest.get("total_pages", 0) if manifest else 0
            return store, total_pages, pdf_paths

        # If Parquet dataset exists, load chunks directly to avoid re-extracting PDFs
        manifest = load_dataset_manifest(self.dataset_dir)
        if not force_rebuild and manifest is not None and self.dataset_dir.exists():
            pq_store = ParquetChunkStore.from_dataset(self.dataset_dir)
            chunks = pq_store.get_all_chunks()
            total_pages = manifest.get("total_pages", 0)
        else:
            # Full PDF extraction
            chunks = []
            total_pages = 0
            current_id = 0
            for path in pdf_paths:
                doc_name = os.path.basename(path)
                pages = extract_text_from_pdf(path)
                total_pages += len(pages)
                doc_chunks = chunk_document_structured(
                    doc_name=doc_name,
                    pages=pages,
                    start_chunk_id=current_id,
                    max_words=self.max_words,
                    overlap_sentences=self.overlap_sentences,
                )
                chunks.extend(doc_chunks)
                current_id += len(doc_chunks)

        # Upsert chunks into Qdrant collection
        store.add_chunks(chunks)
        return store, total_pages, pdf_paths
