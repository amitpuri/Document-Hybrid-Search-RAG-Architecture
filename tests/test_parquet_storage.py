"""
Comprehensive Unit and Integration Tests for Decoupled Parquet Storage Layer.
Covers metadata preservation, monotonic insertion-order IDs, regime guard,
projection pushdown, predicate filtering, and atomic write safety.
"""

import os
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

from src.common.types import DocumentChunk
from src.ingestion.storage import (
    InMemoryChunkStore,
    ParquetChunkStore,
    CHUNK_PYARROW_SCHEMA,
    chunks_to_pyarrow_table,
    pyarrow_row_to_chunk,
)
from src.ingestion.cache import (
    compute_chunking_regime_hash,
    load_dataset_manifest,
    save_dataset_manifest_atomic,
    atomic_write_parquet_table,
)
from src.ingestion.pipeline import (
    IngestionPipeline,
    IngestionRegimeMismatchError,
)


class TestParquetStorage(unittest.TestCase):
    """Test suite for ParquetChunkStore and decoupled storage abstractions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Sample chunks with rich metadata
        self.sample_chunks = [
            DocumentChunk(
                chunk_id=0,
                doc_name="paper_alpha.pdf",
                page_num=1,
                section="Abstract",
                text="Alpha paper investigates POMDP algorithms.",
                metadata={"doi": "10.1000/alpha", "score": 0.95, "tags": ["pomdp", "planning"]},
            ),
            DocumentChunk(
                chunk_id=1,
                doc_name="paper_alpha.pdf",
                page_num=2,
                section="Introduction",
                text="The state of the art in belief state filtering.",
                metadata={"doi": "10.1000/alpha", "section_num": 1},
            ),
            DocumentChunk(
                chunk_id=2,
                doc_name="paper_gamma.pdf",
                page_num=1,
                section="Abstract",
                text="Gamma paper analyzes LLM execution harnesses.",
                metadata={"doi": "10.1000/gamma", "experimental": True},
            ),
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metadata_preservation_bug_fix(self):
        """Verifies that rich chunk metadata is preserved across Parquet serialization and reload."""
        store = ParquetChunkStore.from_chunks(self.sample_chunks)
        parquet_file = self.root_path / "test_chunks.parquet"
        store.save(parquet_file)

        # Reload store from Parquet file
        loaded_store = ParquetChunkStore()
        loaded_store.load(parquet_file)

        self.assertEqual(len(loaded_store), 3)
        chunk_0 = loaded_store.get_chunk(0)
        self.assertIsNotNone(chunk_0)
        self.assertEqual(chunk_0.metadata, {"doi": "10.1000/alpha", "score": 0.95, "tags": ["pomdp", "planning"]})
        self.assertEqual(chunk_0.section, "Abstract")

        chunk_2 = loaded_store.get_chunk(2)
        self.assertIsNotNone(chunk_2)
        self.assertEqual(chunk_2.metadata, {"doi": "10.1000/gamma", "experimental": True})

    def test_projection_pushdown_get_texts(self):
        """Verifies that get_texts() extracts formatted text accurately."""
        store = ParquetChunkStore.from_chunks(self.sample_chunks)
        texts = store.get_texts()
        self.assertEqual(len(texts), 3)
        self.assertEqual(texts[0], self.sample_chunks[0].to_formatted_text())
        self.assertEqual(texts[1], self.sample_chunks[1].to_formatted_text())
        self.assertEqual(texts[2], self.sample_chunks[2].to_formatted_text())

    def test_batch_get_chunks_ordering(self):
        """Verifies that get_chunks returns chunks in the exact requested sequence order."""
        store = ParquetChunkStore.from_chunks(self.sample_chunks)
        batch = store.get_chunks([2, 0])
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].chunk_id, 2)
        self.assertEqual(batch[1].chunk_id, 0)
        self.assertEqual(batch[0].doc_name, "paper_gamma.pdf")
        self.assertEqual(batch[1].doc_name, "paper_alpha.pdf")

    def test_predicate_filtering(self):
        """Verifies predicate filtering by doc_name, page_num, and section."""
        store = ParquetChunkStore.from_chunks(self.sample_chunks)

        # Filter by doc_name
        alpha_chunks = store.filter(doc_name="paper_alpha.pdf")
        self.assertEqual(len(alpha_chunks), 2)
        self.assertTrue(all(c.doc_name == "paper_alpha.pdf" for c in alpha_chunks))

        # Filter by page_num
        page_2_chunks = store.filter(page_num=2)
        self.assertEqual(len(page_2_chunks), 1)
        self.assertEqual(page_2_chunks[0].chunk_id, 1)

        # Filter by section
        abstract_chunks = store.filter(section="Abstract")
        self.assertEqual(len(abstract_chunks), 2)
        self.assertEqual({c.chunk_id for c in abstract_chunks}, {0, 2})

    def test_monotonic_insertion_order_partitioned_dataset(self):
        """Verifies that incremental additions preserve existing chunk IDs regardless of filename sorting."""
        dataset_dir = self.root_path / "dataset"
        dataset_dir.mkdir(parents=True)

        # Step 1: Ingest docA (chunks 0, 1) and docC (chunks 2, 3)
        chunks_a = [
            DocumentChunk(chunk_id=0, doc_name="docA.pdf", page_num=1, section="Sec1", text="Doc A chunk 0"),
            DocumentChunk(chunk_id=1, doc_name="docA.pdf", page_num=2, section="Sec2", text="Doc A chunk 1"),
        ]
        chunks_c = [
            DocumentChunk(chunk_id=2, doc_name="docC.pdf", page_num=1, section="Sec1", text="Doc C chunk 2"),
            DocumentChunk(chunk_id=3, doc_name="docC.pdf", page_num=2, section="Sec2", text="Doc C chunk 3"),
        ]

        part_a_dir = dataset_dir / "doc=docA.pdf"
        part_c_dir = dataset_dir / "doc=docC.pdf"
        atomic_write_parquet_table(chunks_to_pyarrow_table(chunks_a), part_a_dir / "data_0000.parquet")
        atomic_write_parquet_table(chunks_to_pyarrow_table(chunks_c), part_c_dir / "data_0000.parquet")

        manifest = {
            "version": "2.0.0",
            "next_chunk_id": 4,
            "total_chunks": 4,
            "total_pages": 4,
            "chunking_regime": {
                "max_words": 120,
                "overlap_sentences": 1,
                "regime_hash": compute_chunking_regime_hash(120, 1),
            },
            "partitions": {
                "docA.pdf": {
                    "partition_path": "doc=docA.pdf/data_0000.parquet",
                    "chunk_id_min": 0,
                    "chunk_id_max": 1,
                    "chunk_count": 2,
                    "mtime": 1000,
                    "size": 100,
                },
                "docC.pdf": {
                    "partition_path": "doc=docC.pdf/data_0000.parquet",
                    "chunk_id_min": 2,
                    "chunk_id_max": 3,
                    "chunk_count": 2,
                    "mtime": 1000,
                    "size": 100,
                },
            },
        }
        save_dataset_manifest_atomic(dataset_dir, manifest)

        # Step 2: Ingest docB (which sorts alphabetically between docA and docC)
        # In an insertion-order model, docB MUST receive chunk IDs 4 and 5 (never 2 and 3)
        chunks_b = [
            DocumentChunk(chunk_id=4, doc_name="docB.pdf", page_num=1, section="Sec1", text="Doc B chunk 4"),
            DocumentChunk(chunk_id=5, doc_name="docB.pdf", page_num=2, section="Sec2", text="Doc B chunk 5"),
        ]
        part_b_dir = dataset_dir / "doc=docB.pdf"
        atomic_write_parquet_table(chunks_to_pyarrow_table(chunks_b), part_b_dir / "data_0000.parquet")

        manifest["partitions"]["docB.pdf"] = {
            "partition_path": "doc=docB.pdf/data_0000.parquet",
            "chunk_id_min": 4,
            "chunk_id_max": 5,
            "chunk_count": 2,
            "mtime": 2000,
            "size": 100,
        }
        manifest["next_chunk_id"] = 6
        manifest["total_chunks"] = 6
        save_dataset_manifest_atomic(dataset_dir, manifest)

        # Step 3: Verify with ParquetChunkStore
        store = ParquetChunkStore.from_dataset(dataset_dir)
        self.assertEqual(len(store), 6)

        # Verify docA and docC IDs are completely unchanged
        c0 = store.get_chunk(0)
        c2 = store.get_chunk(2)
        c4 = store.get_chunk(4)

        self.assertEqual(c0.doc_name, "docA.pdf")
        self.assertEqual(c2.doc_name, "docC.pdf")
        self.assertEqual(c4.doc_name, "docB.pdf")

    def test_chunking_regime_guard(self):
        """Verifies that attempting incremental ingestion with mismatched chunking parameters raises an error."""
        dataset_dir = self.root_path / "dataset"
        dataset_dir.mkdir(parents=True)

        manifest = {
            "version": "2.0.0",
            "next_chunk_id": 10,
            "total_chunks": 10,
            "total_pages": 5,
            "chunking_regime": {
                "max_words": 120,
                "overlap_sentences": 1,
                "regime_hash": compute_chunking_regime_hash(120, 1),
            },
            "partitions": {},
        }
        save_dataset_manifest_atomic(dataset_dir, manifest)

        pipeline = IngestionPipeline(
            corpus_dir=self.root_path,
            cache_dir=self.root_path,
            max_words=200,  # Conflict: 200 vs 120
            overlap_sentences=1,
            dataset_dir=dataset_dir,
            storage_backend="parquet",
        )

        with self.assertRaises(IngestionRegimeMismatchError):
            pipeline.run(force_rebuild=False)

    def test_atomic_write_safety(self):
        """Verifies that atomic write produces valid destination files and cleans temporary files."""
        dest_file = self.root_path / "atomic_test.parquet"
        table = chunks_to_pyarrow_table(self.sample_chunks)

        atomic_write_parquet_table(table, dest_file)
        self.assertTrue(dest_file.exists())

        # Verify no .tmp files linger in directory
        tmp_files = list(self.root_path.glob("*.tmp_*"))
        self.assertEqual(len(tmp_files), 0)


if __name__ == "__main__":
    unittest.main()
