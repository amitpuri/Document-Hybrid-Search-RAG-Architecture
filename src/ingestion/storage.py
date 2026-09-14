"""
Storage Abstractions for Document Chunks and Metadata.
Enables decoupling retrieval from storage, supporting big-data persistence or in-memory stores.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Iterator
from src.common.types import DocumentChunk


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

    def get_all_chunks(self) -> List[DocumentChunk]:
        return self._chunks

    def get_texts(self) -> List[str]:
        return [chunk.to_formatted_text() for chunk in self._chunks]

    def __getitem__(self, idx: int) -> DocumentChunk:
        return self._chunks[idx]

    def __iter__(self) -> Iterator[DocumentChunk]:
        return iter(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)
