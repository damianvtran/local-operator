import json
import os
from pathlib import Path
from typing import List

import numpy as np
from faiss import IndexFlatL2, read_index, write_index
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class RAGException(Exception):
    """Base exception class for RAG-related errors."""

    pass


class InsightResult(BaseModel):
    """Model for storing an insight result with its distance score."""

    insight: str
    distance: float


class EmbeddingManager:
    """Manages embeddings for a retrieval-augmented generation (RAG) system.

    This class handles the storage, indexing, and retrieval of text embeddings using FAISS.
    It maintains both the vector index and associated metadata, providing methods to add new
    insights and query existing ones.

    Args:
        file_path (str): Directory path where index and metadata files will be stored
        embedding_dim (int, optional): Dimension of embeddings. Defaults to 384.
        model_name (str, optional): Name of the sentence transformer model.
        Defaults to "all-MiniLM-L6-v2".

    Attributes:
        index_path (str): Path to the FAISS index file
        metadata_path (str): Path to the metadata JSON file
        embedding_dim (int): Dimension of the embeddings
        model (SentenceTransformer): Model used for text embedding
        index (faiss.Index): FAISS index for similarity search
        metadata (list): List of stored text insights

    Raises:
        RAGException: If there are issues with file operations or model loading
        ValueError: If invalid parameters are provided
    """

    index_path: Path
    metadata_path: Path
    embedding_dim: int
    model: SentenceTransformer
    index: IndexFlatL2
    metadata: list[str]

    def __init__(
        self,
        file_path: Path,
        embedding_dim: int = 384,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        try:
            os.makedirs(file_path, exist_ok=True)
            self.index_path = file_path / "rag_index.index"
            self.metadata_path = file_path / "rag_metadata.jsonl"

            self.embedding_dim = embedding_dim
            self.model = SentenceTransformer(model_name)

            # Load existing index and metadata if available; otherwise, create new ones.
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                try:
                    self.index = read_index(str(self.index_path))
                    self.metadata = []
                    with open(self.metadata_path, "r") as f:
                        for line in f:
                            if line.strip():
                                self.metadata.append(json.loads(line)["text"])
                except (RuntimeError, json.JSONDecodeError, IOError) as e:
                    raise RAGException(f"Failed to load existing index/metadata: {str(e)}")
            else:
                self.index = IndexFlatL2(self.embedding_dim)
                self.metadata = []  # List to store insights (as plain text).
                self.save()  # Creates the initial index and metadata file.
        except Exception as e:
            raise RAGException(f"Failed to initialize EmbeddingManager: {str(e)}")

    def add_insight(self, insight: str) -> None:
        """Adds a new text insight to the RAG system.

        This method encodes the provided text insight into an embedding vector,
        adds it to the FAISS index, stores the original text in metadata,
        and persists the changes to disk. Duplicate insights are not added.

        Args:
            insight (str): The text insight to add to the system

        Raises:
            RAGException: If there are issues with encoding or saving the insight
            ValueError: If insight is empty or invalid
        """
        if not insight or not isinstance(insight, str):
            raise ValueError("Insight must be a non-empty string")

        # Skip if this exact insight already exists
        if insight in self.metadata:
            return

        try:
            embedding = self.model.encode(insight, show_progress_bar=False)
            embedding = np.array(embedding, dtype="float32").reshape(1, self.embedding_dim)
            self.index.add(embedding)  # type: ignore
            self.metadata.append(insight)
            self.save()
        except Exception as e:
            raise RAGException(f"Failed to add insight: {str(e)}")

    def add_large_text(self, text: str, chunk_size: int = 512, overlap: int = 128) -> None:
        """Adds a large text document by breaking it into smaller overlapping chunks.

        This method splits a large text into semantically meaningful chunks by analyzing
        various contextual cues such as punctuation, paragraph breaks, and code block boundaries.
        It uses a sliding window approach to ensure overlapping content between chunks, preserving
        the context across boundaries. Code blocks (denoted by triple backticks) are
        preserved intact.

        Args:
            text (str): The large text document to add
            chunk_size (int, optional): Target size of each chunk in characters. Defaults to 512.
            overlap (int, optional): Number of overlapping characters between chunks.
            Defaults to 128.

        Raises:
            RAGException: If there are issues processing or adding the text
            ValueError: If text is empty or parameters are invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid chunk_size or overlap parameters")

        try:
            import re

            # Pattern to detect code blocks delimited by triple backticks.
            code_block_pattern = r"(```[\s\S]*?```)"
            parts = re.split(code_block_pattern, text)
            segments = []
            # Use multiple contextual delimiters: punctuation marks and paragraph breaks.
            context_split_pattern = r"(?<=[.?!:;])\s+(?=[A-Z])|\n{2,}"
            for part in parts:
                if part.startswith("```") and part.endswith("```"):
                    segments.append(part.strip())
                else:
                    subsegments = re.split(context_split_pattern, part)
                    segments.extend([seg.strip() for seg in subsegments if seg.strip()])

            # Assemble segments into chunks using a sliding window that preserves
            # contextual overlap.
            chunks = []
            current_chunk = []
            current_size = 0
            for seg in segments:
                seg_length = len(seg)
                # If adding this segment would exceed the chunk size and we already have
                # some segments, finalize current chunk.
                if current_chunk and current_size + seg_length + 1 > chunk_size:
                    chunk_text = " ".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    # Compute the overlap from the tail of the current chunk.
                    overlap_segments = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= overlap:
                            overlap_segments.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current_chunk = overlap_segments
                    current_size = sum(len(s) for s in current_chunk)
                current_chunk.append(seg)
                current_size += seg_length + (1 if len(current_chunk) > 1 else 0)
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

            # Add all non-empty chunks as insights.
            for chunk in chunks:
                if chunk:
                    self.add_insight(chunk)

        except Exception as e:
            raise RAGException(f"Failed to process large text: {str(e)}")

    def query_insight(self, query: str, k: int = 5, max_distance: float = 5) -> List[InsightResult]:
        """Retrieves the most similar insights to a given query.

        This method encodes the query text, performs a k-nearest neighbor search
        in the FAISS index, and returns the most similar insights along with their
        distance scores. Results are filtered by a maximum distance threshold.

        Args:
            query (str): The text query to search for
            k (int, optional): Number of results to return. Defaults to 5.
            max_distance (float, optional): Maximum distance threshold for results.
                Higher distances indicate less relevance. Defaults to 1.5.

        Returns:
            list[InsightResult]: List of InsightResult objects containing retrieved
            insights and their distances. Returns empty list if no insights found.

        Raises:
            ValueError: If query is empty or k is invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")

        try:
            query_embedding = self.model.encode(query, show_progress_bar=False)
            query_embedding = np.array(query_embedding, dtype="float32").reshape(
                1, self.embedding_dim
            )
            distances, labels = self.index.search(query_embedding, k)  # type: ignore

            results = []
            for dist, idx in zip(distances[0], labels[0]):
                if idx < len(self.metadata) and idx != -1 and dist <= max_distance:
                    results.append(InsightResult(insight=self.metadata[idx], distance=float(dist)))

            return results
        except Exception:
            return []

    def save(self) -> None:
        """Persists the current state of the RAG system to disk.

        Saves both the FAISS index and the metadata JSON file containing
        the original text insights.

        Raises:
            RAGException: If there are issues saving the index or metadata
        """
        try:
            write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "w") as f:
                for insight in self.metadata:
                    f.write(json.dumps({"text": insight}) + "\n")
        except Exception as e:
            raise RAGException(f"Failed to save RAG system state: {str(e)}")
