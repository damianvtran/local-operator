import json
import os
from pathlib import Path
from typing import List

import faiss
import numpy as np
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
    index: faiss.Index
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
            self.metadata_path = file_path / "rag_metadata.json"

            self.embedding_dim = embedding_dim
            self.model = SentenceTransformer(model_name)

            # Load existing index and metadata if available; otherwise, create new ones.
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    with open(self.metadata_path, "r") as f:
                        self.metadata = json.load(f)
                except (RuntimeError, json.JSONDecodeError, IOError) as e:
                    raise RAGException(f"Failed to load existing index/metadata: {str(e)}")
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.metadata = []  # List to store insights (as plain text).
                self.save()  # Creates the initial index and metadata file.
        except Exception as e:
            raise RAGException(f"Failed to initialize EmbeddingManager: {str(e)}")

    def add_insight(self, insight: str) -> None:
        """Adds a new text insight to the RAG system.

        This method encodes the provided text insight into an embedding vector,
        adds it to the FAISS index, stores the original text in metadata,
        and persists the changes to disk.

        Args:
            insight (str): The text insight to add to the system

        Raises:
            RAGException: If there are issues with encoding or saving the insight
            ValueError: If insight is empty or invalid
        """
        if not insight or not isinstance(insight, str):
            raise ValueError("Insight must be a non-empty string")

        try:
            embedding = self.model.encode(insight, show_progress_bar=False)
            embedding = np.array(embedding, dtype="float32").reshape(1, self.embedding_dim)
            self.index.add(embedding)  # type: ignore
            self.metadata.append(insight)
            self.save()
        except Exception as e:
            raise RAGException(f"Failed to add insight: {str(e)}")

    def add_large_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> None:
        """Adds a large text document by breaking it into smaller overlapping chunks.

        This method splits a large text into smaller, semantically meaningful chunks
        with some overlap to maintain context across boundaries. Each chunk is then
        added as a separate insight. Code blocks are preserved intact.

        Args:
            text (str): The large text document to add
            chunk_size (int, optional): Target size of each chunk in characters. Defaults to 500.
            overlap (int, optional): Number of overlapping characters between chunks.
            Defaults to 50.

        Raises:
            RAGException: If there are issues processing or adding the text
            ValueError: If text is empty or parameters are invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid chunk_size or overlap parameters")

        try:
            # Split text into chunks while preserving code blocks
            chunks = []
            current_chunk = ""
            lines = text.split("\n")
            in_code_block = False
            code_block = []
            code_block_language = ""

            for line in lines:
                # Check for code block markers
                if line.strip().startswith("```"):
                    if not in_code_block:
                        in_code_block = True
                        # Extract language if specified
                        code_block_language = line.strip()[3:].strip()
                        # If we have content before code block, add it as a chunk
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        code_block = [f"```{code_block_language}"]
                        continue
                    else:
                        in_code_block = False
                        code_block.append("```")
                        # Add complete code block as a chunk
                        chunks.append("\n".join(code_block))
                        code_block = []
                        code_block_language = ""
                        continue

                # Handle content based on whether we're in a code block
                if in_code_block:
                    code_block.append(line)
                else:
                    # Regular text processing
                    if len(current_chunk) + len(line) <= chunk_size:
                        current_chunk += line + "\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        # Start new chunk with overlap
                        if overlap > 0 and len(current_chunk) > overlap:
                            # Keep last few lines that fit within overlap
                            overlap_lines = current_chunk.split("\n")
                            overlap_text = ""
                            for ol in reversed(overlap_lines):
                                if len(overlap_text) + len(ol) <= overlap:
                                    overlap_text = ol + "\n" + overlap_text
                                else:
                                    break
                            current_chunk = overlap_text + line + "\n"
                        else:
                            current_chunk = line + "\n"

            # Add final chunk if it exists
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Add all chunks as insights
            for chunk in chunks:
                if chunk.strip():
                    self.add_insight(chunk)

        except Exception as e:
            raise RAGException(f"Failed to process large text: {str(e)}")

    def query_insight(
        self, query: str, k: int = 5, max_distance: float = 1.5
    ) -> List[InsightResult]:
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
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            raise RAGException(f"Failed to save RAG system state: {str(e)}")
