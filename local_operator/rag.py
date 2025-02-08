import json
import os
from typing import Dict, List, Union

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

    index_path: str
    metadata_path: str
    embedding_dim: int
    model: SentenceTransformer
    index: faiss.Index
    metadata: list[str]

    def __init__(
        self,
        file_path: str,
        embedding_dim: int = 384,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if not file_path:
            raise ValueError("file_path cannot be empty")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        try:
            os.makedirs(file_path, exist_ok=True)
            self.index_path = os.path.join(file_path, "rag_index.index")
            self.metadata_path = os.path.join(file_path, "rag_metadata.json")

            self.embedding_dim = embedding_dim
            self.model = SentenceTransformer(model_name)

            # Load existing index and metadata if available; otherwise, create new ones.
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                try:
                    self.index = faiss.read_index(self.index_path)
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

    def query_insight(self, query: str, k: int = 5) -> List[InsightResult]:
        """Retrieves the most similar insights to a given query.

        This method encodes the query text, performs a k-nearest neighbor search
        in the FAISS index, and returns the most similar insights along with their
        distance scores.

        Args:
            query (str): The text query to search for
            k (int, optional): Number of results to return. Defaults to 5.

        Returns:
            list[InsightResult]: List of InsightResult objects containing retrieved
            insights and their distances

        Raises:
            RAGException: If there are issues with query processing or search
            ValueError: If query is empty or k is invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")

        try:
            query_embedding = self.model.encode(query, show_progress_bar=False)
            query_embedding = np.array(query_embedding, dtype="float32").reshape(
                1, self.embedding_dim
            )
            distances, labels = self.index.search(query_embedding, k)  # type: ignore

            results = []
            for dist, idx in zip(distances[0], labels[0]):
                if idx < len(self.metadata) and idx != -1:
                    results.append(InsightResult(insight=self.metadata[idx], distance=float(dist)))

            return results
        except Exception as e:
            raise RAGException(f"Failed to query insights: {str(e)}")

    def save(self) -> None:
        """Persists the current state of the RAG system to disk.

        Saves both the FAISS index and the metadata JSON file containing
        the original text insights.

        Raises:
            RAGException: If there are issues saving the index or metadata
        """
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f)
        except Exception as e:
            raise RAGException(f"Failed to save RAG system state: {str(e)}")


def rag_tool_fetch(
    embedding_manager: EmbeddingManager, query: str, k: int = 5
) -> List[InsightResult]:
    """Queries the RAG system for similar insights.

    A tool function that interfaces with the EmbeddingManager to retrieve
    insights similar to the provided query.

    Args:
        embedding_manager (EmbeddingManager): Instance of the RAG system manager
        query (str): Text query to search for similar insights
        k (int, optional): Number of results to return. Defaults to 5.

    Returns:
        list[InsightResult]: List of InsightResult objects containing retrieved
        insights and their distances

    Raises:
        RAGException: If there are issues with the query operation
        ValueError: If parameters are invalid
    """
    return embedding_manager.query_insight(query, k)
