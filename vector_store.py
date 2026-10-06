"""
vector_store.py - In-Memory Local Vector Store with NumPy Cosine Similarity
Part of BuildMate AI - Intelligent Construction Site Assistant
"""

import numpy as np

class LocalVectorStore:
    def __init__(self):
        self.embeddings = []  # List of float32 vectors
        self.chunks = []      # List of text strings
        self.metadata = []    # List of metadata dicts {"source": filename, "page": page_no}

    def add_chunks(self, chunks: list, embeddings: list, metadata: list = None):
        """Adds chunks, embeddings, and metadata into the local vector store."""
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.embeddings.append(embeddings[i])
            if metadata and i < len(metadata):
                self.metadata.append(metadata[i])
            else:
                self.metadata.append({"source": "Uploaded Document", "page": 1})

    def search(self, query_embedding: list, top_k: int = 3, threshold: float = 0.35) -> list:
        """
        Executes normalized dot product (cosine similarity) between query vector and chunk vectors.
        Returns top-K relevant chunks exceeding the similarity threshold.
        """
        if not self.embeddings:
            return []

        matrix = np.array(self.embeddings, dtype=np.float32)
        q_vec = np.array(query_embedding, dtype=np.float32)

        # Dot product
        dot_products = np.dot(matrix, q_vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        q_norm = np.linalg.norm(q_vec)

        # Cosine similarity calculation
        similarities = dot_products / (matrix_norms * q_norm + 1e-10)

        # Rank indices descending
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": score
                })
        return results

    def clear(self):
        """Clears stored vectors and chunks."""
        self.embeddings = []
        self.chunks = []
        self.metadata = []
