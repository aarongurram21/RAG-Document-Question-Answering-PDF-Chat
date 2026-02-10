from __future__ import annotations

from typing import List, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSRetriever:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.index: faiss.IndexFlatIP | None = None
        self.texts: List[str] = []
        self.metadatas: List[dict] = []

    def add_documents(self, texts: Sequence[str], metadatas: Sequence[dict]) -> None:
        embeddings = self._embed(texts)
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, dict, float]]:
        if self.index is None or not self.texts:
            return []
        query_vec = self._embed([query])
        scores, indices = self.index.search(query_vec, k)
        results: List[Tuple[str, dict, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.texts[idx], self.metadatas[idx], float(score)))
        return results

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        return embeddings.astype("float32")
