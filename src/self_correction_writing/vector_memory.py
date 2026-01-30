"""向量檢索反思記憶庫 — ChromaDB + Amazon Titan Embedding.

提供跨 session 持久化的語意反思檢索，供 Strategy 3 (Reflexion) 使用。
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

import chromadb
from langchain_aws import BedrockEmbeddings


class ReflectionVectorStore:
    """Singleton 向量記憶庫，封裝 ChromaDB collection 操作。"""

    _instance: ReflectionVectorStore | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        persist_directory: str = "vector_store/reflexion",
        collection_name: str = "reflexion_reflections",
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def get_instance(
        cls,
        persist_directory: str = "vector_store/reflexion",
        collection_name: str = "reflexion_reflections",
    ) -> ReflectionVectorStore:
        """取得或建立 singleton 實例。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        persist_directory=persist_directory,
                        collection_name=collection_name,
                    )
        return cls._instance

    def add_reflection(self, reflection: str, metadata: dict[str, Any]) -> str:
        """將反思文字寫入向量庫，回傳 document id。"""
        doc_id = str(uuid.uuid4())
        embedding = self._embeddings.embed_query(reflection)
        # ChromaDB 的 metadata value 只接受 str / int / float / bool
        safe_metadata = {
            k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))
        }
        self._collection.add(
            ids=[doc_id],
            documents=[reflection],
            embeddings=[embedding],
            metadatas=[safe_metadata],
        )
        return doc_id

    def retrieve_reflections(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
    ) -> list[dict[str, Any]]:
        """語意檢索相關反思。

        回傳格式: [{"document": str, "metadata": dict, "distance": float}, ...]
        ChromaDB cosine distance = 1 - cosine_similarity，
        所以 threshold 0.75 → distance <= 0.25。
        """
        query_embedding = self._embeddings.embed_query(query)
        # ChromaDB 多欄位 filter 需用 $and 包裝
        where = None
        if metadata_filter:
            items = [{k: v} for k, v in metadata_filter.items()]
            where = {"$and": items} if len(items) > 1 else items[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        # results 是 dict[str, list[list[...]]]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        max_distance = 1.0 - similarity_threshold
        filtered: list[dict[str, Any]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            if dist <= max_distance:
                filtered.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                })
        return filtered
