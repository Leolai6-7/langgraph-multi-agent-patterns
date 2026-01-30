"""向量檢索反思記憶庫 — ChromaDB + Amazon Titan Embedding.

提供跨 session 持久化的語意反思檢索，供 Strategy 3 (Reflexion) 使用。
含三層記憶維護策略：語意去重、效用衰減修剪、LLM 概括合併。
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

import chromadb
from langchain_aws import BedrockEmbeddings

logger = logging.getLogger(__name__)


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
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        max_distance = 1.0 - similarity_threshold
        filtered: list[dict[str, Any]] = []
        for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            if dist <= max_distance:
                filtered.append({
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                })
        return filtered

    # ------------------------------------------------------------------
    # 策略 2：語意去重 (Semantic Deduplication)
    # ------------------------------------------------------------------
    def add_reflection_with_dedup(
        self,
        reflection: str,
        metadata: dict[str, Any],
        dedup_threshold: float = 0.92,
    ) -> str | None:
        """寫入前查重，重複則跳過，回傳 doc_id 或 None。"""
        embedding = self._embeddings.embed_query(reflection)

        # 取出同 criteria_hash 的既有記憶做相似度比對
        where = None
        filter_keys = ["task_type", "criteria_hash"]
        filter_items = [{k: metadata[k]} for k in filter_keys if k in metadata]
        if filter_items:
            where = {"$and": filter_items} if len(filter_items) > 1 else filter_items[0]

        existing = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where=where,
        )
        distances = existing.get("distances", [[]])[0]
        if distances:
            similarity = 1.0 - distances[0]
            if similarity >= dedup_threshold:
                logger.info(
                    "Dedup: skipping reflection (similarity=%.3f >= %.2f)",
                    similarity,
                    dedup_threshold,
                )
                return None

        # 新增 utility_score 到 metadata
        metadata_with_utility = {**metadata, "utility_score": 1.0}
        return self.add_reflection(reflection, metadata_with_utility)

    # ------------------------------------------------------------------
    # 策略 3：效用衰減與修剪 (Utility Decay & Pruning)
    # ------------------------------------------------------------------
    def boost_utility(self, doc_ids: list[str], boost: float = 0.1) -> None:
        """被檢索命中的記憶增加 utility_score。"""
        if not doc_ids:
            return
        results = self._collection.get(ids=doc_ids)
        batch_ids: list[str] = []
        batch_metas: list[dict[str, Any]] = []
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            current = float(meta.get("utility_score", 1.0))
            new_score = min(current + boost, 2.0)
            batch_ids.append(doc_id)
            batch_metas.append({**meta, "utility_score": new_score})
        if batch_ids:
            self._collection.update(ids=batch_ids, metadatas=batch_metas)

    def _decay_and_prune(
        self,
        metadata_filter: dict[str, Any],
        decay_rate: float = 0.05,
        prune_threshold: float = 0.3,
    ) -> int:
        """衰減所有記憶的 utility_score，刪除低於閾值的。回傳刪除數量。"""
        where = self._build_where(metadata_filter)
        all_docs = self._collection.get(where=where)
        ids_to_delete: list[str] = []
        for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"]):
            current = float(meta.get("utility_score", 1.0))
            new_score = current - decay_rate
            if new_score < prune_threshold:
                ids_to_delete.append(doc_id)
            else:
                self._collection.update(
                    ids=[doc_id],
                    metadatas=[{**meta, "utility_score": new_score}],
                )
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info("Pruned %d low-utility memories", len(ids_to_delete))
        return len(ids_to_delete)

    # ------------------------------------------------------------------
    # 策略 1：LLM 概括合併 (Summarization & Consolidation)
    # ------------------------------------------------------------------
    def _consolidate(
        self,
        metadata_filter: dict[str, Any],
        max_count: int = 10,
    ) -> bool:
        """當記憶數量超過 max_count，用 LLM 壓縮為 3 條核心原則。回傳是否執行了合併。"""
        where = self._build_where(metadata_filter)
        all_docs = self._collection.get(where=where)
        count = len(all_docs["ids"])
        if count <= max_count:
            return False

        # 延遲匯入避免循環依賴
        from self_correction_writing.common import invoke

        documents = all_docs["documents"]
        numbered = "\n".join(f"{i+1}. {doc}" for i, doc in enumerate(documents))
        system = (
            "你是一位記憶管理專家。以下是多條寫作反思記憶，"
            "請將它們概括合併為 3 條最核心的原則。\n"
            "每條原則獨佔一行，以 '- ' 開頭。只回覆 3 條原則，不要其他文字。"
        )
        human = f"反思記憶列表：\n{numbered}"
        raw = invoke(system, human)

        # 解析 LLM 回覆的 3 條原則
        principles = self._parse_principles(raw)
        if not principles:
            logger.warning("Consolidation: LLM returned no parseable principles")
            return False

        # 先寫入新原則，再刪除舊記憶（避免寫入失敗導致資料遺失）
        base_meta = {
            k: v
            for k, v in metadata_filter.items()
            if isinstance(v, (str, int, float, bool))
        }
        base_meta["utility_score"] = 1.0
        base_meta["consolidated"] = True
        new_ids: list[str] = []
        try:
            for principle in principles:
                new_ids.append(self.add_reflection(principle, base_meta))
        except Exception:
            logger.exception("Consolidation: failed to write principles, rolling back")
            if new_ids:
                self._collection.delete(ids=new_ids)
            return False

        self._collection.delete(ids=all_docs["ids"])

        logger.info(
            "Consolidated %d memories into %d principles",
            count,
            len(principles),
        )
        return True

    @staticmethod
    def _parse_principles(raw: str) -> list[str]:
        """解析 LLM 回覆中以 '- ' 或 'N. ' 開頭的原則列表。"""
        principles: list[str] = []
        for line in raw.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("-", "*")) and len(stripped) > 2:
                principles.append(stripped.lstrip("-* ").strip())
            elif stripped[0].isdigit() and "." in stripped[:4]:
                _, _, rest = stripped.partition(".")
                if rest.strip():
                    principles.append(rest.strip())
        return principles[:3]

    # ------------------------------------------------------------------
    # 統一入口
    # ------------------------------------------------------------------
    def run_maintenance(
        self,
        metadata_filter: dict[str, Any],
        decay_rate: float = 0.05,
        prune_threshold: float = 0.3,
        consolidation_threshold: int = 10,
    ) -> None:
        """依序執行：decay_and_prune → consolidate。"""
        self._decay_and_prune(
            metadata_filter,
            decay_rate=decay_rate,
            prune_threshold=prune_threshold,
        )
        self._consolidate(
            metadata_filter,
            max_count=consolidation_threshold,
        )

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _build_where(metadata_filter: dict[str, Any]) -> dict | None:
        """將 metadata_filter 轉換為 ChromaDB where 語法。"""
        if not metadata_filter:
            return None
        items = [{k: v} for k, v in metadata_filter.items()]
        return {"$and": items} if len(items) > 1 else items[0]
