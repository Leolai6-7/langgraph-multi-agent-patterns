"""Node functions for Strategy 3: Reflexion — Verbal Reinforcement Learning."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from self_correction_writing.common import invoke, parse_json
from self_correction_writing.strategy3_reflexion.state import WritingState
from self_correction_writing.vector_memory import ReflectionVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node: retrieve_memory
# ---------------------------------------------------------------------------
def retrieve_memory(state: WritingState) -> dict[str, Any]:
    """從 ChromaDB 檢索向量記憶 + boost utility，合併 session reflections。

    寫入 state["retrieved_memories"]，供 grade_relevance 評分。
    """
    reflections = state.get("reflections", [])
    topic = state["topic"]
    criteria = state["criteria"]
    criteria_hash = hashlib.md5(criteria.encode()).hexdigest()[:8]

    retrieved: list[dict] = []
    try:
        vector_store = ReflectionVectorStore.get_instance()
        query = f"{topic} {criteria}"
        retrieved = vector_store.retrieve_reflections(
            query=query,
            metadata_filter={"task_type": "writing", "criteria_hash": criteria_hash},
            top_k=5,
            similarity_threshold=0.75,
        )
        # 被檢索命中的記憶增加 utility_score
        if retrieved:
            vector_store.boost_utility([r["id"] for r in retrieved])
    except Exception:
        logger.exception("Failed to retrieve/boost reflections from vector store")

    # 合併 session reflections 與向量記憶，去重
    retrieved_texts = [r["document"] for r in retrieved]
    all_texts = list(dict.fromkeys(reflections + retrieved_texts))

    # 將每條記憶存為 dict 以便 grader 逐條評分
    retrieved_memories = [{"text": t} for t in all_texts]

    return {"retrieved_memories": retrieved_memories}


# ---------------------------------------------------------------------------
# Node: grade_relevance (Self-RAG Grader)
# ---------------------------------------------------------------------------
def grade_relevance(state: WritingState) -> dict[str, Any]:
    """用 LLM 批量評分每條記憶的相關性，過濾不相關記憶。

    一次 LLM call 送出所有記憶，回傳 JSON array ["YES", "NO", ...]。
    只保留 "YES" 的記憶寫入 state["graded_memories"]。
    """
    memories = state.get("retrieved_memories", [])
    if not memories:
        return {"graded_memories": []}

    topic = state["topic"]
    criteria = state["criteria"]

    numbered = "\n".join(f"{i+1}. {m['text']}" for i, m in enumerate(memories))

    system = (
        "你是一位相關性評分員。判斷每條記憶對當前寫作任務是否相關。\n"
        '回覆 JSON array，每個元素是 "YES" 或 "NO"，順序對應輸入的記憶。\n'
        '只回覆 JSON array，例如：["YES", "NO", "YES"]'
    )
    human = (
        f"主題：{topic}\n"
        f"標準：{criteria}\n\n"
        f"記憶列表：\n{numbered}\n\n"
        "請回覆 JSON array。"
    )
    raw = invoke(system, human)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        verdicts = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        verdicts = []

    # 若解析失敗或長度不符，保守地保留所有記憶
    if not isinstance(verdicts, list) or len(verdicts) != len(memories):
        logger.warning(
            "Grade relevance: unexpected LLM response, keeping all %d memories",
            len(memories),
        )
        return {"graded_memories": [m["text"] for m in memories]}

    graded = [
        m["text"]
        for m, v in zip(memories, verdicts)
        if str(v).strip().upper() == "YES"
    ]
    logger.info(
        "Grade relevance: kept %d / %d memories", len(graded), len(memories)
    )
    return {"graded_memories": graded}


# ---------------------------------------------------------------------------
# Node: generate (純生成)
# ---------------------------------------------------------------------------
def generate(state: WritingState) -> dict[str, Any]:
    """讀取已過濾的 graded_memories，注入 prompt 生成文章。"""
    graded = state.get("graded_memories", [])

    reflection_block = ""
    if graded:
        memory = "\n".join(f"- {r}" for r in graded)
        reflection_block = (
            f"\n\n從過去的嘗試中學習到的教訓：\n{memory}\n"
            "請務必將這些教訓融入你的寫作中，避免重蹈覆轍。"
        )

    system = (
        "你是一位專業的撰稿人。請根據指定的主題與標準，"
        "撰寫一篇結構完整、內容充實的長篇文章（至少 1500 字）。"
        "文章須包含引言、多個論述段落與結論。請全程使用繁體中文。"
    )
    human = (
        f"主題：{state['topic']}\n"
        f"標準：{state['criteria']}"
        f"{reflection_block}\n\n"
        "請撰寫完整文章。"
    )
    draft = invoke(system, human)
    return {
        "current_draft": draft,
        "revision_history": [draft],
    }


# ---------------------------------------------------------------------------
# Node: evaluator
# ---------------------------------------------------------------------------
def evaluator(state: WritingState) -> dict[str, Any]:
    """評分並產出文字評語 (critique)。

    輸出 score（控制條件路由）和 critique（給 reflector 參考）。
    人類可在 reflector 之前覆寫 critique，
    修正錯誤歸因，確保 reflector 基於正確的原因生成記憶。
    """
    system = (
        "你是一位嚴格的寫作評估者。請根據以下維度為文章評分與評語：\n"
        "1. 內容深度與論證品質\n"
        "2. 結構完整性（引言、論述、結論）\n"
        "3. 篇幅充實度（優秀文章應至少 1500 字，過短則扣分）\n"
        "4. 是否符合指定標準\n"
        "請嚴格評分，只有真正優秀的文章才能得到 0.8 以上。\n"
        "請回覆有效的 JSON：\n"
        '{"score": <0.0-1.0 浮點數>, "critique": "<具體指出文章的問題與改進方向>"}'
    )
    human = (
        f"標準：{state['criteria']}\n\n"
        f"文章：\n{state['current_draft']}"
    )
    raw = invoke(system, human)
    parsed = parse_json(raw, {"score": 0.5, "critique": "無法解析評語"})
    score = max(0.0, min(1.0, float(parsed.get("score", 0.5))))
    critique = parsed.get("critique", "")
    return {"score": score, "critique": critique}


# ---------------------------------------------------------------------------
# Node: reflector (Reflexion 架構的核心)
# ---------------------------------------------------------------------------
def reflector(state: WritingState) -> dict[str, Any]:
    """根據 evaluator 的 critique 生成反思，追加到記憶庫。

    不直接修改文章，只產出教訓字串。
    這就是 Verbal Reinforcement Learning 的核心：
    用自然語言更新 context，而非梯度更新。

    雙寫入：
    1. 回傳 reflections → state（session 內）
    2. 寫入 ChromaDB（跨 session 持久化）

    人類可在中斷時以 as_node="evaluator" 覆寫 critique，
    reflector 會基於修正後的評語生成正確的反思。
    """
    critique = state.get("critique", "")

    system = (
        "你是一位反思學習代理。根據評審的意見，"
        "生成一段簡短的「反思」，解釋為什麼會犯錯，以及下次該如何避免。\n"
        "範例：「我忽略了語氣的一致性，下次應確保全篇使用正式語氣。」\n"
        "請使用繁體中文，只回覆反思內容本身。"
    )
    human = (
        f"分數：{state['score']}\n\n"
        f"評審意見：{critique}\n\n"
        f"標準：{state['criteria']}\n\n"
        f"文章：\n{state['current_draft']}\n\n"
        "請生成反思。"
    )
    reflection = invoke(system, human)
    iteration = state.get("iteration", 0)

    # --- 寫入向量記憶庫（跨 session 持久化）---
    topic = state["topic"]
    criteria = state["criteria"]
    criteria_hash = hashlib.md5(criteria.encode()).hexdigest()[:8]

    vector_store = ReflectionVectorStore.get_instance()
    metadata_filter = {"task_type": "writing", "criteria_hash": criteria_hash}

    try:
        # 策略 2：語意去重寫入
        vector_store.add_reflection_with_dedup(
            reflection=reflection,
            metadata={
                "task_type": "writing",
                "topic": topic,
                "score": state.get("score", 0.0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iteration": iteration,
                "criteria_hash": criteria_hash,
            },
        )
        # 寫入後觸發記憶維護（衰減 + 修剪 + 合併）
        vector_store.run_maintenance(metadata_filter=metadata_filter)
    except Exception:
        logger.exception("Failed to persist reflection or run maintenance")

    return {
        "reflections": [reflection],
        "iteration": iteration + 1,
    }


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------
def finalize(state: WritingState) -> dict[str, Any]:
    return {"final_output": state["current_draft"]}
