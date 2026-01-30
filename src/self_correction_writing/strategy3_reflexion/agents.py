"""Node functions for Strategy 3: Reflexion — Verbal Reinforcement Learning."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from self_correction_writing.common import invoke, parse_json
from self_correction_writing.strategy3_reflexion.state import WritingState
from self_correction_writing.vector_memory import ReflectionVectorStore


# ---------------------------------------------------------------------------
# Node: generator (Actor)
# ---------------------------------------------------------------------------
def generator(state: WritingState) -> dict[str, Any]:
    """根據主題、標準、以及反思記憶庫中的教訓來撰寫文章。

    關鍵：把 reflections 動態注入到 prompt 中，
    等於告訴模型「你過去犯過這些錯，這次不要再犯」。

    雙記憶系統：
    1. State reflections — session 內的反思（即時）
    2. ChromaDB — 跨 session 持久化的語意反思（向量檢索）
    """
    # --- Session 內反思 ---
    reflections = state.get("reflections", [])

    # --- 跨 session 向量記憶檢索 ---
    topic = state["topic"]
    criteria = state["criteria"]
    criteria_hash = hashlib.md5(criteria.encode()).hexdigest()[:8]

    vector_store = ReflectionVectorStore.get_instance()
    query = f"{topic} {criteria}"
    retrieved = vector_store.retrieve_reflections(
        query=query,
        metadata_filter={"task_type": "writing", "criteria_hash": criteria_hash},
        top_k=5,
        similarity_threshold=0.75,
    )
    retrieved_texts = [r["document"] for r in retrieved]

    # 合併：去重（向量庫可能包含本 session 已有的反思）
    all_reflections = list(dict.fromkeys(reflections + retrieved_texts))

    reflection_block = ""
    if all_reflections:
        memory = "\n".join(f"- {r}" for r in all_reflections)
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
        f"主題：{topic}\n"
        f"標準：{criteria}"
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
    vector_store.add_reflection(
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

    return {
        "reflections": [reflection],
        "iteration": iteration + 1,
    }


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------
def finalize(state: WritingState) -> dict[str, Any]:
    return {"final_output": state["current_draft"]}
