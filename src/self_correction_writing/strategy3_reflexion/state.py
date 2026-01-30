"""State for Strategy 3: Reflexion — Verbal Reinforcement Learning.

核心：維護一個「反思記憶庫 (Reflection Memory)」，
透過更新提示詞上下文（Context）讓模型「變聰明」，而非調整權重。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class WritingState(TypedDict):
    topic: str
    criteria: str
    current_draft: str
    revision_history: Annotated[list[str], operator.add]
    # 反思記憶庫：累積的教訓字串列表
    reflections: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int
    score: float
    score_threshold: float
    # evaluator 產出的文字評語，reflector 據此生成反思
    # 人類可在中斷時覆寫，修正錯誤歸因
    critique: str
    final_output: str
    # Self-RAG 新增
    retrieved_memories: list[dict]   # retrieve_memory 寫入的原始檢索結果
    graded_memories: list[str]       # grade_relevance 過濾後的記憶文字
