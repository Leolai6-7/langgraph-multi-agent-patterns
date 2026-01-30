"""Strategy 3: Reflexion + Self-RAG + Human-in-the-Loop.

Flow:
  retrieve_memory → grade_relevance → generate → evaluator
      ↑                                           ↓
      └── reflector ←─────── [INTERRUPT] ←────────┘
                                                   ↓
                                          [INTERRUPT] → finalize → END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from self_correction_writing.strategy3_reflexion.state import WritingState
from self_correction_writing.strategy3_reflexion.agents import (
    retrieve_memory,
    grade_relevance,
    generate,
    evaluator,
    reflector,
    finalize,
)


def _should_retry_or_finish(state: WritingState) -> str:
    score = state.get("score", 0.0)
    threshold = state.get("score_threshold", 0.7)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if score >= threshold or iteration >= max_iter:
        return "finalize"
    return "reflector"


def build_graph_strategy3():
    builder = StateGraph(WritingState)

    builder.add_node("retrieve_memory", retrieve_memory)
    builder.add_node("grade_relevance", grade_relevance)
    builder.add_node("generate", generate)
    builder.add_node("evaluator", evaluator)
    builder.add_node("reflector", reflector)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "retrieve_memory")
    builder.add_edge("retrieve_memory", "grade_relevance")
    builder.add_edge("grade_relevance", "generate")
    builder.add_edge("generate", "evaluator")

    builder.add_conditional_edges(
        "evaluator",
        _should_retry_or_finish,
        {
            "reflector": "reflector",
            "finalize": "finalize",
        },
    )
    builder.add_edge("reflector", "retrieve_memory")
    builder.add_edge("finalize", END)

    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["reflector", "finalize"],
    )
