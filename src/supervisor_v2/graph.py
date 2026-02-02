"""Whitepaper supervisor graph builder."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import editor_node, researcher_node, supervisor_node, writer_node
from .state import WhitepaperState


def build_whitepaper_graph():
    """Build and compile the Whitepaper Supervisor graph.

    Flow:
        START → Supervisor → (Researcher | Writer | Editor | FINISH→END)
        Researcher → Supervisor
        Writer → Supervisor
        Editor → Supervisor

    Supervisor 根據 Editor 的 verdict 決定下一步：
        - PASS → FINISH
        - NEEDS_MORE_DATA → Researcher
        - NEEDS_REVISION → Writer
        - revision_count >= 3 → FINISH（安全閥）
    """
    builder = StateGraph(WhitepaperState)

    builder.add_node("Supervisor", supervisor_node)
    builder.add_node("Researcher", researcher_node)
    builder.add_node("Writer", writer_node)
    builder.add_node("Editor", editor_node)

    builder.add_edge(START, "Supervisor")
    for member in ["Researcher", "Writer", "Editor"]:
        builder.add_edge(member, "Supervisor")

    builder.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {
            "Researcher": "Researcher",
            "Writer": "Writer",
            "Editor": "Editor",
            "FINISH": END,
        },
    )

    return builder.compile()
