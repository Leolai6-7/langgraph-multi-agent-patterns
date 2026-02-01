"""Supervisor pattern graph builder."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import researcher_node, supervisor_node, writer_node
from .state import SupervisorState


def build_supervisor_graph():
    """Build and compile the Supervisor graph.

    Flow:
        START → Supervisor → (Researcher | Writer | FINISH→END)
        Researcher → Supervisor
        Writer → Supervisor
    """
    builder = StateGraph(SupervisorState)

    builder.add_node("Supervisor", supervisor_node)
    builder.add_node("Researcher", researcher_node)
    builder.add_node("Writer", writer_node)

    builder.add_edge(START, "Supervisor")
    builder.add_edge("Researcher", "Supervisor")
    builder.add_edge("Writer", "Supervisor")

    builder.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {"Researcher": "Researcher", "Writer": "Writer", "FINISH": END},
    )

    return builder.compile()
