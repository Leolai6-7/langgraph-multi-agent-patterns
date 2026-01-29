"""LangGraph graph definition for confidence-weighted voting."""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, START, END

from confidence_voting.state import VotingState
from confidence_voting.agents import agent_optimist, agent_skeptic, agent_analyst
from confidence_voting.aggregator import aggregator


def dispatcher(state: VotingState) -> dict[str, Any]:
    """Pass-through node that prepares the query for fan-out."""
    return {}


def build_graph() -> StateGraph:
    """Build and compile the confidence-weighted voting graph."""
    builder = StateGraph(VotingState)

    # Add nodes
    builder.add_node("dispatcher", dispatcher)
    builder.add_node("agent_optimist", agent_optimist)
    builder.add_node("agent_skeptic", agent_skeptic)
    builder.add_node("agent_analyst", agent_analyst)
    builder.add_node("aggregator", aggregator)

    # START -> dispatcher
    builder.add_edge(START, "dispatcher")

    # dispatcher -> fan-out to all agents
    builder.add_edge("dispatcher", "agent_optimist")
    builder.add_edge("dispatcher", "agent_skeptic")
    builder.add_edge("dispatcher", "agent_analyst")

    # all agents -> aggregator (fan-in)
    builder.add_edge("agent_optimist", "aggregator")
    builder.add_edge("agent_skeptic", "aggregator")
    builder.add_edge("agent_analyst", "aggregator")

    # aggregator -> END
    builder.add_edge("aggregator", END)

    return builder.compile()
