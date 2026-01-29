"""Agent definitions for the confidence-weighted voting system."""

from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from confidence_voting.state import Vote, VotingState

_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

_AGENT_PERSONAS: dict[str, str] = {
    "optimist": (
        "You are an optimistic analyst. You tend to see opportunities and upsides. "
        "Evaluate the query and provide your choice with a confidence score."
    ),
    "skeptic": (
        "You are a skeptical analyst. You focus on risks, downsides, and potential pitfalls. "
        "Evaluate the query and provide your choice with a confidence score."
    ),
    "analyst": (
        "You are a balanced, data-driven analyst. You weigh evidence objectively. "
        "Evaluate the query and provide your choice with a confidence score."
    ),
}

_VOTE_INSTRUCTION = """
Given the query below, respond with ONLY valid JSON (no markdown fences) in this format:
{
  "choice": "<your recommended choice as a short string>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one or two sentences explaining your reasoning>"
}

Query: {query}
"""


def _make_agent_node(agent_name: str):
    """Create a graph node function for a given agent persona."""

    def node(state: VotingState) -> dict[str, Any]:
        persona = _AGENT_PERSONAS[agent_name]
        query = state["query"]

        response = _LLM.invoke([
            SystemMessage(content=persona),
            HumanMessage(content=_VOTE_INSTRUCTION.format(query=query)),
        ])

        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            parsed = {
                "choice": "uncertain",
                "confidence": 0.1,
                "reasoning": f"Failed to parse response: {response.content[:200]}",
            }

        vote: Vote = {
            "agent_name": agent_name,
            "choice": parsed["choice"],
            "confidence": max(0.0, min(1.0, float(parsed["confidence"]))),
            "reasoning": parsed["reasoning"],
        }
        return {"votes": [vote]}

    node.__name__ = f"agent_{agent_name}"
    return node


agent_optimist = _make_agent_node("optimist")
agent_skeptic = _make_agent_node("skeptic")
agent_analyst = _make_agent_node("analyst")
