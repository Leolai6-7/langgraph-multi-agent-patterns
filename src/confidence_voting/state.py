"""State definitions for the confidence-weighted voting graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Vote(TypedDict):
    agent_name: str
    choice: str
    confidence: float  # 0.0 ~ 1.0
    reasoning: str


class VotingState(TypedDict):
    query: str
    votes: Annotated[list[Vote], operator.add]
    final_decision: str
