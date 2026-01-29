"""Confidence-weighted vote aggregation logic."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from confidence_voting.state import VotingState


def aggregator(state: VotingState) -> dict[str, Any]:
    """Aggregate votes using confidence-weighted scoring.

    Groups votes by choice, sums confidence scores per group,
    and selects the choice with the highest total confidence.
    """
    votes = state["votes"]

    if not votes:
        return {"final_decision": "No votes received."}

    # Sum confidence per choice
    scores: dict[str, float] = defaultdict(float)
    vote_details: dict[str, list[str]] = defaultdict(list)

    for v in votes:
        choice = v["choice"]
        scores[choice] += v["confidence"]
        vote_details[choice].append(
            f"  - {v['agent_name']} (confidence={v['confidence']:.2f}): {v['reasoning']}"
        )

    # Pick highest scoring choice
    winner = max(scores, key=scores.get)  # type: ignore[arg-type]

    # Build summary
    lines = [f"Decision: {winner} (weighted score: {scores[winner]:.2f})", ""]
    lines.append("Vote breakdown:")
    for choice, detail_list in vote_details.items():
        marker = ">>> " if choice == winner else "    "
        lines.append(f"{marker}{choice} — total confidence: {scores[choice]:.2f}")
        lines.extend(detail_list)

    return {"final_decision": "\n".join(lines)}
