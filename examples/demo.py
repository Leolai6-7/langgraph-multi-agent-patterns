"""Demo script for the confidence-weighted voting system."""

import sys
import os

# Allow running from project root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from confidence_voting import build_graph


def main() -> None:
    query = (
        "Should a mid-stage startup invest heavily in AI-powered features "
        "for their existing SaaS product this quarter?"
    )

    print(f"Query: {query}\n")
    print("Running agents...\n")

    graph = build_graph()
    result = graph.invoke({"query": query, "votes": []})

    print("=" * 60)
    print("Individual votes:")
    print("=" * 60)
    for vote in result["votes"]:
        print(f"\n[{vote['agent_name']}]")
        print(f"  Choice:     {vote['choice']}")
        print(f"  Confidence: {vote['confidence']:.2f}")
        print(f"  Reasoning:  {vote['reasoning']}")

    print("\n" + "=" * 60)
    print("Final Decision (confidence-weighted)")
    print("=" * 60)
    print(result["final_decision"])


if __name__ == "__main__":
    main()
