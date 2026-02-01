"""Test script for the Supervisor pattern.

Usage:
    python examples/test_supervisor.py
"""

from langchain_core.messages import HumanMessage

from supervisor import build_supervisor_graph


def main():
    graph = build_supervisor_graph()

    print("=" * 60)
    print("Supervisor Pattern Demo")
    print("=" * 60)

    initial_state = {
        "messages": [
            HumanMessage(content="What are the latest trends in renewable energy in 2024?")
        ],
        "next": "",
    }

    # Stream step-by-step to observe routing
    for step in graph.stream(initial_state):
        node_name = list(step.keys())[0]
        node_output = step[node_name]

        print(f"\n{'─' * 40}")
        print(f"Node: {node_name}")
        print(f"{'─' * 40}")

        if "next" in node_output:
            print(f"  Route → {node_output['next']}")
        if "messages" in node_output:
            for msg in node_output["messages"]:
                preview = msg.content[:200]
                if len(msg.content) > 200:
                    preview += "..."
                print(f"  Message: {preview}")

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
