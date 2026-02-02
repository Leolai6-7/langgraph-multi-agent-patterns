"""Test script for the Whitepaper Supervisor v2 pattern.

Usage:
    python examples/test_supervisor_v2.py
"""

from langchain_core.messages import HumanMessage

from supervisor_v2 import build_whitepaper_graph


def main():
    graph = build_whitepaper_graph()

    print("=" * 60)
    print("Whitepaper Supervisor v2 Demo")
    print("=" * 60)

    initial_state = {
        "messages": [
            HumanMessage(
                content="Edge Computing 在工業 4.0 中的應用與挑戰"
            )
        ],
        "next": "",
        "last_actor": "",
        "scratchpad": {},
    }

    # Stream step-by-step to observe routing
    for step in graph.stream(initial_state):
        node_name = list(step.keys())[0]
        node_output = step[node_name]

        print(f"\n{'─' * 50}")
        print(f"Node: {node_name}")
        print(f"{'─' * 50}")

        if "next" in node_output:
            print(f"  Route → {node_output['next']}")

        if "messages" in node_output:
            for msg in node_output["messages"]:
                preview = msg.content[:200]
                if len(msg.content) > 200:
                    preview += "..."
                print(f"  [公共區] {preview}")

        if "scratchpad" in node_output:
            sp = node_output["scratchpad"]
            # 顯示 scratchpad 中的關鍵狀態（不顯示完整內容）
            if "research_data" in sp:
                data = str(sp["research_data"])
                print(f"  [私有區] research_data: {len(data)} 字元")
            if "current_draft" in sp:
                draft = sp["current_draft"]
                print(f"  [私有區] current_draft: {len(draft)} 字元")
                # 顯示草稿前 3 行
                lines = draft.split("\n")[:3]
                for line in lines:
                    print(f"           {line}")
            if "revision_count" in sp:
                print(f"  [私有區] revision_count: {sp['revision_count']}")
            if "editor_critique" in sp:
                critique = sp["editor_critique"][:150]
                if len(sp["editor_critique"]) > 150:
                    critique += "..."
                print(f"  [私有區] editor_critique: {critique}")

    # 取得最終狀態，輸出白皮書
    print(f"\n{'=' * 60}")
    print("最終白皮書")
    print("=" * 60)

    # 用 invoke 取得最終完整 state
    final_state = graph.invoke(initial_state)
    draft = final_state.get("scratchpad", {}).get("current_draft", "")
    if draft:
        print(draft[:2000])
        if len(draft) > 2000:
            print(f"\n... (共 {len(draft)} 字元，已截斷)")
    else:
        print("（未產出草稿）")

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
