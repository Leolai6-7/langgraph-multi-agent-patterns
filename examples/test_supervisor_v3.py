"""Test script for the Whitepaper Supervisor v3 pattern.

驗證重點：
  - Supervisor 每次都用 LLM 路由，reasoning 反映歷史脈絡
  - messages 在累積到 6 條後觸發壓縮
  - compressed_history 正確累積歷史摘要
  - 壓縮後 messages 只剩最新一條
  - 安全閥仍正常運作（revision_count >= 3）
  - 最終產出完整白皮書

Usage:
    python examples/test_supervisor_v3.py
"""

from langchain_core.messages import HumanMessage

from supervisor_v3 import build_whitepaper_graph_v3


def main():
    graph = build_whitepaper_graph_v3()

    print("=" * 60)
    print("Whitepaper Supervisor v3 Demo")
    print("  (全 LLM 路由 + Messages 壓縮機制)")
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
        "compressed_history": "",
    }

    # Stream step-by-step to observe routing and compression
    for step in graph.stream(initial_state):
        node_name = list(step.keys())[0]
        node_output = step[node_name]

        print(f"\n{'─' * 50}")
        print(f"Node: {node_name}")
        print(f"{'─' * 50}")

        if not node_output:
            print("  (pass through)")
            continue

        if "next" in node_output:
            print(f"  Route → {node_output['next']}")

        if "messages" in node_output:
            for msg in node_output["messages"]:
                content = getattr(msg, "content", str(msg))
                preview = content[:200]
                if len(content) > 200:
                    preview += "..."
                print(f"  [公共區] {preview}")

        if "compressed_history" in node_output:
            ch = node_output["compressed_history"]
            preview = ch[:300]
            if len(ch) > 300:
                preview += "..."
            print(f"  [壓縮歷史] {preview}")

        if "scratchpad" in node_output:
            sp = node_output["scratchpad"]
            if "research_data" in sp:
                data = str(sp["research_data"])
                print(f"  [私有區] research_data: {len(data)} 字元")
            if "current_draft" in sp:
                draft = sp["current_draft"]
                print(f"  [私有區] current_draft: {len(draft)} 字元")
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

    # 取得最終狀態
    print(f"\n{'=' * 60}")
    print("最終狀態")
    print("=" * 60)

    final_state = graph.invoke(initial_state)

    # 顯示壓縮歷史
    ch = final_state.get("compressed_history", "")
    if ch:
        print(f"\n[compressed_history] ({len(ch)} 字元)")
        print(ch[:500])
        if len(ch) > 500:
            print("...")

    # 顯示剩餘 messages 數量
    msgs = final_state.get("messages", [])
    print(f"\n[messages] 剩餘 {len(msgs)} 條")
    for msg in msgs:
        content = msg.content if hasattr(msg, "content") else str(msg)
        print(f"  - {content[:100]}...")

    # 顯示白皮書
    draft = final_state.get("scratchpad", {}).get("current_draft", "")
    if draft:
        print(f"\n{'=' * 60}")
        print("最終白皮書")
        print("=" * 60)
        print(draft[:2000])
        if len(draft) > 2000:
            print(f"\n... (共 {len(draft)} 字元，已截斷)")
    else:
        print("（未產出草稿）")

    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
