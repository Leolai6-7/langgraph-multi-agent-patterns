"""Test Strategy 3: Pure Reflexion — save output to outputs/."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from self_correction_writing import build_graph_strategy3

TOPIC = "人工智慧對現代教育的影響"
CRITERIA = (
    "文章應結構清晰、論點明確，以具體證據支撐，"
    "語氣專業但易於理解，並同時探討 AI 的優勢與潛在風險。"
    "請以繁體中文撰寫。"
)

STATE = {
    "topic": TOPIC,
    "criteria": CRITERIA,
    "current_draft": "",
    "revision_history": [],
    "reflections": [],
    "iteration": 0,
    "max_iterations": 3,
    "score": 0.0,
    "score_threshold": 0.95,
    "critique": "",
    "final_output": "",
    "retrieved_memories": [],
    "graded_memories": [],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def format_s3(result):
    lines = []
    lines.append("=" * 70)
    lines.append("  策略三：Reflexion（記憶式自我修正）")
    lines.append("=" * 70)

    lines.append(f"\n迭代次數：{result.get('iteration', 0)}")
    lines.append(f"最終分數：{result.get('score', 0.0)}")

    reflections = result.get("reflections", [])
    if reflections:
        lines.append("\n【反思記憶庫】\n")
        for i, r in enumerate(reflections, 1):
            lines.append(f"  {i}. {r}")
            lines.append("")

    lines.append("=" * 70)
    lines.append("  最終文章")
    lines.append("=" * 70)
    lines.append(result.get("final_output", ""))

    return "\n".join(lines)


def main():
    print("執行策略三（Reflexion + HITL）...")
    print("提示：圖內建中斷點，自動執行時會自動放行。\n")
    graph = build_graph_strategy3()
    config = {"configurable": {"thread_id": "auto-run"}}

    # 自動跑完所有迭代：每次中斷後直接 resume
    for event in graph.stream(dict(STATE), config=config):
        pass
    while True:
        snapshot = graph.get_state(config)
        if snapshot.next == ():
            break
        for event in graph.stream(None, config=config):
            pass
    result = graph.get_state(config).values

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "strategy3_reflexion.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_s3(result))
    print(f"已儲存: {path}")

    print(f"\n迭代次數：{result.get('iteration', 0)}")
    print(f"最終分數：{result.get('score', 0.0)}")
    print(f"文章長度：{len(result.get('final_output', ''))} 字元")
    print("\n完成！")


if __name__ == "__main__":
    main()
