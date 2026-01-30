"""Strategy 3: Reflexion + Human-in-the-Loop demo.

人類作為「高階導師」，在 reflector 之前介入，修正 evaluator 的評語 (critique)，
確保寫入反思記憶庫的教訓是正確的。

流程：
  retrieve_memory → grade_relevance → generate → evaluator
      ↑                                           ↓
      └── reflector ←─── [INTERRUPT] ←────────────┘
                                                   ↓
                                          [INTERRUPT] → finalize → END

HITL 介入點：evaluator 之後、reflector 之前
  - 人類檢視 evaluator 的 score + critique
  - 以 as_node="evaluator" 偽裝，覆寫 critique（修正錯誤歸因）
  - reflector 基於修正後的 critique 生成正確的反思記憶
"""

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


def main():
    graph = build_graph_strategy3()
    config = {"configurable": {"thread_id": "hitl-demo"}}

    print("=" * 70)
    print("  策略三：Reflexion + Human-in-the-Loop")
    print("=" * 70)
    print(f"主題：{TOPIC}")
    print(f"標準：{CRITERIA}\n")

    # 首次執行：generator → evaluator → 中斷
    for event in graph.stream(STATE, config=config):
        _print_event(event)

    while True:
        snapshot = graph.get_state(config)
        state = snapshot.values

        if snapshot.next == ():
            _print_final(state)
            break

        _print_review(state, snapshot)
        choice = _ask_human()

        if choice == "v":
            print("\n" + "=" * 70)
            print(state.get("current_draft", ""))
            print("=" * 70 + "\n")
            continue

        if choice == "q":
            print("使用者中止。")
            break

        # 人類介入 — 以 as_node="evaluator" 偽裝成評審的最終輸出
        if choice == "d":
            new_critique = input("請輸入修正後的評語：")
            graph.update_state(config, {"critique": new_critique}, as_node="evaluator")
            print(f"已覆寫評語: {new_critique[:100]}")

        # choice == "c": 不呼叫 update_state，直接放行

        # 恢復執行
        for event in graph.stream(None, config=config):
            _print_event(event)


# ── 輔助函式 ────────────────────────────────────────────────────────

def _ask_human() -> str:
    print("\n[c] 放行（不修改）")
    print("[v] 檢視完整文章")
    print("[d] 修正評語（修正 evaluator 的錯誤歸因）")
    print("[q] 結束")
    return input("請選擇: ").strip().lower()


def _print_review(state: dict, snapshot):
    score = state.get("score", 0.0)
    threshold = state.get("score_threshold", 0.95)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    critique = state.get("critique", "")
    next_node = snapshot.next[0] if snapshot.next else "?"

    print("\n" + "-" * 70)
    print(f"  Evaluator 完成 → 條件路由選擇: {next_node}")
    print("-" * 70)
    print(f"分數：{score}（閾值：{threshold}）")
    print(f"迭代：{iteration} / {max_iter}")

    if critique:
        print(f"\n【Evaluator 評語】")
        print(f"  {critique}")

    if next_node == "finalize":
        print("\n→ 即將結束流程（finalize）")
    else:
        print("\n→ 即將進入反思與重寫（reflector → retrieve_memory → grade_relevance → generate）")

    reflections = state.get("reflections", [])
    if reflections:
        print("\n【反思記憶庫】")
        for i, r in enumerate(reflections, 1):
            print(f"  {i}. {r}")
        print()

    draft = state.get("current_draft", "")
    if draft:
        print(f"目前文章長度：{len(draft)} 字元")


def _print_final(state: dict):
    print("\n" + "=" * 70)
    print("  流程結束")
    print("=" * 70)
    print(f"迭代次數：{state.get('iteration', 0)}")
    print(f"最終分數：{state.get('score', 0.0)}")
    final = state.get("final_output", "")
    print(f"文章長度：{len(final)} 字元")
    print("\n" + "-" * 70)
    print(final)
    print("-" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "strategy3_reflexion_hitl.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(final)
    print(f"已儲存: {path}")


def _print_event(event: dict):
    for node_name, output in event.items():
        if node_name == "__interrupt__":
            continue
        print(f"\n>>> 節點 [{node_name}] 完成")
        if not output:
            continue
        if "score" in output:
            print(f"    分數: {output['score']}")
        if "critique" in output:
            print(f"    評語: {output['critique'][:100]}")
        if "iteration" in output:
            print(f"    迭代: {output['iteration']}")
        if "current_draft" in output:
            print(f"    文章長度: {len(output['current_draft'])} 字元")
        if "reflections" in output:
            for r in output["reflections"]:
                print(f"    反思: {r[:100]}...")
        if "final_output" in output:
            print(f"    最終輸出: {len(output['final_output'])} 字元")


if __name__ == "__main__":
    main()
