"""Whitepaper supervisor v3 agent nodes: supervisor, researcher, writer, editor.

v3 核心變更（相比 v2）：
  1. Supervisor 改為全 LLM 路由（每次都呼叫 LLM）
  2. Supervisor prompt 組裝包含 compressed_history + 近期 messages + 結構化指標
  3. Worker summary 資訊密度提升（含結果指標）
  4. 保留：WorkerHandoff、_apply_handoff、scratchpad 分離、Worker 名片、安全閥
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_aws import ChatBedrock
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .state import WhitepaperState, WorkerHandoff

# ── LLM ─────────────────────────────────────────────────────────────
_llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.7, "max_tokens": 4096},
)

# ── Supervisor 路由 ──────────────────────────────────────────────────
WORKERS = ["Researcher", "Writer", "Editor"]


class RouteResponse(BaseModel):
    """Supervisor routing decision."""

    reasoning: str  # 為什麼這樣選，方便 debug
    next: Literal["Researcher", "Writer", "Editor", "FINISH"]


# ── Worker 名片（給 Supervisor 看的）────────────────────────────────
_WORKER_DESCRIPTIONS = {
    "Researcher": (
        "能力：使用搜尋工具蒐集客觀資料，輸出原始搜尋結果與來源。\n"
        "觸發條件：當目前沒有任何研究資料，或審稿意見指出「缺乏數據」"
        "「資料不足」「需要更多來源佐證」等資料缺口問題時使用。\n"
        "邊界：只負責蒐集資料。不會撰寫文章，不會審稿，不會做決策。"
    ),
    "Writer": (
        "能力：根據現有研究資料撰寫 Markdown 格式的技術白皮書。\n"
        "觸發條件：當研究資料已充足但尚無草稿時使用；"
        "或當審稿意見指出「語氣問題」「結構不完整」「表達需修改」等寫作品質問題時使用。\n"
        "邊界：只負責寫作。不會自行搜尋資料，不會審稿，不會做決策。"
        "如果資料不足，它無法產出好的文章——這時應該先找 Researcher。"
    ),
    "Editor": (
        "能力：審核白皮書草稿，輸出具體的審稿意見（哪裡好、哪裡有問題）。\n"
        "觸發條件：當草稿已完成但尚未審核時使用。\n"
        "邊界：只負責審稿。不會改寫文章，不會搜尋資料，不會做決策。"
        "它的意見是客觀描述，由你（主管）判斷下一步。"
    ),
}

# ── Supervisor：全 LLM 路由 ───────────────────────────────────────
_SUPERVISOR_SYSTEM = (
    "你是一位白皮書專案主管。你的職責是根據目前的工作進展，決定下一步由誰執行。\n\n"
    "你的團隊成員如下，請根據各自的能力和觸發條件來分配工作：\n\n"
    "{worker_cards}\n\n"
    "=== 決策規則 ===\n"
    "1. 初始狀態（沒有任何進展）→ Researcher（先蒐集資料）\n"
    "2. 有資料但沒有草稿 → Writer（開始寫作）\n"
    "3. 有草稿但未審核 → Editor（進行審核）\n"
    "4. 審稿指出資料缺口 → Researcher（補充搜尋）\n"
    "5. 審稿指出寫作問題 → Writer（修改草稿）\n"
    "6. Writer 剛完成修改 → Editor（每次修改後必須重新審核，不可跳過）\n"
    "7. 審稿通過（品質良好、無重大問題）→ FINISH\n"
    "8. 已修改 3 輪以上 → FINISH（避免無限循環）\n\n"
    "請根據以下上下文資訊做出判斷。\n"
)

_supervisor_llm = _llm.with_structured_output(RouteResponse)


def _apply_handoff(
    actor: str, handoff: WorkerHandoff, scratchpad: dict,
) -> dict[str, Any]:
    """將 WorkerHandoff 轉換為 LangGraph state update。"""
    scratchpad.update(handoff["artifacts"])
    return {
        "messages": [HumanMessage(content=f"[{actor}] {handoff['summary']}")],
        "scratchpad": scratchpad,
        "last_actor": actor,
    }


def supervisor_node(state: WhitepaperState) -> dict[str, Any]:
    """Supervisor：全 LLM 路由。

    每次都呼叫 LLM，但提供結構化的完整上下文：
      - compressed_history（長期記憶）
      - 近期 messages（短期記憶）
      - 結構化指標（code 組出）
      - Worker 名片
      - Editor 完整意見（僅 last_actor=="Editor" 時 page-in）
    """
    last_actor = state.get("last_actor", "")
    scratchpad = state.get("scratchpad", {})
    revision_count = scratchpad.get("revision_count", 0)
    compressed_history = state.get("compressed_history", "")
    messages = state.get("messages", [])

    # ── 安全閥 ──
    if revision_count >= 3:
        print("  [Supervisor] 安全閥觸發 (revision_count >= 3) → FINISH")
        return {"next": "FINISH"}

    # ── 組裝 Supervisor prompt ──
    worker_cards = "\n\n".join(
        f"【{name}】\n{desc}" for name, desc in _WORKER_DESCRIPTIONS.items()
    )

    context_parts = []

    # 1. 歷史摘要（長期記憶）
    if compressed_history:
        context_parts.append(f"=== 歷史摘要 ===\n{compressed_history}")

    # 2. 近期進展（短期記憶：未壓縮的 messages）
    if messages:
        recent_lines = "\n".join(f"- {msg.content}" for msg in messages)
        context_parts.append(f"=== 近期進展 ===\n{recent_lines}")

    # 3. 當前狀況（結構化指標）
    research_count = scratchpad.get("research_count", 0)
    has_draft = bool(scratchpad.get("current_draft", ""))
    has_research = bool(scratchpad.get("research_data", ""))

    status_brief = (
        f"=== 當前狀況 ===\n"
        f"審稿輪次：第 {revision_count} 輪\n"
        f"Researcher 已搜尋：{research_count} 次\n"
        f"Writer 已修改：{max(0, revision_count - 1) if revision_count > 0 else 0} 次\n"
        f"是否有研究資料：{'是' if has_research else '否'}\n"
        f"是否有草稿：{'是' if has_draft else '否'}\n"
        f"上一位完成者：{last_actor or '（無，初始狀態）'}"
    )
    context_parts.append(status_brief)

    # 4. Editor 完整意見（僅 last_actor=="Editor" 時 page-in）
    if last_actor == "Editor":
        editor_critique = scratchpad.get("editor_critique", "")
        if editor_critique:
            context_parts.append(f"=== Editor 完整意見 ===\n{editor_critique}")

    context_text = "\n\n".join(context_parts)

    result = _supervisor_llm.invoke([
        SystemMessage(
            content=_SUPERVISOR_SYSTEM.format(worker_cards=worker_cards)
        ),
        HumanMessage(content=context_text if context_text else "（初始狀態，尚無任何進展）"),
    ])

    print(f"  [Supervisor reasoning] {result.reasoning}")
    print(f"  [Supervisor] → {result.next}")
    return {"next": result.next}


# ── Researcher ───────────────────────────────────────────────────────
_search_tool = DuckDuckGoSearchResults(num_results=4)

_RESEARCHER_SYSTEM = (
    "你是一位專業的資料研究員。你的唯一職責是使用搜尋工具蒐集客觀資料。\n\n"
    "輸入：你會收到一個主題，以及可能的「補充需求」描述。\n"
    "輸出：針對主題的搜尋關鍵字。只回覆關鍵字本身，不要加任何說明文字。\n\n"
    "你不是作家：不要將搜尋結果改寫成文章或摘要。\n"
    "你不是決策者：不要提供建議或判斷。\n"
    "你不負責排版：只輸出原始數據與來源。\n\n"
    "若收到「補充需求」，代表先前的搜尋結果有資料缺口，"
    "請根據缺口描述產出更精準的補充搜尋關鍵字。\n"
)


def researcher_node(state: WhitepaperState) -> dict[str, Any]:
    """Researcher：蒐集客觀資料。

    v3 提升：summary 包含涵蓋主題與資料量。
    """
    scratchpad = dict(state.get("scratchpad", {}))

    topic = scratchpad.get("topic", "")
    if not topic:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                topic = msg.content
                break

    if not topic:
        handoff: WorkerHandoff = {
            "summary": "找不到搜尋查詢。",
            "status": "FAILED",
            "artifacts": {},
        }
        return _apply_handoff("Researcher", handoff, scratchpad)

    editor_critique = scratchpad.get("editor_critique", "")
    user_content = f"主題：{topic}"
    if editor_critique:
        user_content += f"\n\n補充需求：{editor_critique}"

    refine_result = _llm.invoke([
        SystemMessage(content=_RESEARCHER_SYSTEM),
        HumanMessage(content=user_content),
    ])
    query = refine_result.content.strip()

    raw_results = _search_tool.invoke(query)

    result_count = raw_results.count("snippet:") if isinstance(raw_results, str) else 0

    existing_data = scratchpad.get("research_data", "")
    if existing_data and editor_critique:
        combined_data = f"{existing_data}\n\n--- 補充搜尋 ({query}) ---\n{raw_results}"
    else:
        combined_data = raw_results

    research_count = scratchpad.get("research_count", 0) + 1

    # v3 提升：summary 包含涵蓋主題與資料量
    data_len = len(combined_data) if isinstance(combined_data, str) else 0
    # 嘗試提取涵蓋主題（從 snippet 中取前幾個關鍵詞）
    topics_covered = query.replace(",", "、")

    handoff: WorkerHandoff = {
        "summary": (
            f"已完成搜尋 '{query}'，找到 {result_count} 筆結果，"
            f"涵蓋主題：{topics_covered}。資料量：{data_len} 字元。"
        ),
        "status": "SUCCESS",
        "artifacts": {
            "topic": topic,
            "research_query": query,
            "research_data": combined_data,
            "research_count": research_count,
        },
    }
    return _apply_handoff("Researcher", handoff, scratchpad)


# ── Writer ───────────────────────────────────────────────────────────
_WRITER_SYSTEM = (
    "你是一位專業的技術白皮書撰稿人。\n\n"
    "輸入：你會收到研究資料，以及可能的「修改建議」。\n"
    "輸出：一份完整的 Markdown 格式技術白皮書。\n\n"
    "格式要求：\n"
    "- 使用企業級正式語氣，避免口語化表達\n"
    "- 必須包含以下結構：\n"
    "  1. 摘要 (Executive Summary)\n"
    "  2. 背景與問題描述\n"
    "  3. 技術分析（至少 2-3 個章節）\n"
    "  4. 結論與建議\n"
    "  5. 參考來源\n"
    "- 所有論點必須有提供的資料佐證，避免無根據的陳述\n"
    "- 使用與主題相同的語言\n\n"
    "你不是研究員：不要自行編造數據或來源，只使用提供的資料。\n"
    "你不是審稿人：不要自我評價文章品質，專注於寫作。\n"
    "你不是決策者：不要建議「下一步該做什麼」。\n"
)


def writer_node(state: WhitepaperState) -> dict[str, Any]:
    """Writer：從 scratchpad 讀取資料，產出 Markdown 白皮書。

    v3 提升：summary 包含字元數與章節數。
    """
    scratchpad = dict(state.get("scratchpad", {}))
    research_data = scratchpad.get("research_data", "（無搜尋結果）")
    editor_critique = scratchpad.get("editor_critique", "")
    topic = scratchpad.get("topic", "")

    user_content = f"主題：{topic}\n\n研究資料：\n{research_data}\n\n"

    if editor_critique:
        user_content += f"請根據以下審稿修改建議修正草稿：\n{editor_critique}\n\n"
        user_content += f"上一版草稿：\n{scratchpad.get('current_draft', '')}\n"
        action = "修改"
    else:
        user_content += "請根據以上資料撰寫技術白皮書。"
        action = "完成"

    result = _llm.invoke([
        SystemMessage(content=_WRITER_SYSTEM),
        HumanMessage(content=user_content),
    ])

    draft = result.content
    draft_len = len(draft)
    # 計算章節數（以 ## 開頭的行）
    section_count = sum(1 for line in draft.split("\n") if line.strip().startswith("## "))

    handoff: WorkerHandoff = {
        "summary": f"草稿已{action}，共 {draft_len} 字元，包含 {section_count} 個章節。",
        "status": "SUCCESS",
        "artifacts": {
            "current_draft": draft,
        },
    }
    return _apply_handoff("Writer", handoff, scratchpad)


# ── Editor ───────────────────────────────────────────────────────────
_EDITOR_SYSTEM = (
    "你是一位嚴格的技術白皮書審稿編輯。\n\n"
    "輸入：你會收到一份技術白皮書草稿。\n"
    "輸出：針對草稿的具體審稿意見。\n\n"
    "審核面向：\n"
    "1. 企業語氣是否一致（正式、專業、避免口語化）\n"
    "2. 是否有未佐證的說法（缺乏數據支撐的論點）\n"
    "3. 結構完整性（是否包含摘要、章節、結論）\n"
    "4. 資料充分度（現有資料是否足以支撐論點）\n\n"
    "請直接描述你觀察到的問題：\n"
    "- 哪些部分做得好\n"
    "- 哪些部分有問題（指出具體位置和問題類型）\n"
    "- 具體的修改建議\n\n"
    "你不是作家：不要幫忙改寫文章內容。\n"
    "你不是研究員：不要自行補充數據。\n"
    "你不是決策者：不要指示下一步該做什麼或該由誰處理。\n"
)


def editor_node(state: WhitepaperState) -> dict[str, Any]:
    """Editor：審核草稿，只提供意見。

    v3 提升：summary 包含問題數量與類型分類。
    """
    scratchpad = dict(state.get("scratchpad", {}))
    current_draft = scratchpad.get("current_draft", "")
    revision_count = scratchpad.get("revision_count", 0)

    result = _llm.invoke([
        SystemMessage(content=_EDITOR_SYSTEM),
        HumanMessage(content=f"請審核以下技術白皮書草稿：\n\n{current_draft}"),
    ])

    critique = result.content
    revision_count += 1

    # v3 提升：summary 包含問題數量與分類
    # 簡單啟發式：計算「問題」「不足」「缺乏」等關鍵詞出現次數
    issue_keywords = ["問題", "不足", "缺乏", "需要", "建議修改", "不夠", "缺少"]
    merit_keywords = ["做得好", "優點", "完整", "清晰", "良好"]
    issue_count = sum(critique.count(kw) for kw in issue_keywords)
    merit_count = sum(critique.count(kw) for kw in merit_keywords)

    # 簡化分類
    categories = []
    if any(kw in critique for kw in ["語氣", "口語", "正式"]):
        categories.append("語氣")
    if any(kw in critique for kw in ["資料不足", "缺乏數據", "佐證", "來源"]):
        categories.append("資料不足")
    if any(kw in critique for kw in ["結構", "章節", "段落"]):
        categories.append("結構")
    if not categories:
        categories.append("一般品質")

    category_str = "、".join(categories)

    summary = (
        f"審稿意見（第 {revision_count} 輪）："
        f"發現 {issue_count} 個問題（{category_str}），"
        f"{merit_count} 個優點。"
    )

    handoff: WorkerHandoff = {
        "summary": summary,
        "status": "SUCCESS",
        "artifacts": {
            "editor_critique": critique,
            "revision_count": revision_count,
        },
    }
    return _apply_handoff("Editor", handoff, scratchpad)
