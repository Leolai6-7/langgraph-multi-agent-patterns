"""Whitepaper supervisor agent nodes: supervisor, researcher, writer, editor.

設計原則：
  1. 上下文隔離 — messages (公共區) vs scratchpad (私有區)
  2. 決策權歸 Supervisor — Worker 不知道彼此存在，不做路由決策
  3. 孤獨的專才 — 每個 Worker 只知道：我是誰、我的輸入、我的輸出、我不做什麼
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_aws import ChatBedrock
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .state import WhitepaperState

# ── LLM ─────────────────────────────────────────────────────────────
_llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.7, "max_tokens": 4096},
)

# ── Supervisor 路由 ──────────────────────────────────────────────────
WORKERS = ["Researcher", "Writer", "Editor"]


class RouteResponse(BaseModel):
    """Supervisor routing decision."""

    reasoning: str  # 為什麼這樣選，方便 debug
    next: Literal["Researcher", "Writer", "Editor", "FINISH"]


# ── Worker 名片（給 Supervisor 看的，不是給 Worker 自己看的）────────
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

# ── Supervisor：確定性路由 + LLM 判斷 Editor 意見 ─────────────────
# 只有 Editor 完成後需要 LLM 判斷意見內容，其他轉移都是確定性的。

_EDITOR_JUDGE_SYSTEM = (
    "你是一位主管。Editor 剛完成審稿，以下是他的意見。\n"
    "請仔細閱讀意見內容，判斷下一步：\n\n"
    "=== 團隊成員 ===\n"
    "{worker_cards}\n\n"
    "=== 判斷規則 ===\n"
    "- 如果意見指出「缺乏數據」「資料不足」「需要更多來源佐證」"
    "等資料缺口問題 → Researcher\n"
    "- 如果意見指出「語氣問題」「結構不完整」「表達需修改」"
    "等寫作品質問題 → Writer\n"
    "- 如果意見表示品質良好、無重大問題 → FINISH\n"
)

_supervisor_llm = _llm.with_structured_output(RouteResponse)


def supervisor_node(state: WhitepaperState) -> dict[str, Any]:
    """Supervisor：混合路由。

    確定性轉移用 code 寫死（零 LLM 成本）：
      初始 → Researcher
      Researcher → Writer
      Writer → Editor

    只有 Editor 完成後才呼叫 LLM 判斷意見內容：
      Editor → Researcher / Writer / FINISH
    """
    last_actor = state.get("last_actor", "")
    scratchpad = state.get("scratchpad", {})
    revision_count = scratchpad.get("revision_count", 0)

    # ── 安全閥 ──
    if revision_count >= 3:
        print("  [Supervisor] 安全閥觸發 (revision_count >= 3) → FINISH")
        return {"next": "FINISH"}

    # ── 確定性路由 ──
    if last_actor == "":
        # 初始狀態，開始搜尋
        print("  [Supervisor] 初始狀態 → Researcher")
        return {"next": "Researcher"}

    if last_actor == "Researcher":
        print("  [Supervisor] Researcher 完成 → Writer")
        return {"next": "Writer"}

    if last_actor == "Writer":
        print("  [Supervisor] Writer 完成 → Editor")
        return {"next": "Editor"}

    # ── Editor 完成：唯一需要 LLM 判斷的地方 ──
    if last_actor == "Editor":
        # 只把最新的 Editor 意見（從 messages 最後一條）傳給 LLM
        editor_msg = ""
        for msg in reversed(state["messages"]):
            if msg.content.startswith("[Editor]"):
                editor_msg = msg.content
                break

        worker_cards = "\n\n".join(
            f"【{name}】\n{desc}" for name, desc in _WORKER_DESCRIPTIONS.items()
        )
        result = _supervisor_llm.invoke([
            SystemMessage(
                content=_EDITOR_JUDGE_SYSTEM.format(worker_cards=worker_cards)
            ),
            HumanMessage(content=f"Editor 的審稿意見：\n{editor_msg}"),
        ])
        print(f"  [Supervisor reasoning] {result.reasoning}")
        return {"next": result.next}

    # fallback（不應該到這裡）
    print(f"  [Supervisor] 未知的 last_actor: {last_actor} → FINISH")
    return {"next": "FINISH"}


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
    """Researcher：蒐集客觀資料的孤獨專才。

    不知道團隊存在，只負責：收到主題 → 搜尋 → 輸出原始結果。
    """
    scratchpad = dict(state.get("scratchpad", {}))

    # 取得原始主題
    topic = scratchpad.get("topic", "")
    if not topic:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                topic = msg.content
                break

    if not topic:
        return {
            "messages": [HumanMessage(content="[Researcher] 找不到搜尋查詢。")],
            "last_actor": "Researcher",
        }

    # 用 LLM 產出搜尋關鍵字（統一路徑，不論首次或補充）
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

    # 計算結果數量
    result_count = raw_results.count("snippet:") if isinstance(raw_results, str) else 0

    # 若是補充搜尋，合併舊的 research_data
    existing_data = scratchpad.get("research_data", "")
    if existing_data and editor_critique:
        combined_data = f"{existing_data}\n\n--- 補充搜尋 ({query}) ---\n{raw_results}"
    else:
        combined_data = raw_results

    scratchpad["topic"] = topic
    scratchpad["research_query"] = query
    scratchpad["research_data"] = combined_data

    return {
        "messages": [
            HumanMessage(
                content=f"[Researcher] 已完成搜尋 '{query}'，找到 {result_count} 筆結果。"
            ),
        ],
        "scratchpad": scratchpad,
        "last_actor": "Researcher",
    }


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

    - 讀取 research_data + editor_critique（如有）
    - 獨立 LLM 呼叫，不依賴 messages
    - 草稿 → scratchpad（私有區）
    - 一句話狀態 → messages（公共區）
    """
    scratchpad = dict(state.get("scratchpad", {}))
    research_data = scratchpad.get("research_data", "（無搜尋結果）")
    editor_critique = scratchpad.get("editor_critique", "")
    topic = scratchpad.get("topic", "")

    user_content = f"主題：{topic}\n\n研究資料：\n{research_data}\n\n"

    if editor_critique:
        user_content += f"請根據以下審稿修改建議修正草稿：\n{editor_critique}\n\n"
        user_content += f"上一版草稿：\n{scratchpad.get('current_draft', '')}\n"
        status_msg = "[Writer] 草稿已根據審稿意見修改。"
    else:
        user_content += "請根據以上資料撰寫技術白皮書。"
        status_msg = "[Writer] 草稿已完成。"

    result = _llm.invoke([
        SystemMessage(content=_WRITER_SYSTEM),
        HumanMessage(content=user_content),
    ])

    scratchpad["current_draft"] = result.content

    return {
        "messages": [HumanMessage(content=status_msg)],
        "scratchpad": scratchpad,
        "last_actor": "Writer",
    }


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
    """Editor：審核草稿，只提供意見，不做路由判定。

    - 從 scratchpad 讀取 current_draft
    - 獨立 LLM 呼叫審核
    - critique → scratchpad（私有區，供 Writer/Researcher 參考）
    - 審稿意見摘要 → messages（公共區，供 Supervisor 判斷路由）
    """
    scratchpad = dict(state.get("scratchpad", {}))
    current_draft = scratchpad.get("current_draft", "")
    revision_count = scratchpad.get("revision_count", 0)

    result = _llm.invoke([
        SystemMessage(content=_EDITOR_SYSTEM),
        HumanMessage(content=f"請審核以下技術白皮書草稿：\n\n{current_draft}"),
    ])

    critique = result.content

    # 更新 scratchpad
    revision_count += 1
    scratchpad["editor_critique"] = critique
    scratchpad["revision_count"] = revision_count

    # 公共區放審稿意見摘要（截取前 200 字），讓 Supervisor 能判斷
    summary = critique[:200]
    if len(critique) > 200:
        summary += "..."
    status_msg = f"[Editor] 審稿意見（第 {revision_count} 輪）：{summary}"

    # 安全閥提示
    if revision_count >= 3:
        status_msg += "\n（revision_count >= 3，安全閥觸發）"

    return {
        "messages": [HumanMessage(content=status_msg)],
        "scratchpad": scratchpad,
        "last_actor": "Editor",
    }
