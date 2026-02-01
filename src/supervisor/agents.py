"""Supervisor pattern agent nodes: supervisor, researcher, writer.

設計原則：
  1. 上下文隔離 — messages (公共區) vs scratchpad (私有區)
  2. 純 Prompt 路由 — Supervisor 只用 structured_output 分類，不產生任何訊息
  3. 精簡歷史 — Supervisor 看到的只有 [需求] → [Worker 產出] → [判斷]
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_aws import ChatBedrock
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .state import SupervisorState

# ── LLM ─────────────────────────────────────────────────────────────
_llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.7, "max_tokens": 4096},
)

# ── Supervisor 路由 ──────────────────────────────────────────────────
WORKERS = ["Researcher", "Writer"]


class RouteResponse(BaseModel):
    """Supervisor routing decision."""

    next: Literal["Researcher", "Writer", "FINISH"]


_SUPERVISOR_SYSTEM = (
    "你是一位主管，負責管理一個研究與寫作團隊。\n"
    "可用的工作者：{workers}。\n\n"
    "根據目前的對話紀錄，決定下一步該由哪個工作者執行，"
    "或者如果任務已完成則回覆 FINISH。\n"
    "規則：\n"
    "- 如果尚未進行任何搜尋研究，請路由到 Researcher。\n"
    "- 如果已有搜尋結果但尚未撰寫摘要，請路由到 Writer。\n"
    "- 如果已經有完整的摘要，請回覆 FINISH。\n"
)

_supervisor_llm = _llm.with_structured_output(RouteResponse)


def supervisor_node(state: SupervisorState) -> dict[str, Any]:
    """Supervisor：純路由，不產生任何訊息。

    只讀取 messages，用 structured_output 輸出 next，
    不往 messages 寫入任何東西，零 Token 浪費。
    """
    # 把 messages 壓縮成一段文字，塞進單一 HumanMessage
    # 避免 role alternation 問題，同時最省 Token
    conversation = "\n".join(
        f"[{msg.type}] {msg.content}" for msg in state["messages"]
    )
    result = _supervisor_llm.invoke([
        SystemMessage(content=_SUPERVISOR_SYSTEM.format(workers=", ".join(WORKERS))),
        HumanMessage(content=f"對話紀錄：\n{conversation}\n\n下一步該由誰執行？"),
    ])
    return {"next": result.next}


# ── Researcher ───────────────────────────────────────────────────────
_search_tool = DuckDuckGoSearchResults(num_results=4)


def researcher_node(state: SupervisorState) -> dict[str, Any]:
    """Researcher：執行搜尋。

    - 原始結果 → scratchpad（私有區）
    - 一句話摘要 → messages（公共區，給 Supervisor 判斷用）
    """
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        return {
            "messages": [HumanMessage(content="[Researcher] 找不到搜尋查詢。")],
        }

    raw_results = _search_tool.invoke(query)

    return {
        "messages": [
            HumanMessage(content=f"[Researcher] 已完成搜尋 '{query}'，找到相關結果。"),
        ],
        "scratchpad": {
            "research_query": query,
            "research_raw_results": raw_results,
        },
    }


# ── Writer ───────────────────────────────────────────────────────────
def writer_node(state: SupervisorState) -> dict[str, Any]:
    """Writer：從 scratchpad 讀取原始數據，撰寫摘要。

    獨立 LLM 呼叫，不依賴 messages 中的歷史。
    摘要結果 → messages（公共區）。
    """
    scratchpad = state.get("scratchpad", {})
    query = scratchpad.get("research_query", "")
    raw_results = scratchpad.get("research_raw_results", "（無搜尋結果）")

    result = _llm.invoke([
        SystemMessage(content=(
            "你是一位專業的撰稿人。根據提供的搜尋研究結果，"
            "撰寫一份簡潔且結構清晰的摘要，"
            "使用與使用者原始查詢相同的語言。"
            "請包含關鍵發現，並盡可能引用來源。"
        )),
        HumanMessage(content=(
            f"使用者查詢：{query}\n\n"
            f"原始搜尋結果：\n{raw_results}\n\n"
            "請根據以上資料撰寫一份摘要。"
        )),
    ])

    return {
        "messages": [
            HumanMessage(content=f"[Writer] 摘要：\n\n{result.content}"),
        ],
    }
