"""Supervisor pattern agent nodes: supervisor, researcher, writer."""

from __future__ import annotations

from typing import Any, Literal

from langchain_aws import ChatBedrock
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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


_supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a supervisor managing a research-and-writing team.\n"
            "Available workers: {workers}.\n\n"
            "Given the conversation so far, decide which worker should act next, "
            "or respond with FINISH if the task is complete.\n"
            "Rules:\n"
            "- If no research has been done yet, route to Researcher.\n"
            "- If research results exist but no summary has been written, route to Writer.\n"
            "- If a satisfactory summary already exists, respond FINISH.\n",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "human",
            "Given the conversation above, who should act next? "
            "Respond with one of: {workers}, or FINISH.",
        ),
    ]
)

_supervisor_chain = (
    _supervisor_prompt
    | _llm.with_structured_output(RouteResponse)
)


def supervisor_node(state: SupervisorState) -> dict[str, Any]:
    """Supervisor：讀取對話，決定下一步路由。"""
    result = _supervisor_chain.invoke(
        {"messages": state["messages"], "workers": ", ".join(WORKERS)}
    )
    return {"next": result.next}


# ── Researcher ───────────────────────────────────────────────────────
_search_tool = DuckDuckGoSearchResults(num_results=4)


def researcher_node(state: SupervisorState) -> dict[str, Any]:
    """Researcher：從最後一條 HumanMessage 提取關鍵字，執行 DuckDuckGo 搜尋。"""
    # 找最後一條 HumanMessage 作為搜尋查詢
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        return {"messages": [AIMessage(content="[Researcher] No query found.")]}

    search_results = _search_tool.invoke(query)
    return {
        "messages": [
            AIMessage(content=f"[Researcher] Search results for '{query}':\n\n{search_results}")
        ]
    }


# ── Writer ───────────────────────────────────────────────────────────
_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional writer. Based on the research results in the "
            "conversation, write a concise and well-structured summary in the same "
            "language as the user's original query. "
            "Include key findings and cite sources when possible.",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Please write a summary based on the research above."),
    ]
)

_writer_chain = _writer_prompt | _llm


def writer_node(state: SupervisorState) -> dict[str, Any]:
    """Writer：根據搜尋結果撰寫摘要。"""
    result = _writer_chain.invoke({"messages": state["messages"]})
    return {
        "messages": [
            AIMessage(content=f"[Writer] Summary:\n\n{result.content}")
        ]
    }
