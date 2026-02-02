"""Whitepaper supervisor state definition.

訊息隔離設計：
  messages (公共區)   — Supervisor 看得到，只放摘要級別的資訊
  scratchpad (私有區) — Worker 的中間數據（原始搜尋結果、草稿、審稿意見等）
                        Supervisor 不會讀取，避免上下文污染
"""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class WhitepaperState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # 公共區：Supervisor 決策用
    next: str  # 路由決策: "Researcher" | "Writer" | "Editor" | "FINISH"
    last_actor: str  # 上一個完成的 Worker: "Researcher" | "Writer" | "Editor" | ""
    scratchpad: dict  # 私有區，結構如下：
    #   {
    #     "topic": str,                   # 使用者主題
    #     "research_query": str,          # 搜尋關鍵字
    #     "research_data": str,           # 原始搜尋結果
    #     "current_draft": str,           # Writer 最新草稿（Markdown）
    #     "editor_critique": str,         # Editor 的審稿意見（純描述，不含 verdict）
    #     "revision_count": int,          # 修改次數（防無限迴圈）
    #   }
