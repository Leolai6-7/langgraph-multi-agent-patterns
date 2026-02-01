"""Supervisor pattern state definition.

訊息隔離設計：
  messages (公共區)   — Supervisor 看得到，只放摘要級別的資訊
  scratchpad (私有區) — Worker 的中間數據（原始搜尋結果、重試紀錄等）
                        Supervisor 不會讀取，避免上下文污染
"""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str  # Supervisor 的路由決策: "Researcher" | "Writer" | "FINISH"
    scratchpad: dict  # Worker 私有暫存區，Supervisor 不讀取
