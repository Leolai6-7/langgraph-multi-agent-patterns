"""Whitepaper supervisor v3 state definition.

v3 核心變更：
  - messages 角色從「人類日誌」升級為「LLM 工作記憶」
  - reducer 從 operator.add 改為 add_messages（支援 RemoveMessage 壓縮）
  - 新增 compressed_history 欄位存放壓縮後的長期記憶
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── 結構化交接協議（同 v2）──────────────────────────────────────
WorkerStatus = Literal["SUCCESS", "FAILED"]


class WorkerHandoff(TypedDict):
    """Worker → Supervisor 的結構化交接物件。"""

    summary: str         # 控制流：一句話摘要，進入 messages
    status: WorkerStatus  # 控制流：明確狀態碼，給狀態機判斷
    artifacts: dict      # 資料流：要寫入 scratchpad 的欄位


class WhitepaperState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 改用 add_messages（支援 RemoveMessage）
    next: str  # 路由決策: "Researcher" | "Writer" | "Editor" | "FINISH"
    last_actor: str  # 上一個完成的 Worker
    scratchpad: dict  # 資料流（同 v2）
    compressed_history: str  # 壓縮後的歷史摘要（長期記憶）
