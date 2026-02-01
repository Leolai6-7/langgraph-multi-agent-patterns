"""Supervisor pattern state definition."""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str  # Supervisor 的路由決策: "Researcher" | "Writer" | "FINISH"
