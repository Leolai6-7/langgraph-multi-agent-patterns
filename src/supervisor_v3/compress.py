"""Message compression node for supervisor v3.

壓縮機制：
  觸發條件：len(messages) >= 6
  流程：
    1. 保留最新 1 條 message（剛完成的 Worker 回報）
    2. 取出其餘所有 messages
    3. LLM 壓縮為一段摘要
    4. 累積更新 compressed_history
    5. 用 RemoveMessage 刪除舊 messages
  不觸發時：直接 pass through
"""

from __future__ import annotations

from typing import Any

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from .state import WhitepaperState

_compress_llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0, "max_tokens": 1024},
)

COMPRESS_THRESHOLD = 6

_COMPRESS_SYSTEM = (
    "你是一個對話摘要壓縮器。你會收到一段多輪對話記錄，"
    "請將其壓縮為一段精簡的摘要。\n\n"
    "要求：\n"
    "- 保留關鍵事實：誰做了什麼、結果如何、重要數據指標\n"
    "- 保留時間順序\n"
    "- 移除重複資訊\n"
    "- 使用條列式，每項一句話\n"
    "- 使用與原文相同的語言\n"
)


def compress_messages(state: WhitepaperState) -> dict[str, Any]:
    """壓縮節點：Worker → compress → Supervisor。

    當 messages 數量 >= COMPRESS_THRESHOLD 時觸發壓縮，
    否則直接 pass through。
    """
    messages = state.get("messages", [])

    if len(messages) < COMPRESS_THRESHOLD:
        # 不觸發壓縮，直接通過
        return {}

    # 保留最新 1 條（剛完成的 Worker 回報）
    latest = messages[-1]
    old_messages = messages[:-1]

    # 組裝要壓縮的對話內容
    conversation_text = "\n".join(
        f"- {msg.content}" for msg in old_messages
    )

    # 加入既有的壓縮歷史
    existing_history = state.get("compressed_history", "")
    if existing_history:
        compress_input = (
            f"=== 先前的歷史摘要 ===\n{existing_history}\n\n"
            f"=== 新的對話記錄（需壓縮） ===\n{conversation_text}"
        )
    else:
        compress_input = f"=== 對話記錄（需壓縮） ===\n{conversation_text}"

    # LLM 壓縮
    result = _compress_llm.invoke([
        SystemMessage(content=_COMPRESS_SYSTEM),
        HumanMessage(content=compress_input),
    ])

    new_compressed_history = result.content.strip()

    # 用 RemoveMessage 刪除舊 messages（保留最新一條）
    remove_ops = [RemoveMessage(id=msg.id) for msg in old_messages]

    print(f"  [compress] 壓縮 {len(old_messages)} 條舊訊息 → compressed_history")
    print(f"  [compress] 保留最新 1 條: {latest.content[:80]}...")

    return {
        "messages": remove_ops,
        "compressed_history": new_compressed_history,
    }
