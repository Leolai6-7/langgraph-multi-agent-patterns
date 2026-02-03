"""Whitepaper supervisor v3 graph builder.

v3 核心變更：Workers → compress → Supervisor（壓縮節點插入在中間）
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import editor_node, researcher_node, supervisor_node, writer_node
from .compress import compress_messages
from .state import WhitepaperState


def build_whitepaper_graph_v3():
    """Build and compile the Whitepaper Supervisor v3 graph.

    Flow:
        START → Supervisor → (Researcher | Writer | Editor | FINISH→END)
        Researcher → compress → Supervisor
        Writer     → compress → Supervisor
        Editor     → compress → Supervisor

    壓縮節點在 Worker 完成後、Supervisor 決策前觸發，
    當 messages 數量 >= 6 時壓縮舊訊息為 compressed_history。
    """
    builder = StateGraph(WhitepaperState)

    builder.add_node("Supervisor", supervisor_node)
    builder.add_node("Researcher", researcher_node)
    builder.add_node("Writer", writer_node)
    builder.add_node("Editor", editor_node)
    builder.add_node("compress", compress_messages)

    builder.add_edge(START, "Supervisor")

    # Workers → compress → Supervisor
    for member in ["Researcher", "Writer", "Editor"]:
        builder.add_edge(member, "compress")
    builder.add_edge("compress", "Supervisor")

    builder.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {
            "Researcher": "Researcher",
            "Writer": "Writer",
            "Editor": "Editor",
            "FINISH": END,
        },
    )

    return builder.compile()
