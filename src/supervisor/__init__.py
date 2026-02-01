"""Supervisor pattern – LLM-based router with Researcher and Writer workers."""

from .graph import build_supervisor_graph

__all__ = ["build_supervisor_graph"]
