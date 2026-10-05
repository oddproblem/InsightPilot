from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RetrievedChunk(TypedDict, total=False):
    """Structured representation of a document chunk retrieved from pgvector."""

    id: str
    content: str
    source: str
    page: int | None
    score: float
    metadata: dict[str, Any]


class EvidenceItem(TypedDict, total=False):
    """Atomic factual statement extracted from retrieved documents or tool results."""

    statement: str
    source: str
    verified: bool


class Citation(TypedDict, total=False):
    """Verified citation tying generated statements to specific source snippets."""

    source: str
    snippet: str
    verified: bool
    page: int | None


class ToolResult(TypedDict, total=False):
    """Execution output from specialized agent tools (e.g., calculator, web search)."""

    tool_name: str
    tool_input: dict[str, Any] | str
    output: Any
    status: str


class AgentState(TypedDict, total=False):
    """Primary state schema for the InsightPilot agentic workflow graph.

    Adheres to LangGraph's functional state model:
    - `messages` uses the `add_messages` reducer to accumulate conversational turns.
    - Specialized fields track agent reasoning, tool dispatching, evidence extraction,
      and citation validation across the pipeline.
    """

    # ── Core Messaging & Session Tracking ────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    tenant_id: str
    run_id: str
    input_tokens: int
    output_tokens: int

    # ── Context & Reasoning Flow ─────────────────────────────────────────────
    context: list[str]  # Legacy text context for prompt injection
    query: str  # Current normalized user query
    route: str  # Decided execution route ("document_search", "web_search", "calculator", "direct")

    # ── Retrieval & Tool Results ─────────────────────────────────────────────
    retrieved_documents: list[RetrievedChunk]
    web_results: list[dict[str, Any]]
    tool_results: list[ToolResult]

    # ── Evidence Synthesis & Citations ───────────────────────────────────────
    evidence: list[EvidenceItem]
    citations: list[Citation]
    answer: str
    confidence: str  # "high" | "medium" | "low"

    # ── Policy, Guards & Safety ──────────────────────────────────────────────
    safety_flags: list[str]
    insufficient_evidence: bool
