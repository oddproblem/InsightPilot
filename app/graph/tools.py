"""InsightPilot agentic tools.

Tool selection is the LLM's responsibility.  The tools here are intentionally
narrow: each does one thing and returns structured text that the agent can
reason about.

Tenant isolation
----------------
LangChain @tool functions have no access to agent state.  Tenant identity is
propagated via a Python contextvars.ContextVar (_tenant_ctx).  Call
    _tenant_ctx.set(tenant_id)
in tool_dispatch (nodes.py) before invoking any tool.

Registered tools (TOOLS list at the bottom, imported by graph.py):
  document_search        — semantic search over tenant-scoped pgvector chunks
  get_document_metadata  — fetch file-level metadata for a known document_id
  calculate              — evaluate safe mathematical expressions (no exec)
  web_search             — live Tavily search; disabled if key is absent
"""

import ast
import contextvars
import logging
import operator
import uuid
from typing import Any

from langchain_core.tools import BaseTool, tool
from openai import OpenAI

from app.config import config
from app.db.connection import db_conn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tenant context — set by tool_dispatch before any tool invocation
# ---------------------------------------------------------------------------

_tenant_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("tenant_id", default="default")

# ---------------------------------------------------------------------------
# Safe calculator — AST-based, no exec/eval of arbitrary code
# ---------------------------------------------------------------------------

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_MAX_EXPRESSION_LEN = 512


def _safe_eval(node: ast.expr) -> float:
    """Recursively evaluate an AST expression using only whitelisted operators."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        bin_op = type(node.op)
        if bin_op not in _SAFE_OPS:
            raise ValueError(f"Unsupported binary operator: {bin_op.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if bin_op is ast.Div and right == 0:
            raise ValueError("Division by zero")
        return float(_SAFE_OPS[bin_op](left, right))
    if isinstance(node, ast.UnaryOp):
        unary_op = type(node.op)
        if unary_op not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary operator: {unary_op.__name__}")
        return float(_SAFE_OPS[unary_op](_safe_eval(node.operand)))
    raise ValueError(f"Unsupported node type: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Tool: calculate
# ---------------------------------------------------------------------------


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Use for CAGR, growth rates, percentage changes, margin variance,
    and any arithmetic the user explicitly requests.

    Examples:
      (120 - 100) / 100 * 100
      ((150/100)**(1/3)-1)*100
      (500 - 450) / 450 * 100
    """
    if len(expression) > _MAX_EXPRESSION_LEN:
        return f"Expression too long (max {_MAX_EXPRESSION_LEN} chars)."

    # Reject expressions containing dangerous names / attribute access
    forbidden = ["__", "import", "exec", "eval", "open", "os", "sys", "subprocess"]
    lower_expr = expression.lower()
    for keyword in forbidden:
        if keyword in lower_expr:
            return f"Unsafe expression rejected: contains '{keyword}'."

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        rounded = round(result, 6)
        return f"Result: {rounded}"
    except (ValueError, ZeroDivisionError) as exc:
        return f"Calculation error: {exc}"
    except SyntaxError:
        return "Syntax error: could not parse expression."
    except Exception as exc:
        logger.warning("calculate_unexpected_error", extra={"expr": expression, "error": str(exc)})
        return f"Unexpected error: {exc}"


# ---------------------------------------------------------------------------
# Tool: document_search
# ---------------------------------------------------------------------------


@tool
def document_search(
    query: str,
    doc_type: str = "",
    company: str = "",
    financial_year: str = "",
) -> str:
    """Search the uploaded document knowledge base using semantic similarity.

    Use when the user asks about content from uploaded financial or business
    reports, 10-K, 10-Q, earnings releases, or internal documents.

    Args:
        query:          Natural-language search query.
        doc_type:       Optional filter — e.g. '10-K', '10-Q', 'earnings'.
        company:        Optional filter — e.g. 'Acme Corp'.
        financial_year: Optional filter — e.g. 'FY2025', '2024'.

    Returns formatted snippets with source, page, and similarity score.
    """
    # Tenant identity is injected by tool_dispatch via _tenant_ctx
    tenant_id = _tenant_ctx.get()

    try:
        openai_client = OpenAI(api_key=config.embedding_api_key)
        embed_resp = openai_client.embeddings.create(
            input=query,
            model=config.embedding_model,
        )
        embedding: list[float] = embed_resp.data[0].embedding

        filter_clauses = ["tenant_id = %(tenant_id)s"]
        params: dict[str, Any] = {
            "embedding": str(embedding),
            "tenant_id": tenant_id,
            "top_k": 5,
        }

        if doc_type:
            filter_clauses.append("doc_type ILIKE %(doc_type)s")
            params["doc_type"] = f"%{doc_type}%"
        if company:
            filter_clauses.append("company ILIKE %(company)s")
            params["company"] = f"%{company}%"
        if financial_year:
            filter_clauses.append("financial_year ILIKE %(financial_year)s")
            params["financial_year"] = f"%{financial_year}%"

        where_sql = " AND ".join(filter_clauses)

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id,
                        document_name,
                        content,
                        source,
                        page_number,
                        chunk_index,
                        doc_type,
                        company,
                        financial_year,
                        (embedding <=> %(embedding)s::vector) AS distance
                    FROM documents
                    WHERE {where_sql}
                    ORDER BY distance ASC
                    LIMIT %(top_k)s
                    """,
                    params,
                )
                rows = cur.fetchall()

        if not rows:
            return "No relevant documents found. Upload financial documents to enable retrieval."

        results: list[str] = []
        for row in rows:
            (
                _doc_id,
                doc_name,
                content,
                source,
                page,
                _chunk_idx,
                dtype,
                _co,
                fy,
                distance,
            ) = row
            score = round(1.0 - float(distance), 4)
            header_parts = [f"[{doc_name}"]
            if page:
                header_parts.append(f"p.{page}")
            if dtype:
                header_parts.append(dtype)
            if fy:
                header_parts.append(fy)
            header = " | ".join(header_parts) + f"] score={score}"
            results.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(results)

    except Exception as exc:
        logger.warning("document_search_failed", extra={"query": query, "error": str(exc)})
        return f"Document search failed: {exc}"


# ---------------------------------------------------------------------------
# Tool: get_document_metadata
# ---------------------------------------------------------------------------


@tool
def get_document_metadata(document_id: str) -> str:
    """Retrieve file-level metadata for a specific uploaded document.

    Use when you need to confirm a document's type, company, financial year,
    upload date, or total page/chunk count before citing it.

    Args:
        document_id: UUID of the document (returned in document_search results).
    """
    try:
        parsed_id = str(uuid.UUID(document_id))
    except ValueError:
        return f"Invalid document_id format: {document_id!r}"

    # Scope to the caller's tenant
    tenant_id = _tenant_ctx.get()

    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        document_id,
                        document_name,
                        source,
                        doc_type,
                        company,
                        financial_year,
                        MIN(upload_date) AS upload_date,
                        COUNT(*) AS chunk_count,
                        MAX(page_number) AS page_count
                    FROM documents
                    WHERE document_id = %(document_id)s AND tenant_id = %(tenant_id)s
                    GROUP BY document_id, document_name, source, doc_type, company, financial_year
                    """,
                    {"document_id": parsed_id, "tenant_id": tenant_id},
                )
                row = cur.fetchone()

        if row is None:
            return f"No document found with id={document_id}"

        doc_id, doc_name, source, dtype, company, fy, upload_date, chunks, pages = row
        lines = [
            f"Document ID:     {doc_id}",
            f"Name:            {doc_name}",
            f"Source:          {source}",
            f"Type:            {dtype or 'unknown'}",
            f"Company:         {company or 'unknown'}",
            f"Financial Year:  {fy or 'unknown'}",
            f"Upload Date:     {upload_date}",
            f"Chunk Count:     {chunks}",
            f"Page Count:      {pages or 'unknown'}",
        ]
        return "\n".join(lines)

    except Exception as exc:
        logger.warning(
            "get_document_metadata_failed",
            extra={"document_id": document_id, "error": str(exc)},
        )
        return f"Metadata retrieval failed: {exc}"


# ---------------------------------------------------------------------------
# Tool: web_search
# ---------------------------------------------------------------------------


@tool
def web_search(query: str) -> str:
    """Search the web for current, real-time information.

    Use ONLY when the user explicitly asks about recent events, live market
    data, news, or information that cannot be in uploaded documents.
    Do NOT use this for questions answerable from uploaded documents.
    """
    if not config.tavily_api_key:
        return "Web search is not configured. Set TAVILY_API_KEY to enable it."

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.tavily_api_key)
        results = client.search(query=query, max_results=3)
        formatted: list[str] = []
        for r in results.get("results", []):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "No content")
            formatted.append(f"**{title}**\n{url}\n{snippet}")
        return "\n\n".join(formatted) if formatted else "No results found."
    except Exception as exc:
        logger.warning("web_search_failed", extra={"query": query, "error": str(exc)})
        return f"Web search failed: {exc}"


# ---------------------------------------------------------------------------
# Registry — import this list in graph.py and pass to bind_tools() if needed
# ---------------------------------------------------------------------------

TOOLS: list[BaseTool] = [document_search, get_document_metadata, calculate, web_search]
