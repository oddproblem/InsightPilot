"""InsightPilot agent graph nodes.

Pipeline (in execution order):
  query_analyzer  → classify the user query and choose a route
  tool_dispatch   → run the appropriate tool(s) based on the route
  answer_generator→ synthesise a grounded answer from tool results
  citation_validator → verify each claim is backed by a retrieved source
  safety_check    → apply input/output guards; set insufficient_evidence flag

Each node receives AgentState and returns a dict of state updates.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import config
from app.graph.prompts import (
    ANSWER_GENERATION_PROMPT,
    QUERY_ANALYSIS_PROMPT,
)
from app.graph.routing import RouteType, classify_query_route
from app.graph.state import AgentState, Citation, EvidenceItem, RetrievedChunk, ToolResult
from app.graph.tools import (
    TOOLS,
    _tenant_ctx,
    calculate,
    document_search,
    web_search,
)
from app.rag.citations import verify_citation_grounding
from app.rag.retriever import retrieve_relevant_chunks
from app.security.input_guard import check_input_safety
from app.security.output_guard import sanitize_output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm() -> Any:
    kwargs: dict[str, Any] = {
        "model": config.llm_model,
        "api_key": config.llm_api_key,
    }
    if config.llm_base_url:
        kwargs["base_url"] = config.llm_base_url
    if config.llm_extra_headers:
        kwargs["default_headers"] = config.llm_extra_headers
    return ChatOpenAI(**kwargs)


def _llm_with_tools() -> Any:
    return _llm().bind_tools(TOOLS)


def _latest_user_query(state: AgentState) -> str:
    """Extract the most recent HumanMessage content from state."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return state.get("query", "")


# ---------------------------------------------------------------------------
# Node 1: query_analyzer
# ---------------------------------------------------------------------------


def query_analyzer(state: AgentState) -> dict[str, Any]:
    """Classify the user query and select an execution route.

    Tries LLM-based classification first; falls back to deterministic
    pattern matching if the LLM response is malformed.

    Updates: query, route, safety_flags
    """
    query = _latest_user_query(state)

    # ── Input safety guard ──────────────────────────────────────────────────
    safety_flags = list(state.get("safety_flags", []))
    input_flags = check_input_safety(query)
    safety_flags.extend(input_flags)

    if input_flags:
        logger.warning(
            "input_safety_flags",
            extra={"session_id": state.get("session_id"), "flags": input_flags},
        )
        # Short-circuit: route to direct so the LLM produces a refusal
        return {
            "query": query,
            "route": "direct",
            "safety_flags": safety_flags,
        }

    # ── LLM classification ──────────────────────────────────────────────────
    route: RouteType = "direct"
    try:
        llm = _llm()
        sys_msg = SystemMessage(content=QUERY_ANALYSIS_PROMPT)
        user_msg = HumanMessage(content=query)
        response = llm.invoke([sys_msg, user_msg])
        raw = str(response.content).strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        candidate = parsed.get("route", "direct")
        if candidate in ("document_search", "web_search", "calculator", "direct"):
            route = candidate
        logger.info(
            "query_routed",
            extra={
                "session_id": state.get("session_id"),
                "route": route,
                "reasoning": parsed.get("reasoning", ""),
            },
        )
    except Exception as exc:
        logger.warning(
            "query_analyzer_llm_fallback",
            extra={"error": str(exc), "query": query[:120]},
        )
        route = classify_query_route(query)

    return {"query": query, "route": route, "safety_flags": safety_flags}


# ---------------------------------------------------------------------------
# Node 2: tool_dispatch
# ---------------------------------------------------------------------------


def tool_dispatch(state: AgentState) -> dict[str, Any]:
    """Execute the tool selected by the query_analyzer.

    Calls the tool synchronously and populates:
      - retrieved_documents (for document_search)
      - web_results (for web_search)
      - tool_results (for calculate / get_document_metadata)

    Does NOT call the LLM — that is the answer_generator's job.
    """
    route = state.get("route", "direct")
    query = state.get("query", "")
    tenant_id = state.get("tenant_id", "default")
    _tenant_ctx.set(tenant_id)

    retrieved_documents: list[RetrievedChunk] = list(state.get("retrieved_documents", []))
    web_results: list[dict[str, Any]] = list(state.get("web_results", []))
    tool_results: list[ToolResult] = list(state.get("tool_results", []))

    if route == "document_search":
        raw = document_search.invoke({"query": query})
        tool_results.append(
            ToolResult(
                tool_name="document_search",
                tool_input={"query": query, "tenant_id": tenant_id},
                output=raw,
                status="ok" if "failed" not in raw.lower() else "error",
            )
        )
        # Parse structured chunks from retriever for evidence layer
        chunks = retrieve_relevant_chunks(query=query, tenant_id=tenant_id, top_k=5)
        for c in chunks:
            retrieved_documents.append(
                RetrievedChunk(
                    id=c["id"],
                    content=c["content"],
                    source=c["source"],
                    page=c.get("page_number") or c.get("metadata", {}).get("page_number"),
                    score=c.get("score", 0.0),
                    metadata=c.get("metadata", {}),
                )
            )

    elif route == "web_search":
        raw = web_search.invoke({"query": query})
        web_results.append({"query": query, "results": raw})
        tool_results.append(
            ToolResult(
                tool_name="web_search",
                tool_input={"query": query},
                output=raw,
                status="ok" if "failed" not in raw.lower() else "error",
            )
        )

    elif route == "calculator":
        # Extract math expression from the query (use LLM to isolate it)
        expression = _extract_expression(query)
        raw = calculate.invoke({"expression": expression})
        status = "ok" if "error" not in raw.lower() and "unsafe" not in raw.lower() else "error"
        tool_results.append(
            ToolResult(
                tool_name="calculate",
                tool_input={"expression": expression},
                output=raw,
                status=status,
            )
        )

    # route == "direct" → no tool call; answer_generator handles it with messages alone

    logger.info(
        "tool_dispatched",
        extra={
            "session_id": state.get("session_id"),
            "route": route,
            "tool_results_count": len(tool_results),
        },
    )

    return {
        "retrieved_documents": retrieved_documents,
        "web_results": web_results,
        "tool_results": tool_results,
    }


def _extract_expression(query: str) -> str:
    """Use the LLM to extract a calculable expression from a natural-language query.

    Falls back to returning the raw query if extraction fails.
    """
    extraction_prompt = (
        "Extract ONLY the mathematical expression from the following user query. "
        "Return ONLY the expression — no explanation, no units, no extra text.\n\n"
        f"Query: {query}"
    )
    try:
        llm = _llm()
        resp = llm.invoke([HumanMessage(content=extraction_prompt)])
        expr = str(resp.content).strip()
        # Basic sanity: must contain a digit
        if any(ch.isdigit() for ch in expr):
            return expr
    except Exception as exc:
        logger.warning("expression_extraction_failed", extra={"error": str(exc)})
    return query


# ---------------------------------------------------------------------------
# Node 3: answer_generator
# ---------------------------------------------------------------------------


def answer_generator(state: AgentState) -> dict[str, Any]:
    """Synthesise a grounded, evidence-backed answer using the LLM.

    Builds a context prompt from all tool results + retrieved documents,
    then calls the LLM to produce the final answer.

    Updates: messages, answer, evidence, input_tokens, output_tokens
    """
    query = state.get("query", "")
    route = state.get("route", "direct")
    tool_results = state.get("tool_results", [])
    retrieved_documents = state.get("retrieved_documents", [])
    web_results = state.get("web_results", [])
    safety_flags = state.get("safety_flags", [])

    # Build system context
    context_parts: list[str] = []

    if tool_results:
        for tr in tool_results:
            context_parts.append(
                f"[Tool: {tr.get('tool_name', 'unknown')}]\n{tr.get('output', '')}"
            )

    if retrieved_documents:
        doc_texts = []
        for chunk in retrieved_documents:
            src = chunk.get("source", "Unknown")
            page = chunk.get("page")
            page_str = f" (p.{page})" if page else ""
            doc_texts.append(f"[Source: {src}{page_str}]\n{chunk.get('content', '')}")
        context_parts.append("Retrieved Document Chunks:\n" + "\n\n".join(doc_texts))

    if web_results:
        for wr in web_results:
            q = wr.get("query", "")
            r = wr.get("results", "")
            context_parts.append(f"[Web Results for: {q}]\n{r}")

    # Safety refusal if injection detected
    if safety_flags:
        refusal = (
            "I cannot process this request as it contains content that violates safety policies."
        )
        return {
            "messages": [AIMessage(content=refusal)],
            "answer": refusal,
            "evidence": [],
            "insufficient_evidence": False,
        }

    # Assemble messages for the LLM
    messages: list[BaseMessage] = [SystemMessage(content=ANSWER_GENERATION_PROMPT)]
    if context_parts:
        messages.append(
            SystemMessage(content="RETRIEVED CONTEXT:\n\n" + "\n\n---\n\n".join(context_parts))
        )

    # Include prior conversation history (without re-adding current user message)
    history = list(state.get("messages", []))
    messages.extend(history)

    # Append current query as the final human turn if not already last
    if not history or not isinstance(history[-1], HumanMessage):
        messages.append(HumanMessage(content=query))

    llm = _llm()
    response = llm.invoke(messages)
    answer = str(response.content)

    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))

    # Extract evidence items from retrieved documents
    evidence: list[EvidenceItem] = []
    for chunk in retrieved_documents:
        evidence.append(
            EvidenceItem(
                statement=chunk.get("content", "")[:300],
                source=chunk.get("source", "Unknown"),
                verified=False,  # citation_validator will flip this
            )
        )

    logger.info(
        "answer_generated",
        extra={
            "session_id": state.get("session_id"),
            "route": route,
            "answer_length": len(answer),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )

    return {
        "messages": [response],
        "answer": answer,
        "evidence": evidence,
        "input_tokens": state.get("input_tokens", 0) + input_tokens,
        "output_tokens": state.get("output_tokens", 0) + output_tokens,
    }


# ---------------------------------------------------------------------------
# Node 4: citation_validator
# ---------------------------------------------------------------------------


def citation_validator(state: AgentState) -> dict[str, Any]:
    """Verify that each claim in the answer is grounded in a retrieved source.

    Uses verify_citation_grounding (string-match heuristic) to produce
    structured Citation objects, then calls the LLM to flag hallucinations.

    Updates: citations, confidence, insufficient_evidence
    """
    answer = state.get("answer", "")
    retrieved_documents = state.get("retrieved_documents", [])
    tool_results = state.get("tool_results", [])

    # Convert RetrievedChunks to the dict format citations.py expects
    chunk_dicts: list[dict[str, Any]] = [
        {
            "id": c.get("id", ""),
            "content": c.get("content", ""),
            "source": c.get("source", "Unknown"),
            "metadata": {**c.get("metadata", {}), "page_number": c.get("page")},
        }
        for c in retrieved_documents
    ]

    raw_citations = verify_citation_grounding(answer, chunk_dicts)
    citations: list[Citation] = [
        Citation(
            source=rc["source"],
            snippet=rc["snippet"],
            verified=rc["verified"],
            page=rc.get("page"),
        )
        for rc in raw_citations
    ]

    # Determine confidence based on verification coverage
    verified_count = sum(1 for c in citations if c.get("verified"))
    total = len(citations)

    if not retrieved_documents and not tool_results:
        # Pure direct answer — no retrieval was done
        confidence = "medium"
        insufficient = False
    elif total == 0:
        confidence = "low"
        insufficient = True
    elif verified_count == 0:
        confidence = "low"
        insufficient = True
    elif verified_count / total >= 0.7:
        confidence = "high"
        insufficient = False
    else:
        confidence = "medium"
        insufficient = False

    # If no evidence at all and the answer says "not sufficient", mark it
    not_enough_phrases = [
        "do not contain sufficient",
        "not enough information",
        "cannot find",
        "no information available",
        "documents do not",
    ]
    if any(phrase in answer.lower() for phrase in not_enough_phrases):
        insufficient = True
        confidence = "low"

    logger.info(
        "citations_validated",
        extra={
            "session_id": state.get("session_id"),
            "total_citations": total,
            "verified_citations": verified_count,
            "confidence": confidence,
        },
    )

    return {
        "citations": citations,
        "confidence": confidence,
        "insufficient_evidence": insufficient,
    }


# ---------------------------------------------------------------------------
# Node 5: safety_check
# ---------------------------------------------------------------------------


def safety_check(state: AgentState) -> dict[str, Any]:
    """Apply output safety guard: redact PII, flag policy violations.

    This is the final gate before the answer leaves the system.

    Updates: answer, messages (last AIMessage), safety_flags
    """
    answer = state.get("answer", "")
    safety_flags = list(state.get("safety_flags", []))

    cleaned_answer, output_flags = sanitize_output(answer)
    safety_flags.extend(output_flags)

    if output_flags:
        logger.warning(
            "output_safety_flags",
            extra={"session_id": state.get("session_id"), "flags": output_flags},
        )

    # Replace last AIMessage in messages with the sanitized answer
    messages = list(state.get("messages", []))
    if messages and isinstance(messages[-1], AIMessage):
        messages[-1] = AIMessage(content=cleaned_answer)
    elif cleaned_answer:
        messages.append(AIMessage(content=cleaned_answer))

    return {
        "answer": cleaned_answer,
        "messages": messages,
        "safety_flags": safety_flags,
    }


# ---------------------------------------------------------------------------
# Routing predicate (used by graph.py for conditional edges)
# ---------------------------------------------------------------------------


def route_after_analyzer(state: AgentState) -> str:
    """Return next node name based on the route chosen by query_analyzer."""
    route = state.get("route", "direct")
    if route in ("document_search", "web_search", "calculator"):
        return "tool_dispatch"
    return "answer_generator"
