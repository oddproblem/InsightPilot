"""InsightPilot compiled LangGraph agent graph.

Pipeline topology:
  query_analyzer
      │
      ├─(route == document_search | web_search | calculator)─► tool_dispatch
      │                                                              │
      └─(route == direct)───────────────────────────────────────────┘
                                                                     │
                                                              answer_generator
                                                                     │
                                                           citation_validator
                                                                     │
                                                              safety_check
                                                                     │
                                                                    END
"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    answer_generator,
    citation_validator,
    query_analyzer,
    route_after_analyzer,
    safety_check,
    tool_dispatch,
)
from app.graph.state import AgentState


def build_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build and compile the InsightPilot agent graph."""
    builder: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("query_analyzer", query_analyzer)
    builder.add_node("tool_dispatch", tool_dispatch)
    builder.add_node("answer_generator", answer_generator)
    builder.add_node("citation_validator", citation_validator)
    builder.add_node("safety_check", safety_check)

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.set_entry_point("query_analyzer")

    # After classification: tool-using routes go to tool_dispatch; direct → answer_generator
    builder.add_conditional_edges(
        "query_analyzer",
        route_after_analyzer,
        {
            "tool_dispatch": "tool_dispatch",
            "answer_generator": "answer_generator",
        },
    )

    # Tool results always flow into answer_generator
    builder.add_edge("tool_dispatch", "answer_generator")

    # Linear synthesis pipeline
    builder.add_edge("answer_generator", "citation_validator")
    builder.add_edge("citation_validator", "safety_check")
    builder.add_edge("safety_check", END)

    return builder.compile()


# Module-level compiled graph — initialized once at startup via init_graph().
_graph: CompiledStateGraph[AgentState, None, AgentState, AgentState] | None = None

# Pre-compiled graph instance for testing and direct invocation
app_graph: CompiledStateGraph[AgentState, None, AgentState, AgentState] = build_graph()


def get_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Return the compiled graph. Raises if init_graph() has not been called."""
    if _graph is None:
        raise RuntimeError("Graph not initialized. Call init_graph() at startup.")
    return _graph


def init_graph() -> None:
    """Compile and store the graph. Called once during application startup."""
    global _graph
    _graph = build_graph()
