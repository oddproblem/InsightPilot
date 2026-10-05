"""Unit tests for query routing classification."""

from app.graph.routing import classify_query_route


def test_classify_calculator_queries() -> None:
    assert classify_query_route("Calculate the CAGR from 2022 to 2025") == "calculator"
    assert classify_query_route("Compute the percentage change in revenue") == "calculator"
    assert classify_query_route("Calculate the margin variance between Q1 and Q2") == "calculator"


def test_classify_document_search_queries() -> None:
    assert classify_query_route("What was revenue in FY2025?") == "document_search"
    assert classify_query_route("Why did operating margin decline?") == "document_search"
    assert classify_query_route("Find evidence supporting this statement") == "document_search"
    assert classify_query_route("Check the 10-K report for risks") == "document_search"


def test_classify_web_search_queries() -> None:
    assert classify_query_route("What is today's market sentiment?") == "web_search"
    assert classify_query_route("Show me the latest news on competitors") == "web_search"


def test_classify_direct_queries() -> None:
    assert classify_query_route("Hello, how are you?") == "direct"
    assert classify_query_route("") == "direct"
