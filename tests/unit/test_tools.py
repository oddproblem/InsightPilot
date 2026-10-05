"""Unit tests for the four agentic tools.

Tests the calculator's AST sandbox extensively — this is safety-critical.
document_search and get_document_metadata are tested against a mock DB.
web_search is tested only for the no-key branch (no real network calls).
"""

from unittest.mock import MagicMock, patch

from app.graph.tools import calculate, document_search, get_document_metadata, web_search

# ---------------------------------------------------------------------------
# calculate
# ---------------------------------------------------------------------------


class TestCalculate:
    def test_basic_arithmetic(self) -> None:
        assert calculate.invoke({"expression": "2 + 3"}) == "Result: 5.0"

    def test_float_result(self) -> None:
        assert calculate.invoke({"expression": "10 / 3"}) == "Result: 3.333333"

    def test_percentage_change(self) -> None:
        result = calculate.invoke({"expression": "(120 - 100) / 100 * 100"})
        assert result == "Result: 20.0"

    def test_cagr_formula(self) -> None:
        # CAGR: ((150/100)**(1/3)-1)*100 ≈ 14.471
        result = calculate.invoke({"expression": "((150/100)**(1/3)-1)*100"})
        assert "14.47" in result

    def test_power_operator(self) -> None:
        result = calculate.invoke({"expression": "2**10"})
        assert result == "Result: 1024.0"

    def test_division_by_zero(self) -> None:
        result = calculate.invoke({"expression": "10 / 0"})
        assert "error" in result.lower() or "Error" in result

    def test_syntax_error(self) -> None:
        result = calculate.invoke({"expression": "2 +"})
        assert "Syntax error" in result or "error" in result.lower()

    def test_rejects_import(self) -> None:
        result = calculate.invoke({"expression": "__import__('os').system('dir')"})
        assert "Unsafe" in result or "rejected" in result.lower()

    def test_rejects_exec(self) -> None:
        result = calculate.invoke({"expression": "exec('print(1)')"})
        assert "Unsafe" in result or "rejected" in result.lower()

    def test_rejects_dunder(self) -> None:
        result = calculate.invoke({"expression": "__builtins__['exec']('x=1')"})
        assert "Unsafe" in result or "rejected" in result.lower()

    def test_rejects_long_expression(self) -> None:
        result = calculate.invoke({"expression": "1+" * 600})
        assert "too long" in result.lower()

    def test_unary_minus(self) -> None:
        result = calculate.invoke({"expression": "-5 * 2"})
        assert result == "Result: -10.0"

    def test_modulo(self) -> None:
        result = calculate.invoke({"expression": "10 % 3"})
        assert result == "Result: 1.0"

    def test_floor_division(self) -> None:
        result = calculate.invoke({"expression": "7 // 2"})
        assert result == "Result: 3.0"

    def test_string_in_expression_rejected(self) -> None:
        # ast.Constant with a string should raise
        result = calculate.invoke({"expression": "'hello' + 'world'"})
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# web_search — no-key branch
# ---------------------------------------------------------------------------


def test_web_search_no_key() -> None:
    with patch("app.graph.tools.config") as mock_cfg:
        mock_cfg.tavily_api_key = None
        result = web_search.invoke({"query": "latest S&P 500 news"})
    assert "not configured" in result.lower() or "TAVILY_API_KEY" in result


# ---------------------------------------------------------------------------
# document_search — mock DB
# ---------------------------------------------------------------------------


def test_document_search_no_documents(mock_openai_embeddings: MagicMock) -> None:
    """When documents table is empty, return informative message."""
    with (
        patch("app.graph.tools.OpenAI", return_value=mock_openai_embeddings),
        patch("app.graph.tools.db_conn") as mock_db,
    ):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        result = document_search.invoke({"query": "revenue FY2025"})

    assert "No relevant documents found" in result or "Upload" in result


def test_document_search_returns_results(mock_openai_embeddings: MagicMock) -> None:
    """When rows are returned, format them with source and score."""
    fake_row = (
        "doc-uuid-1",  # id
        "Annual Report 2025",  # document_name
        "Revenue was $10M in FY2025.",  # content
        "annual_report_2025.pdf",  # source
        3,  # page_number
        0,  # chunk_index
        "10-K",  # doc_type
        "Acme Corp",  # company
        "FY2025",  # financial_year
        0.12,  # distance (cosine)
    )
    with (
        patch("app.graph.tools.OpenAI", return_value=mock_openai_embeddings),
        patch("app.graph.tools.db_conn") as mock_db,
    ):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [fake_row]
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        result = document_search.invoke({"query": "revenue FY2025"})

    assert "Annual Report 2025" in result
    assert "Revenue was $10M" in result
    assert "score=" in result


# ---------------------------------------------------------------------------
# get_document_metadata
# ---------------------------------------------------------------------------


def test_get_document_metadata_invalid_uuid() -> None:
    result = get_document_metadata.invoke({"document_id": "not-a-uuid"})
    assert "Invalid document_id" in result


def test_get_document_metadata_not_found() -> None:
    with patch("app.graph.tools.db_conn") as mock_db:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        result = get_document_metadata.invoke(
            {"document_id": "00000000-0000-0000-0000-000000000000"}
        )
    assert "No document found" in result


def test_get_document_metadata_returns_info() -> None:
    import datetime

    fake_row = (
        "00000000-0000-0000-0000-000000000001",
        "Q4 Earnings",
        "q4_earnings.pdf",
        "earnings",
        "Acme Corp",
        "FY2025",
        datetime.datetime(2025, 1, 15, tzinfo=datetime.UTC),
        42,
        10,
    )
    with patch("app.graph.tools.db_conn") as mock_db:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = fake_row
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        result = get_document_metadata.invoke(
            {"document_id": "00000000-0000-0000-0000-000000000001"}
        )

    assert "Q4 Earnings" in result
    assert "Acme Corp" in result
    assert "FY2025" in result
    assert "42" in result  # chunk count
