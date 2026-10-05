"""Live End-to-End Database & pgvector Roundtrip Verification.

Executes real PostgreSQL queries against the live Supabase database:
1. Verifies extensions ('vector', 'uuid-ossp').
2. Inserts document chunks with 1536-dim vectors into 'documents'.
3. Executes live cosine distance similarity retrieval (<=>) with pgvector.
4. Verifies vector retrieval isolation: Tenant B cannot retrieve Tenant A's chunks.
5. Verifies conversation history isolation: Identical session_id across Tenant A
   and Tenant B returns strictly isolated messages.
6. Cleans up test artifacts.
"""

import math
import random
import uuid
from typing import Any

from app.db.connection import close_pool, db_conn, init_pool
from app.db.queries import (
    CREATE_SESSION,
    DELETE_DOCUMENT,
    GET_SESSION_MESSAGES,
    INCREMENT_SESSION_MESSAGE_COUNT,
    INSERT_DOCUMENT_CHUNK,
    INSERT_MESSAGE,
)


def _generate_normalized_vector(dim: int = 1536, seed: int = 42) -> list[float]:
    """Generate a deterministic unit-length vector for pgvector cosine distance testing."""
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [round(x / norm, 6) for x in raw]


def main() -> None:
    print("==================================================")
    print("LIVE POSTGRESQL + PGVECTOR VERIFICATION")
    print("==================================================")

    init_pool()

    tenant_a = f"test-tenant-a-{uuid.uuid4().hex[:6]}"
    tenant_b = f"test-tenant-b-{uuid.uuid4().hex[:6]}"
    doc_id_a = str(uuid.uuid4())
    session_id = f"financial-audit-{uuid.uuid4().hex[:6]}"

    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                # ─── 1. Extension Verification ──────────────────────────
                cur.execute(
                    "SELECT extname, extversion FROM pg_extension "
                    "WHERE extname IN ('vector', 'uuid-ossp');"
                )
                exts = dict(cur.fetchall())
                print(f"[OK] Live Extensions: {exts}")
                assert "vector" in exts, "pgvector extension is missing!"
                assert "uuid-ossp" in exts, "uuid-ossp extension is missing!"

                # ─── 2. Document Chunks & 1536-dim Vector Insertion ────
                # Target vector for query
                query_vec = _generate_normalized_vector(1536, seed=100)
                # Chunk 1: Very close to query_vec
                vec_chunk_1 = [round(x + 0.001, 6) for x in query_vec]
                norm_1 = math.sqrt(sum(x * x for x in vec_chunk_1))
                vec_chunk_1 = [round(x / norm_1, 6) for x in vec_chunk_1]

                # Chunk 2: Orthogonal/distant vector
                vec_chunk_2 = _generate_normalized_vector(1536, seed=999)

                # Insert Chunk 1 (Financial Revenue excerpt)
                chunk_1_id = str(uuid.uuid4())
                cur.execute(
                    INSERT_DOCUMENT_CHUNK,
                    {
                        "id": chunk_1_id,
                        "tenant_id": tenant_a,
                        "document_id": doc_id_a,
                        "document_name": "acme_q3_financial_report.pdf",
                        "source": "acme_q3_financial_report.pdf",
                        "content": (
                            "Acme Corp reported Q3 2026 total revenue of $42.5 million, "
                            "up 14.8% YoY."
                        ),
                        "page_number": 3,
                        "chunk_index": 0,
                        "doc_type": "10-Q",
                        "company": "Acme Corp",
                        "financial_year": "FY2026",
                        "metadata": (
                            '{"section": "Consolidated Financial Statements", "table": true}'
                        ),
                        "embedding": str(vec_chunk_1),
                    },
                )

                # Insert Chunk 2 (Risk factors excerpt)
                chunk_2_id = str(uuid.uuid4())
                cur.execute(
                    INSERT_DOCUMENT_CHUNK,
                    {
                        "id": chunk_2_id,
                        "tenant_id": tenant_a,
                        "document_id": doc_id_a,
                        "document_name": "acme_q3_financial_report.pdf",
                        "source": "acme_q3_financial_report.pdf",
                        "content": (
                            "Risks include supply chain disruptions in semiconductor procurement."
                        ),
                        "page_number": 12,
                        "chunk_index": 1,
                        "doc_type": "10-Q",
                        "company": "Acme Corp",
                        "financial_year": "FY2026",
                        "metadata": '{"section": "Risk Factors", "table": false}',
                        "embedding": str(vec_chunk_2),
                    },
                )
                print(
                    f"[OK] Inserted 2 vector chunks for Document ID: {doc_id_a} "
                    f"(Tenant: {tenant_a})"
                )

                # ─── 3. Live Cosine Similarity Search (<=>) ───────────
                search_sql = """
                    SELECT document_name, content, page_number,
                           (embedding <=> %(embedding)s::vector) AS distance
                    FROM documents
                    WHERE tenant_id = %(tenant_id)s
                    ORDER BY distance ASC
                    LIMIT 2;
                """
                cur.execute(search_sql, {"embedding": str(query_vec), "tenant_id": tenant_a})
                results: list[Any] = cur.fetchall()
                print(f"[OK] Live pgvector Cosine Search retrieved {len(results)} chunks:")
                for r in results:
                    print(f"     * Distance: {r[3]:.6f} | Page {r[2]} | Content: {r[1]}")

                # Verify nearest neighbor ordering
                assert results[0][2] == 3, "Expected Chunk 1 (revenue) to be nearest match!"
                assert results[0][3] < results[1][3], "Expected cosine distance to increase!"

                # ─── 4. Cross-Tenant Vector Isolation Proof ────────────
                cur.execute(search_sql, {"embedding": str(query_vec), "tenant_id": tenant_b})
                tenant_b_results = cur.fetchall()
                print(
                    f"[OK] Tenant B search for Tenant A's documents returned "
                    f"{len(tenant_b_results)} chunks (Strictly Isolated)."
                )
                assert len(tenant_b_results) == 0, "Cross-tenant vector data leak detected!"

                # ─── 5. Conversation History & Tenant Isolation Proof ──
                # Create session for Tenant A
                cur.execute(
                    CREATE_SESSION,
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "tenant_id": tenant_a,
                    },
                )
                # Create session for Tenant B on the IDENTICAL session_id
                cur.execute(
                    CREATE_SESSION,
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "tenant_id": tenant_b,
                    },
                )

                # Tenant A conversation turn
                cur.execute(
                    INSERT_MESSAGE,
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "tenant_id": tenant_a,
                        "role": "user",
                        "content": "What was Acme Q3 revenue?",
                        "metadata": "{}",
                    },
                )
                cur.execute(
                    INSERT_MESSAGE,
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "tenant_id": tenant_a,
                        "role": "assistant",
                        "content": "Acme Q3 revenue was $42.5 million.",
                        "metadata": '{"confidence": "high"}',
                    },
                )
                cur.execute(
                    INCREMENT_SESSION_MESSAGE_COUNT,
                    {"session_id": session_id, "tenant_id": tenant_a},
                )
                cur.execute(
                    INCREMENT_SESSION_MESSAGE_COUNT,
                    {"session_id": session_id, "tenant_id": tenant_a},
                )

                # Tenant B conversation turn on identical session_id
                cur.execute(
                    INSERT_MESSAGE,
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "tenant_id": tenant_b,
                        "role": "user",
                        "content": "What is Tenant B's confidential roadmap?",
                        "metadata": "{}",
                    },
                )
                cur.execute(
                    INSERT_MESSAGE,
                    {
                        "id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "tenant_id": tenant_b,
                        "role": "assistant",
                        "content": "Tenant B confidential roadmap details.",
                        "metadata": '{"confidence": "high"}',
                    },
                )
                cur.execute(
                    INCREMENT_SESSION_MESSAGE_COUNT,
                    {"session_id": session_id, "tenant_id": tenant_b},
                )
                cur.execute(
                    INCREMENT_SESSION_MESSAGE_COUNT,
                    {"session_id": session_id, "tenant_id": tenant_b},
                )

                # Retrieve messages for Tenant A
                cur.execute(GET_SESSION_MESSAGES, {"session_id": session_id, "tenant_id": tenant_a})
                msgs_a = cur.fetchall()
                print(f"[OK] Tenant A session retrieved {len(msgs_a)} messages:")
                for m in msgs_a:
                    print(f"     * Role: {m[3]} | Content: {m[4]}")
                assert len(msgs_a) == 2
                assert "Acme" in msgs_a[0][4]
                assert "Tenant B" not in msgs_a[0][4]

                # Retrieve messages for Tenant B
                cur.execute(GET_SESSION_MESSAGES, {"session_id": session_id, "tenant_id": tenant_b})
                msgs_b = cur.fetchall()
                print(f"[OK] Tenant B session retrieved {len(msgs_b)} messages:")
                for m in msgs_b:
                    print(f"     * Role: {m[3]} | Content: {m[4]}")
                assert len(msgs_b) == 2
                assert "Tenant B" in msgs_b[0][4]
                assert "Acme" not in msgs_b[0][4]

                # Clean up test document
                cur.execute(DELETE_DOCUMENT, {"document_id": doc_id_a, "tenant_id": tenant_a})
                print(f"[OK] Cleaned up test document {doc_id_a}")

        print("==================================================")
        print("ALL LIVE DATABASE & PGVECTOR CHECKS PASSED 100%!")
        print("==================================================")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
