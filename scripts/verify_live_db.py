"""Verify live database schema, tables, pgvector extension, and indexes."""

from app.db.connection import close_pool, db_conn, init_pool


def main() -> None:
    init_pool()
    with db_conn() as conn:
        with conn.cursor() as cur:
            # 1. Check extensions
            cur.execute(
                "SELECT extname, extversion FROM pg_extension "
                "WHERE extname IN ('vector', 'uuid-ossp');"
            )
            print("=== Extensions ===")
            for row in cur.fetchall():
                print(f"  {row[0]}: version {row[1]}")

            # 2. Check public tables
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name;"
            )
            print("\n=== Tables ===")
            tables = [r[0] for r in cur.fetchall()]
            for t in tables:
                print(f"  - {t}")

            # 3. Check agent_messages columns (verify tenant_id is NOT NULL and has NO default)
            cur.execute(
                "SELECT column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_name = 'agent_messages' ORDER BY ordinal_position;"
            )
            print("\n=== agent_messages Columns ===")
            for col in cur.fetchall():
                print(f"  {col[0]}: type={col[1]}, nullable={col[2]}, default={col[3]}")

            # 4. Check documents table indexes
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename = 'documents' ORDER BY indexname;"
            )
            print("\n=== documents Indexes ===")
            for idx in cur.fetchall():
                print(f"  {idx[0]} -> {idx[1]}")

            # 5. Check agent_messages indexes
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename = 'agent_messages' ORDER BY indexname;"
            )
            print("\n=== agent_messages Indexes ===")
            for idx in cur.fetchall():
                print(f"  {idx[0]} -> {idx[1]}")

    close_pool()


if __name__ == "__main__":
    main()
