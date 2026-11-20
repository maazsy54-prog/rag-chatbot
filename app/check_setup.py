"""Step 1 smoke test: confirms Postgres + pgvector, Voyage and Claude all work.

Run from the project root:   python app/check_setup.py
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check(name, fn):
    try:
        print(f"[ OK ] {name}: {fn()}")
        return True
    except Exception as exc:  # show any failure clearly
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return False


def check_postgres():
    import psycopg

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        version = conn.execute("SELECT version()").fetchone()[0].split(",")[0]
        ext = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        tables = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name IN ('users', 'documents', 'chunks')"
        ).fetchone()[0]
    if ext is None:
        raise RuntimeError("pgvector extension is not installed")
    if tables != 3:
        raise RuntimeError(f"expected 3 tables, found {tables} (did schema.sql run?)")
    return f"{version}, pgvector {ext[0]}, 3/3 tables present"


def check_voyage():
    import voyageai

    result = voyageai.Client().embed(
        ["hello world"], model="voyage-4", input_type="document", output_dimension=1024
    )
    return f"voyage-4 returned a vector of length {len(result.embeddings[0])}"


def check_claude():
    import anthropic

    msg = anthropic.Anthropic().messages.create(
        model="claude-sonnet-5-5",
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with the single word: OK"}],
    )
    return f"claude-sonnet-5-5 replied {msg.content[0].text.strip()!r}"


if __name__ == "__main__":
    results = [
        check("Postgres + pgvector", check_postgres),
        check("Voyage embeddings", check_voyage),
        check("Claude", check_claude),
    ]
    sys.exit(0 if all(results) else 1)
