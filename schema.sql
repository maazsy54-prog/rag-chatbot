-- RAG chatbot schema (Step 1). Runs automatically the first time the DB container starts.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id         BIGSERIAL PRIMARY KEY,
    email      TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename   TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | ready | failed
    page_count INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, sha256)                      -- same file uploaded twice is processed once
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    page_start  INT,
    page_end    INT,
    heading     TEXT,
    content     TEXT NOT NULL,
    token_count INT,
    embedding   vector(1024),                     -- voyage-4 at 1024 dimensions
    -- 'simple' config = no language-specific stemming, so it works for any language
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_user_doc_idx  ON chunks (user_id, document_id);

-- One development user until real authentication is added
INSERT INTO users (email) VALUES ('dev@local') ON CONFLICT DO NOTHING;
