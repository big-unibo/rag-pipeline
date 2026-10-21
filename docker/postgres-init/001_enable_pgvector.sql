CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    file_id text PRIMARY KEY,
    title text NOT NULL,
    full_text text
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    file_id text NOT NULL REFERENCES documents(file_id) ON DELETE CASCADE,
    chunk_id integer NOT NULL,
    chunking_model text NOT NULL DEFAULT 'structured',
    text text NOT NULL,
    pgvector_embedding vector(384),
    PRIMARY KEY (file_id, chunking_model, chunk_id)
);

CREATE TABLE IF NOT EXISTS question (
    q_id integer PRIMARY KEY,
    question_text text NOT NULL,
    ground_truth text
);

ALTER TABLE chunk_embeddings
ADD COLUMN IF NOT EXISTS pgvector_embedding vector(384);

ALTER TABLE chunk_embeddings
DROP COLUMN IF EXISTS layout_aware_text,
DROP COLUMN IF EXISTS pgvector_all_mpnet_base_v2,
DROP COLUMN IF EXISTS pgvector_all_mpnet_base_layout_aware;

ALTER TABLE question
DROP COLUMN IF EXISTS pgvector_all_mpnet_base_v2;

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_embedding_vector
ON chunk_embeddings
USING ivfflat (pgvector_embedding vector_cosine_ops)
WITH (lists = 100);
