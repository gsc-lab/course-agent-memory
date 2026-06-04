-- 메모리 저장소 스키마 (idempotent — 앱이 시작할 때마다 안전하게 재적용된다)
-- gen3: memories(원문 청크 + 임베딩)만 사용. gen4·gen5에서 컬럼/테이블을 확장한다.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id         BIGSERIAL    PRIMARY KEY,
    text       TEXT         NOT NULL,
    embedding  vector(1536) NOT NULL,        -- text-embedding-3-small 차원
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 코사인 거리 ANN(HNSW) 인덱스. 데이터가 적으면 없어도 동작하지만 실무 패턴을 보여준다.
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);
