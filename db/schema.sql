-- 메모리 저장소 스키마
-- 여러 번 실행해도 안전하도록 IF NOT EXISTS를 사용한다.
-- gen3: memories 테이블에 원문 조각과 임베딩을 저장한다.
-- gen4 이후: facts 등 사실 단위 테이블을 함께 사용한다.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id         BIGSERIAL    PRIMARY KEY,
    source     TEXT         NOT NULL DEFAULT 'general',
    text       TEXT         NOT NULL,
    embedding  vector(1536) NOT NULL,        -- text-embedding-3-small의 벡터 차원
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 이미 만들어진 DB에도 source 컬럼을 추가한다.
-- 세대별 데모 데이터를 구분해 해당 세대 데이터만 지우기 위함이다.
ALTER TABLE memories ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'general';

-- 코사인 거리 검색용 HNSW 인덱스.
-- 작은 데모에서는 없어도 되지만, 실제 벡터 검색에서 자주 쓰는 방식이다.
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);

-- gen4: 원문 대신 추출된 사실을 저장한다.
-- updated_at은 "서울 → 부산"처럼 값이 바뀐 시점을 추적하는 데 쓴다.
CREATE TABLE IF NOT EXISTS facts (
    id         BIGSERIAL    PRIMARY KEY,
    fact       TEXT         NOT NULL,
    embedding  vector(1536) NOT NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS facts_embedding_idx
    ON facts USING hnsw (embedding vector_cosine_ops);
