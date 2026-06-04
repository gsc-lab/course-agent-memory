"""
gen3 — 3세대 메모리: 단기 + 장기(Vector DB), 고정 청킹 + 유사도 검색(RAG)

[2세대의 어떤 실패를 푸는가]
2세대 요약은 비가역 압축이라 디테일이 날아갔다('사료' → '식사 문제'). 3세대는
원본을 버리거나 뭉개지 않는다. 대화를 **영속 Vector DB(pgvector)** 에 저장해 두고,
질문이 오면 '의미가 가까운 원문'을 그대로 검색해 떠올린다(RAG). 손실 없는 회상.

[이 파일의 두 장면]
  1) 맨손 데모(12줄): 유사도 검색이 '마법'이 아니라 그냥 '거리순 정렬'임을 먼저 본다.
  2) 진짜 Vector DB : 같은 일을 pgvector로. 검색은 SQL 한 줄 —
       SELECT text FROM memories ORDER BY embedding <=> :q LIMIT k;
     (<=> 는 코사인 거리. 'ORDER BY 거리'가 곧 의미 검색이다.)

[그런데 새로 생기는 한계]
- 고정길이로 자른 청크는 턴·화자 경계를 무시해 엉뚱하게 섞인다(아래 청크 출력 참고).
- 저장하는 건 '원문 덩어리'일 뿐 '사실'이 아니다. 그래서 "서울→부산"처럼 바뀐 사실의
  최신값을 모르고(둘 다 그냥 저장됨), 중복·모순을 정리하지 못한다.
  → 4세대: 원문 대신 'LLM이 추출한 정제된 사실'을 저장하고 갱신/충돌을 관리한다.

[준비물]
  1) docker compose up -d   (pgvector 기동)
  2) .env 에 OPENAI_API_KEY  (임베딩 호출). DATABASE_URL은 기본값 사용 가능.
"""

import os
import pathlib
import sys

import numpy as np

# repo 루트를 sys.path에 추가(세대 예제 공통 부트스트랩).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import llm, scoring
from common.scenario import PROBES, conversation

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://agent:agent@localhost:5433/agent_memory"
)
SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
CHUNK_SIZE = 60  # 고정 청크 길이(글자). 일부러 턴 경계를 무시한다 — 한계 시연용.
TOP_K = 3


# ---------------------------------------------------------------------------
# 장면 1) 맨손 유사도 검색 — '마법이 아니라 거리순 정렬'
# (01의 Bag-of-Words 임베딩 + 코사인. pgvector가 내부에서 하는 일의 12줄 축약판)
# ---------------------------------------------------------------------------
def toy_demo() -> None:
    vocab = ["코코", "사료", "부산", "파이썬"]

    def emb(t):
        return np.array([float(t.count(w)) for w in vocab])

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))

    memos = ["코코가 사료를 안 먹는다", "부산으로 이사했다", "파이썬을 공부한다"]
    query = "코코 사료"
    ranked = sorted(memos, key=lambda m: cos(emb(query), emb(m)), reverse=True)

    print("[장면 1] 맨손 유사도 검색(12줄) — 검색 = 거리순 정렬, 마법이 아니다")
    print(f"  질문 '{query}' 와 가까운 순서:")
    for m in ranked:
        print(f"    {cos(emb(query), emb(m)):.3f}  {m}")
    print("  pgvector도 똑같다. 단지 '진짜 임베딩 + 더 빠른 정렬'일 뿐.\n")


# ---------------------------------------------------------------------------
# 장면 2) 진짜 Vector DB (pgvector)
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def connect():
    """pgvector에 연결하고 스키마를 idempotent하게 적용한다."""
    import psycopg
    from pgvector.psycopg import register_vector

    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:  # 연결 실패 = 보통 DB 미기동
        raise SystemExit(
            "pgvector에 연결할 수 없습니다. 먼저 `docker compose up -d` 로 DB를 켜주세요.\n"
            f"  DATABASE_URL={DATABASE_URL}\n  원인: {e}"
        )
    # 스키마(확장 → 테이블 → 인덱스)를 한 문장씩 적용한다. vector 타입이 만들어진
    # 뒤에야 register_vector가 동작하므로 순서가 중요하다(확장 먼저, 등록 나중).
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in (s.strip() for s in schema_sql.split(";")):
        if statement:
            conn.execute(statement)
    conn.commit()
    register_vector(conn)  # 이제 numpy 벡터 <-> vector 타입 자동 변환 등록
    return conn


def pgvector_demo() -> None:
    conn = connect()
    conn.execute("TRUNCATE memories RESTART IDENTITY")  # 데모를 매번 깨끗이
    conn.commit()

    # 대화 전체를 한 덩어리 텍스트로 만든 뒤 '고정길이'로 자른다(naive chunking).
    full = "\n".join(f"{role}: {text}" for role, text in conversation())
    chunks = chunk_text(full, CHUNK_SIZE)

    print("[장면 2] pgvector RAG — 고정길이 청킹 + 의미 검색")
    print(f"  대화를 {CHUNK_SIZE}자 고정 청크 {len(chunks)}개로 분할(턴 경계 무시):")
    for i, c in enumerate(chunks):
        print(f"    청크{i}: {c!r}")
    print()

    # 청크를 임베딩해 저장한다.
    embeddings = llm.embed(chunks)
    for chunk, emb in zip(chunks, embeddings):
        conn.execute(
            "INSERT INTO memories (text, embedding) VALUES (%s, %s)", (chunk, emb)
        )
    conn.commit()

    # 각 probe를 의미 검색으로 답한다: SELECT ... ORDER BY embedding <=> q LIMIT k
    answers = []
    for probe in PROBES:
        q_emb = llm.embed([probe["question"]])[0]
        rows = conn.execute(
            "SELECT text FROM memories ORDER BY embedding <=> %s LIMIT %s",
            (q_emb, TOP_K),
        ).fetchall()
        retrieved = " | ".join(r[0].replace("\n", " ") for r in rows)
        hit = next((kw for kw in probe["expected"] if kw in retrieved), None)
        snippet = rows[0][0].replace("\n", " ")[:32] if rows else ""
        answers.append(
            f"검색된 원문(\"{snippet}…\")에서 '{hit}' 확인" if hit
            else "검색 결과에 관련 내용이 없어요."
        )

    results = scoring.grade(PROBES, answers)
    scoring.print_report("pgvector 검색으로 답하기", results)
    print("  ↳ 작은 대화에선 2세대 요약도 충분하다. 3세대의 진짜 이점은 '규모'다 —")
    print("     수백 턴이면 요약은 디테일을 잃지만, RAG는 원본을 그대로 검색한다.")
    print("     (아래 Q2 실패는 그와 별개인 3세대 고유의 한계다.)\n")
    conn.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 3세대: 장기 메모리(pgvector) + 고정 청킹 RAG ===\n")
    toy_demo()
    pgvector_demo()

    print("[한계] 원본을 손실 없이 보관·검색하지만, 두 가지가 남는다")
    print(" 1) 고정 청킹이 턴·화자를 끊어, 검색 결과에 엉뚱한 조각이 섞인다(위 청크 참고).")
    print(" 2) 저장한 건 '원문'일 뿐 '사실'이 아니다. 그래서 Q2가 실패했다 —")
    print("    '어디 살아?'에 옛 주소(서울) 청크가 더 비슷하게 잡혀 최신값(부산)을 놓쳤다.")
    print("    중복·모순·갱신을 정리할 '사실 단위' 상태가 없기 때문이다.")
    print("  → 4세대: LLM이 원문에서 '정제된 사실'을 뽑아 저장하고 갱신/충돌을 관리한다.")
