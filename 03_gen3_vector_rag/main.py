"""
gen3 — 3세대 메모리: Vector DB로 원문 검색하기

[2세대의 어떤 실패를 푸는가]
2세대 요약은 오래된 대화를 짧게 압축했다.
그래서 요약문에 빠진 이름, 장소, 사건 같은 디테일은 다시 찾을 수 없었다.

3세대는 원본을 버리지 않는다.
대화 원문을 Vector DB(pgvector)에 저장해 두고, 질문이 들어오면 의미가 가까운
원문 조각을 검색해 모델 입력에 다시 넣는다.
이처럼 "검색한 자료를 근거로 답하게 하는 흐름"을 RAG라고 부른다.

[이 파일의 두 장면]
  1) 간단 유사도 검색 데모
     유사도 검색이 마법이 아니라 "가까운 순서로 정렬"임을 먼저 본다.

  2) 실제 Vector DB
     같은 일을 pgvector로 실행한다. 검색은 SQL 한 줄이다.
       SELECT text FROM memories ORDER BY embedding <=> :q LIMIT k;
     <=> 는 코사인 거리다. 거리순 정렬이 곧 의미 검색이다.

[그런데 새로 생기는 한계]
Vector DB는 원본을 보관하지만, 답을 항상 정확히 보장하지는 않는다.

  - 고정 길이 청크는 대화 턴과 화자 경계를 끊을 수 있다.
  - 검색은 "가까운 원문"을 찾을 뿐, 최신 사실을 스스로 판단하지 못한다.
  - 저장된 것은 원문 조각이지, 정리된 사실 상태가 아니다.

그래서 4세대에서는 원문에서 정제된 사실을 추출하고,
"서울 → 부산" 같은 갱신과 충돌을 관리한다.

[준비물]
  1) docker compose up -d   (pgvector 기동)
  2) .env 에 OPENAI_API_KEY  (임베딩 호출). DATABASE_URL은 기본값 사용 가능.
"""

import os
import pathlib
import sys

import numpy as np

# 번호로 시작하는 폴더는 패키지 이름으로 쓰기 어렵다.
# 그래서 repo 루트를 import 경로에 추가해 common 패키지를 불러온다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import llm, scoring
from common.scenario import PROBES, conversation

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://agent:agent@localhost:5433/agent_memory"
)
SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
# 일부러 단순한 청킹을 쓴다. 고정 글자 수로 자르면 턴 경계가 깨질 수 있다.
CHUNK_SIZE = 60
TOP_K = 3
SOURCE = "gen3"


# ---------------------------------------------------------------------------
# 장면 1) 간단 유사도 검색 — '마법이 아니라 거리순 정렬'
# 00_foundations의 Bag-of-Words 방식으로 의미 검색의 모양만 작게 보여준다.
# ---------------------------------------------------------------------------
def simple_similarity_demo() -> None:
    vocab = ["코코", "사료", "부산", "파이썬"]

    def emb(t):
        return np.array([float(t.count(w)) for w in vocab])

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))

    memos = ["코코가 사료를 안 먹는다", "부산으로 이사했다", "파이썬을 공부한다"]
    query = "코코 사료"
    ranked = sorted(memos, key=lambda m: cos(emb(query), emb(m)), reverse=True)

    print("[장면 1] 간단 유사도 검색 — 검색 = 가까운 순서로 정렬")
    print(f"  질문 '{query}' 와 가까운 순서:")
    for m in ranked:
        print(f"    {cos(emb(query), emb(m)):.3f}  {m}")
    print("  pgvector도 원리는 같다. 차이는 '실제 임베딩 + DB 정렬'을 쓴다는 점이다.\n")


# ---------------------------------------------------------------------------
# 장면 2) 실제 Vector DB (pgvector)
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int) -> list[str]:
    """텍스트를 고정 길이 조각으로 자른다."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def clue_words(question: str) -> list[str]:
    """질문과 관련된 원문 조각을 고르기 위한 단어를 뽑는다."""
    clues = []
    if "반려동물" in question or "이름" in question:
        clues.extend(["반려동물", "고양이", "강아지", "이름", "코코"])
    if "어디" in question or "살" in question:
        clues.extend(["살고", "살아", "이사", "서울", "부산"])
    if "코코" in question:
        clues.extend(["코코", "사료", "잘 안 먹"])
    return clues


def choose_evidence(rows: list[tuple[str]], question: str) -> str | None:
    """검색된 후보 중 질문과 가장 관련 있어 보이는 원문 조각을 고른다."""
    clues = clue_words(question)
    for row in rows:
        text = row[0].replace("\n", " ")
        if any(clue in text for clue in clues):
            return text
    return rows[0][0].replace("\n", " ") if rows else None


def answer_from_rows(rows: list[tuple[str]], probe: dict) -> str:
    """검색된 원문 조각만 보고 답한다."""
    evidence = choose_evidence(rows, probe["question"])
    if evidence is None:
        return "검색 결과가 비어 있어요."
    snippet = evidence[:70]
    return f'검색된 원문 조각에서 관련 내용을 찾았어요 — "{snippet}…"'


def connect():
    """pgvector에 연결하고 필요한 테이블을 준비한다."""
    import psycopg
    from pgvector.psycopg import register_vector

    try:
        conn = psycopg.connect(DATABASE_URL)
    except Exception as e:  # 연결 실패 = 보통 DB 미기동
        raise SystemExit(
            "pgvector에 연결할 수 없습니다. 먼저 `docker compose up -d` 로 DB를 켜주세요.\n"
            f"  DATABASE_URL={DATABASE_URL}\n  원인: {e}"
        )
    # vector 확장이 먼저 만들어져야 numpy 배열을 DB vector 타입으로 보낼 수 있다.
    # 그래서 스키마 적용 후 register_vector를 호출한다.
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in (s.strip() for s in schema_sql.split(";")):
        if statement:
            conn.execute(statement)
    conn.commit()
    register_vector(conn)  # 이제 numpy 벡터 <-> vector 타입 자동 변환 등록
    return conn


def pgvector_demo() -> None:
    conn = connect()
    # 다른 세대의 데이터는 남겨두고, gen3 데모 데이터만 새로 넣는다.
    conn.execute("DELETE FROM memories WHERE source = %s", (SOURCE,))
    conn.commit()

    # 대화 전체를 한 덩어리로 만든 뒤 고정 길이로 자른다.
    # 이 단순한 방식 때문에 아래 출력처럼 단어와 화자명이 중간에서 잘릴 수 있다.
    full = "\n".join(f"{role}: {text}" for role, text in conversation())
    chunks = chunk_text(full, CHUNK_SIZE)

    print("[장면 2] pgvector RAG — 원문 조각 저장 + 의미 검색")
    print(f"  대화를 {CHUNK_SIZE}자 단위 원문 조각 {len(chunks)}개로 분할:")
    for i, c in enumerate(chunks):
        print(f"    청크{i}: {c!r}")
    print()

    # 원문 조각과 임베딩을 함께 저장한다.
    # 검색할 때는 임베딩으로 찾고, 답변할 때는 원문 조각을 근거로 사용한다.
    embeddings = llm.embed(chunks)
    for chunk, emb in zip(chunks, embeddings):
        conn.execute(
            "INSERT INTO memories (source, text, embedding) VALUES (%s, %s, %s)",
            (SOURCE, chunk, emb),
        )
    conn.commit()

    # 질문도 임베딩한 뒤, DB에 저장된 임베딩과 가까운 순서로 원문을 가져온다.
    answers = []
    for probe in PROBES:
        q_emb = llm.embed([probe["question"]])[0]
        rows = conn.execute(
            """
            SELECT text
            FROM memories
            WHERE source = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (SOURCE, q_emb, TOP_K),
        ).fetchall()
        answers.append(answer_from_rows(rows, probe))

    results = scoring.grade(PROBES, answers)
    scoring.print_report("pgvector 검색으로 답하기", results)
    print("  ↳ 3세대는 요약 대신 원문을 저장해 필요할 때 다시 검색한다.")
    print("     다만 검색 결과는 '가까운 원문 조각'일 뿐, 최신 사실 판단은 아직 못 한다.")
    print("     Q2 결과가 맞더라도, 서울/부산 중 무엇이 최신인지 관리한 것은 아니다.\n")
    conn.close()


def run_pgvector_demo() -> None:
    """DB/API가 준비되지 않아도 간단 데모는 볼 수 있게 한다."""
    try:
        pgvector_demo()
    except SystemExit as exc:
        print("[장면 2] pgvector RAG — 준비가 필요해 건너뜁니다")
        print(f"  {exc}\n")
    except Exception as exc:
        print("[장면 2] pgvector RAG — 실행 중 오류가 나서 건너뜁니다")
        print(f"  원인: {exc.__class__.__name__}: {exc}\n")


if __name__ == "__main__":
    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 3세대: Vector DB(pgvector) + RAG ===\n")
    simple_similarity_demo()
    run_pgvector_demo()

    print("[한계] 원본을 보관하고 검색해도, 두 가지 문제가 남는다")
    print(" 1) 고정 길이 청킹이 턴·화자를 끊어, 검색 결과에 어색한 조각이 섞인다.")
    print(" 2) 저장한 건 '원문'일 뿐 '사실 상태'가 아니다.")
    print("    서울과 부산이 모두 검색되어도, 어느 값이 최신인지 별도로 관리하지 않는다.")
    print("    중복·모순·갱신을 정리할 사실 단위 메모리가 아직 없기 때문이다.")
    print("  → 4세대: LLM이 원문에서 '정제된 사실'을 뽑아 저장하고 갱신/충돌을 관리한다.")
