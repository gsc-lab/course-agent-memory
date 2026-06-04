"""
gen4 — 4세대 메모리: 사실 추출 + 갱신 관리

[3세대의 어떤 실패를 푸는가]
3세대는 원문 조각을 저장했다.
그래서 "서울 → 부산"처럼 값이 바뀐 사실도 서울과 부산을 모두 원문으로 보관했다.
검색으로 관련 조각을 찾을 수는 있지만, 어느 값이 최신인지는 따로 관리하지 않았다.

4세대는 원문을 그대로 저장하지 않고, LLM이 뽑은 "기억할 사실"을 저장한다.
핵심은 단순히 사실을 추가하는 것이 아니다. 새 사실이 들어올 때 기존 사실과 비교해
ADD / UPDATE / NOOP 중 하나로 상태를 관리하는 것이다.

[메모리는 단순 추가가 아니라 '상태 관리'다]
  - ADD    : 처음 보는 사실이면 새로 저장한다.
  - UPDATE : 같은 속성의 값이 바뀌면 옛 사실을 새 사실로 교체한다.
  - NOOP   : 이미 있는 사실이면 아무것도 하지 않는다.

이렇게 하면 "거주지"처럼 값이 바뀌는 정보는 최신값 하나로 유지할 수 있다.

[그런데 새로 생기는 한계]
사실을 하나씩 찾는 데는 좋아졌지만, 아직 부족한 점이 있다.

  - "코코와 관련된 모든 것"처럼 한 대상에 얽힌 여러 사실을 모으기 어렵다.
  - 벡터 검색은 의미가 비슷한 것을 찾는 데 강하지만, 정확한 키워드 일치에는 약하다.

그래서 5세대에서는 벡터 검색에 엔티티·관계 그래프와 키워드 검색을 함께 쓴다.

[참고] 추출과 판정에 LLM을 쓰므로, 결과(추출 문구·ADD/UPDATE 판정)는 모델·실행에
따라 조금씩 달라질 수 있다.

[준비물]
  1) docker compose up -d
  2) .env에 OPENAI_API_KEY 설정
"""

import pathlib
import sys

# 번호로 시작하는 폴더는 패키지 이름으로 쓰기 어렵다.
# 그래서 repo 루트를 import 경로에 추가해 common 패키지를 불러온다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import db, llm, scoring
from common.scenario import PROBES, user_messages

# 새 사실과 기존 사실의 유사도가 이 값 이상일 때만 LLM에게 비교 판단을 맡긴다.
# 너무 낮으면 무관한 사실까지 비교하고, 너무 높으면 갱신을 놓칠 수 있다.
SIM_THRESHOLD = 0.5
TOP_K = 3


def extract_facts(user_message: str) -> list[str]:
    """사용자 발화에서 오래 기억할 사실만 뽑는다."""
    prompt = (
        "다음 사용자 발화에서 '오래 기억할 사실'만 짧은 문장으로 뽑아줘.\n"
        "규칙:\n"
        "- 한 줄에 하나, 군더더기 없이. 주어를 명확히(예: '지훈은 ...').\n"
        "- 변화·이동은 '현재 상태'로 정규화해줘. "
        "예: '서울에서 부산으로 이사했다' → '지훈은 부산에 산다'.\n"
        "기억할 사실이 없으면 'NONE'만 출력.\n\n"
        f"발화: {user_message}"
    )
    out = llm.chat(prompt)
    facts = []
    for line in out.splitlines():
        cleaned = line.strip().lstrip("-•*0123456789. ").strip()
        # 'NONE', 'NONE.' 같은 빈 표시는 사실이 아니므로 건너뛴다.
        if cleaned and cleaned.strip(" .。!").upper() != "NONE":
            facts.append(cleaned)
    return facts


def decide(new_fact: str, similar_fact: str) -> str:
    """새 사실을 기존 사실과 비교해 ADD/UPDATE/NOOP 중 하나로 분류한다."""
    prompt = (
        "기존 기억과 새 사실을 비교해 한 단어로 결정해줘.\n"
        f"기존: {similar_fact}\n새 사실: {new_fact}\n\n"
        "- 의미가 같으면(중복) NOOP\n"
        "- 같은 속성의 값이 바뀐 것이면(예: 거주지 변경) UPDATE\n"
        "- 서로 다른 새 정보면 ADD\n"
        "ADD, UPDATE, NOOP 중 하나만 출력."
    )
    verdict = llm.chat(prompt).strip().upper()
    # 모델이 'UPDATE.', '**UPDATE**', '결정: UPDATE'처럼 군더더기를 붙여도
    # 견고하게 인식한다. (정확히 일치만 보면 살짝만 달라도 ADD로 잘못 떨어진다)
    for v in ("UPDATE", "NOOP", "ADD"):
        if v in verdict:
            return v
    return "ADD"


def upsert_fact(conn, fact: str) -> tuple[str, str | None]:
    """사실 하나를 추가하거나, 기존 사실을 갱신하거나, 중복이면 무시한다."""
    emb = llm.embed([fact])[0]
    # 먼저 가장 비슷한 기존 사실을 찾는다. 비슷한 사실이 있어야 갱신/중복 여부를 판단할 수 있다.
    row = conn.execute(
        "SELECT id, fact, 1 - (embedding <=> %s) AS sim "
        "FROM facts ORDER BY embedding <=> %s LIMIT 1",
        (emb, emb),
    ).fetchone()

    if row and row[2] >= SIM_THRESHOLD:
        decision, prior = decide(fact, row[1]), row[1]
    else:
        decision, prior = "ADD", None

    if decision == "UPDATE" and row:
        conn.execute(
            "UPDATE facts SET fact=%s, embedding=%s, updated_at=now() WHERE id=%s",
            (fact, emb, row[0]),
        )
    elif decision == "ADD":
        conn.execute("INSERT INTO facts (fact, embedding) VALUES (%s, %s)", (fact, emb))
    # NOOP은 이미 같은 사실이 있다는 뜻이므로 저장소를 바꾸지 않는다.
    conn.commit()
    return decision, prior


def ingest():
    """사용자 발화를 시간 순서대로 처리하며 사실 저장소를 갱신한다."""
    conn = db.connect()
    # 데모를 매번 같은 상태에서 시작하기 위해 facts 테이블을 비운다.
    conn.execute("TRUNCATE facts RESTART IDENTITY")
    conn.commit()

    print("[사실 추출 + 상태 관리] 발화를 순서대로 처리하며 저장소 갱신\n")
    tags = {"ADD": "+ ADD   ", "UPDATE": "~ UPDATE", "NOOP": "= NOOP  "}
    for message in user_messages():
        for fact in extract_facts(message):
            decision, prior = upsert_fact(conn, fact)
            line = f"  {tags[decision]} {fact}"
            if decision == "UPDATE":
                line += f"   (옛 사실 '{prior}' 교체)"
            print(line)
    print()

    rows = conn.execute("SELECT fact FROM facts ORDER BY id").fetchall()
    print("  최종 사실 저장소 (중복 제거 · 최신값 유지):")
    for r in rows:
        print(f"    - {r[0]}")
    print()
    return conn


def answer_probes(conn) -> None:
    """최종 사실 저장소에서 평가 질문에 가까운 사실을 검색한다."""
    answers = []
    for probe in PROBES:
        q_emb = llm.embed([probe["question"]])[0]
        rows = conn.execute(
            "SELECT fact FROM facts ORDER BY embedding <=> %s LIMIT %s",
            (q_emb, TOP_K),
        ).fetchall()
        # 정답 키워드를 미리 보지 않고, 검색된 사실 자체를 답으로 사용한다.
        answers.append(" / ".join(r[0] for r in rows) if rows else "관련 사실이 없어요.")
    results = scoring.grade(PROBES, answers)
    scoring.print_report("정제된 사실로 답하기", results)


if __name__ == "__main__":
    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 4세대: 사실 추출 + 갱신 관리 ===\n")
    conn = ingest()
    answer_probes(conn)
    conn.close()

    print("[한계] 사실을 하나씩 관리해도, 두 가지 문제가 남는다")
    print(" 1) '코코와 관련된 모든 것'처럼 한 대상에 얽힌 여러 사실을 모으기 어렵다.")
    print(" 2) 벡터 검색은 의미 검색이라, 정확한 키워드 일치에는 약할 수 있다.")
    print("  → 5세대: 벡터 + 엔티티·관계 그래프 + 키워드 검색을 함께 사용한다.")
