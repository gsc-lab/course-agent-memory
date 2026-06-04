"""
gen5 — 5세대 메모리: 다중 신호 검색 (벡터 + 키워드 + 그래프)

[4세대의 어떤 약점을 보완하는가]
4세대는 '정제된 사실'을 잘 관리하지만, 검색 신호가 벡터(의미) 하나뿐이었다. 그래서
  - 정확한 단어로 콕 집어 찾기,
  - "코코와 관련된 것 전부"나 "지훈 → 코코 → 사료"처럼 관계를 따라가는 질의
에는 약했다. 5세대는 세 신호를 함께 쓴다.

  1) 벡터(의미)   : pgvector 코사인 — 뜻이 가까운 것
  2) 키워드(BM25) : rank-bm25 — 정확한 단어가 겹치는 것
  3) 그래프(관계) : networkx — 엔티티로 연결된 것 / 여러 단계로 이어진 것

[정직한 현실 점검 — 그래프는 실무에서 항상 쓰나?]
아니다. 벡터·BM25(하이브리드)는 거의 모든 진지한 검색이 쓰지만, 엔티티 그래프는
세 신호 중 채택률이 가장 낮다. Zep(Graphiti)은 제품 핵심으로 쓰고, mem0는 옵션으로
제공하지만, 다수 팀은 '구축·유지 비용 > 이득'이라 건너뛴다. 그래프가 값을 하는 건
**멀티홉 관계 질의**가 진짜 있을 때다. → 이 파일도 그래프는 '가볍게' 보여준다.

[참고] 작은·깨끗한 저장소에선 벡터 하나로도 충분하다. 다중 신호의 진짜 이점은
'규모와 노이즈'에서 나온다(3세대 마무리와 같은 맥락).

[준비물] docker compose up -d (pgvector) + .env OPENAI_API_KEY
"""

import pathlib
import sys

import networkx as nx
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import db, llm

# 4세대가 만들어 둔 '정제된 사실 저장소'를 입력으로 가정한다.
# (gen5의 주제는 '추출'이 아니라 '검색'이므로, 사실은 고정해 두고 검색 신호에 집중한다.)
FACTS = [
    {"text": "지훈은 부산에 산다", "entities": ["지훈", "부산"]},
    {"text": "지훈은 고양이 코코를 키운다", "entities": ["지훈", "고양이", "코코"]},
    {"text": "고양이의 이름은 코코다", "entities": ["고양이", "코코"]},
    {"text": "코코는 사료를 잘 안 먹는다", "entities": ["코코", "사료"]},
    {"text": "지훈은 파이썬 공부를 시작했다", "entities": ["지훈", "파이썬"]},
]


# ---------------------------------------------------------------------------
# 키워드 신호 — BM25
# 한국어는 조사가 붙어("사료"≠"사료를") 띄어쓰기 토큰으로는 정확 일치가 어렵다.
# 그래서 글자 2-gram으로 잘라 색인한다(형태소 분석기 없이도 단어 겹침을 잡는 방법).
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    s = text.replace(" ", "")
    return [s[i:i + 2] for i in range(len(s) - 1)] or [s]


# ---------------------------------------------------------------------------
# 점수 정규화 — 서로 스케일이 다른 신호를 [0,1]로 맞춰야 합칠 수 있다.
# ---------------------------------------------------------------------------
def normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0] * len(scores)  # 전부 같으면 변별력 없음
    return [(s - lo) / (hi - lo) for s in scores]


def build_graph() -> nx.Graph:
    """엔티티 노드와 사실 노드를 잇는 그래프를 만든다(엔티티가 사실에 등장하면 연결)."""
    g = nx.Graph()
    for i, fact in enumerate(FACTS):
        fact_node = f"F{i}"
        g.add_node(fact_node, kind="fact", text=fact["text"])
        for entity in fact["entities"]:
            g.add_node(f"E:{entity}", kind="entity")
            g.add_edge(f"E:{entity}", fact_node)
    return g


def facts_about(graph: nx.Graph, entity: str) -> list[str]:
    """엔티티에 직접 연결된 사실 전부(=그 엔티티가 등장하는 모든 사실)."""
    node = f"E:{entity}"
    if node not in graph:
        return []
    return [graph.nodes[n]["text"] for n in graph.neighbors(node)
            if graph.nodes[n]["kind"] == "fact"]


def related_facts(graph: nx.Graph, entity: str, max_edges: int) -> list[str]:
    """엔티티에서 관계를 따라 닿는 사실들(멀티홉). 직접 언급이 없어도 이어진다."""
    node = f"E:{entity}"
    if node not in graph:
        return []
    reachable = nx.single_source_shortest_path_length(graph, node, cutoff=max_edges)
    return [graph.nodes[n]["text"] for n in reachable
            if graph.nodes[n].get("kind") == "fact"]


def main():
    conn = db.connect()
    # gen5 입력으로 쓸 사실들을 facts 테이블에 적재(벡터 신호용).
    conn.execute("TRUNCATE facts RESTART IDENTITY")
    embeddings = llm.embed([f["text"] for f in FACTS])
    for fact, emb in zip(FACTS, embeddings):
        conn.execute("INSERT INTO facts (fact, embedding) VALUES (%s, %s)", (fact["text"], emb))
    conn.commit()

    bm25 = BM25Okapi([tokenize(f["text"]) for f in FACTS])
    graph = build_graph()

    # === 데모 1: 하이브리드(벡터 + BM25) 점수 합치기 ===
    print("[데모 1] 하이브리드 검색 — 벡터(의미) + BM25(키워드)를 합쳐 랭킹\n")
    query = "코코가 안 먹는 것"
    q_emb = llm.embed([query])[0]
    # 모든 사실의 벡터 유사도를 id 순서(=FACTS 순서)로 가져온다.
    vec = [r[0] for r in conn.execute(
        "SELECT 1 - (embedding <=> %s) AS sim FROM facts ORDER BY id", (q_emb,)
    ).fetchall()]
    bm = list(bm25.get_scores(tokenize(query)))
    vn, bn = normalize(vec), normalize(bm)
    fused = [0.5 * v + 0.5 * b for v, b in zip(vn, bn)]

    print(f"  질문: \"{query}\"")
    print(f"  {'사실':<22} {'벡터':>6} {'BM25':>6} {'합계':>6}")
    ranked = sorted(zip(FACTS, vn, bn, fused), key=lambda t: t[3], reverse=True)
    for fact, v, b, f in ranked:
        print(f"  {fact['text']:<22} {v:>6.2f} {b:>6.2f} {f:>6.2f}")
    print("  ↳ 두 신호를 정규화해 합친다. 깨끗한 소규모 저장소에선 둘이 대체로 같은 답을")
    print("     가리키지만, 규모·노이즈가 커지면 서로의 약점(의미 모호/조사 변형)을 메운다.\n")

    # === 데모 2: 그래프(관계) — 벡터가 놓치는 장면 ===
    print("[데모 2] 그래프 검색 — 엔티티로 모으기 / 관계를 따라가기\n")

    print('  (2a) "코코에 대해 아는 것 전부" — 엔티티에 연결된 사실을 모은다')
    for t in facts_about(graph, "코코"):
        print(f"      - {t}")
    print("      ↳ top-k 벡터는 비슷한 몇 개만 주지만, 그래프는 '코코에 연결된 전부'를")
    print("        정확히 가져온다(사실이 수백 개면 차이가 커진다).\n")

    print('  (2b) 멀티홉 — "지훈"에서 관계를 따라가기')
    q_emb2 = llm.embed(["지훈"])[0]
    top_for_jihun = [r[0] for r in conn.execute(
        "SELECT fact FROM facts ORDER BY embedding <=> %s LIMIT 3", (q_emb2,)
    ).fetchall()]
    print(f"      벡터 top-3('지훈'):")
    for t in top_for_jihun:
        print(f"        · {t}")
    print(f"      그래프 2홉('지훈'에서 관계를 따라):")
    for t in related_facts(graph, "지훈", max_edges=3):
        mark = "  ← 지훈을 직접 언급하지 않지만 코코를 거쳐 도달" if "사료" in t else ""
        print(f"        · {t}{mark}")
    print("      ↳ '코코가 안 먹는 것'은 지훈을 언급하지 않아 벡터('지훈')는 놓친다.")
    print("        그래프는 지훈 → 코코 → 사료로 이어 찾는다(멀티홉의 진짜 쓸모).\n")

    conn.close()

    print("[정리] 벡터·BM25·그래프는 서로의 약점을 메운다.")
    print(" - 벡터: 의미가 가까운 것 / BM25: 정확한 단어 / 그래프: 관계로 이어진 것")
    print(" - 단, 그래프는 구축·유지 비용이 크다 — 멀티홉 관계가 정말 필요할 때만.")
    print("  → 현재(capstone): 이 모든 기법을 묶은 통합 메모리를 직접 구현하고,")
    print("    같은 시나리오로 mem0와 비교하며 '평가'로 마무리한다.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 5세대: 다중 신호 검색 (벡터 + BM25 + 그래프) ===\n")
    main()
