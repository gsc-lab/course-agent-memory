"""
01_embedding_basics.py — AI Agent 메모리 1단계: 임베딩 이해하기

[왜 에이전트 메모리에 임베딩이 필요한가]
에이전트는 과거 대화나 사실을 "기억"으로 저장한다.
새 질문이 들어오면 그중 관련 있는 기억만 찾아야 한다.
하지만 글자가 정확히 같은 것만 찾으면 놓치는 내용이 많다.
  - 저장된 기억: "사용자는 고양이를 키운다"
  - 새 질문:     "반려동물에 대해 뭐라고 했지?"
두 문장은 글자가 거의 다르지만 의미는 가깝다.
임베딩은 이런 "의미 기반 검색"을 가능하게 한다.

[임베딩의 핵심 아이디어]
1) 텍스트를 고정 길이의 숫자 벡터(예: [0.12, -0.04, ...])로 바꾼다.
2) 의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치하도록 만든다.
3) "가깝다"는 보통 코사인 유사도(cosine similarity)로 잰다.

[이 파일의 목표]
실제 임베딩 모델은 다음 예제에서 사용한다.
여기서는 외부 라이브러리 없이 간단한 예제로 원리만 확인한다.
"""

import math
import sys

# ---------------------------------------------------------------------------
# 1) 텍스트 -> 벡터
#
# 이 예제는 "Bag of Words(단어 가방)" 방식을 쓴다.
# VOCAB에 있는 각 단어가 문장에 몇 번 나오는지 세고,
# 그 횟수를 순서대로 나열해 벡터를 만든다.
#
# 한국어는 "고양이를", "고양이가"처럼 조사가 붙는다.
# 그래서 단어를 정확히 나누지 않고, 문장 안에 VOCAB 단어가 포함되는지 확인한다.
#
# 한계: VOCAB에 없는 단어와 동의어("고양이"와 "냥이")는 알아보지 못한다.
# 실제 임베딩 모델은 학습을 통해 이런 한계를 줄인다.
# ---------------------------------------------------------------------------

VOCAB = [
    "고양이", "강아지", "반려동물", "키운다", "좋아한다",
    "파이썬", "자바", "프로그래밍", "언어", "배운다",
]


def embed(text: str) -> list[float]:
    """텍스트를 VOCAB 길이의 벡터로 바꾼다."""
    return [float(text.count(word)) for word in VOCAB]


# ---------------------------------------------------------------------------
# 2) 두 벡터가 얼마나 비슷한가 — 코사인 유사도
#
# 코사인 유사도는 두 벡터의 방향이 얼마나 비슷한지 나타낸다.
# 1에 가까울수록 비슷하고, 0에 가까울수록 관련이 적다.
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:  # VOCAB 단어가 하나도 없는 문장
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"단어 사전(VOCAB) 차원 수: {len(VOCAB)}")
    print(f"VOCAB = {VOCAB}\n")

    # 데모 1: 텍스트가 벡터로 바뀌는 과정
    sample = "고양이를 키운다. 강아지도 좋아한다."
    print(f"[데모 1] 텍스트 -> 벡터")
    print(f"  문장: {sample}")
    print(f"  벡터: {embed(sample)}")
    print("  (VOCAB 순서대로 각 단어 등장 횟수. '고양이'1 '강아지'1 '키운다'1 '좋아한다'1)\n")

    # 데모 2: 기준 기억과 후보 문장의 유사도 비교
    memory = "사용자는 고양이를 반려동물로 키운다"
    candidates = [
        "그 사람은 반려동물로 고양이를 키운다",   # 같은 의미, 다른 표현
        "강아지를 반려동물로 키운다",             # 비슷하지만 동물이 다름
        "나는 파이썬 프로그래밍 언어를 배운다",   # 관련 없는 주제
    ]

    mem_vec = embed(memory)
    print(f"[데모 2] 기준 기억과의 의미 유사도")
    print(f"  기준 기억: {memory}\n")

    scored = [(text, cosine_similarity(mem_vec, embed(text))) for text in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)  # 유사도가 높은 순서

    for rank, (text, score) in enumerate(scored, start=1):
        print(f"  {rank}위  유사도 {score:.3f}  | {text}")

    print("\n[비교 포인트]")
    print(" - 1위: 글자는 달라도 의미가 같으면 유사도가 가장 높다 (의미 기반 검색의 핵심)")
    print(" - 2위: '반려동물·키운다'는 겹치지만 동물이 달라 1위보다 낮다")
    print(" - 3위: 공유 단어가 없어 유사도 0 — 에이전트라면 이 기억은 안 떠올린다")
    print(" - 다음 예제: 이 장난감 embed()를 실제 임베딩 모델로 교체하면,")
    print("   '냥이'/'고양이'처럼 단어가 달라도 가까운 벡터가 나오게 된다")
