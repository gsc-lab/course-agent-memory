"""
02_real_embeddings.py — AI Agent 메모리 2단계: 진짜 임베딩 모델 써보기

[01번 예제와 무엇이 다른가]
01_embedding_basics.py의 장난감 embed()는 '사전 단어가 글자로 겹치는지'만 봤다.
그래서 "고양이"와 "냥이"처럼 의미는 같지만 글자가 다른 단어는 매칭하지 못했다.
이 한계를 학습으로 극복한 것이 진짜 임베딩 모델이다.

여기서는 OpenAI의 text-embedding-3-small 모델로 텍스트를 벡터화한다.
이 모델은 대량의 텍스트로 학습되어, 글자가 하나도 안 겹쳐도
'의미가 가까우면 벡터도 가깝게' 만들어준다. 사고방식(텍스트→벡터→코사인
유사도)은 01번과 똑같고, embed() 한 줄만 진짜 모델로 바뀐 것이다.

[준비물]
  1) pip install -r requirements.txt  (openai, numpy, python-dotenv)
  2) 프로젝트 루트에 .env 파일 생성 후 OPENAI_API_KEY=sk-... 기입
     (.env.example 참고. .env는 .gitignore 처리되어 커밋되지 않는다)

[비용/속도 메모]
text-embedding-3-small은 매우 저렴하지만(100만 토큰당 약 $0.02) 그래도 '유료
API 호출'이다. 그리고 네트워크를 타므로 01번처럼 즉시 끝나진 않는다. 같은
텍스트를 반복 임베딩하지 않도록, 실제 메모리 시스템에서는 벡터를 저장(캐시)해
둔다 — 이 개념은 뒤 단계에서 다룬다.
"""

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# .env에서 OPENAI_API_KEY를 읽어 환경변수로 올린다.
load_dotenv()

MODEL = "text-embedding-3-small"  # 차원 1536, 저렴하고 품질 충분 (학습용 기본값)

client = OpenAI()  # OPENAI_API_KEY 환경변수를 자동으로 사용


def embed(texts: list[str]) -> np.ndarray:
    """여러 문장을 한 번의 API 호출로 임베딩한다.

    문장을 하나씩 호출하지 않고 리스트로 묶어 보내는 이유:
    호출 횟수(=네트워크 왕복)가 줄어 더 빠르고, 묶음 처리라 효율적이다.
    반환은 (문장 수, 1536) 모양의 numpy 배열.
    """
    resp = client.embeddings.create(model=MODEL, input=texts)
    return np.array([item.embedding for item in resp.data])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도. 개념은 01번과 동일하나 numpy로 간결하게 계산한다."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    # 01번에서 장난감 임베딩이 풀지 못했던 바로 그 문제를 진짜 모델로 다시 본다.
    memory = "사용자는 고양이를 반려동물로 키운다"
    candidates = [
        "그 사람은 냥이를 키우는 집사다",        # 같은 의미, 글자는 거의 안 겹침
        "강아지를 반려동물로 키운다",            # 비슷한 주제(반려동물)
        "나는 파이썬 프로그래밍 언어를 배운다",  # 완전히 다른 주제
    ]

    # 기준 기억 + 후보들을 한 번에 임베딩 (첫 줄이 기준)
    vectors = embed([memory] + candidates)
    mem_vec, cand_vecs = vectors[0], vectors[1:]

    print(f"모델: {MODEL}")
    print(f"임베딩 차원 수: {mem_vec.shape[0]}")
    print(f"벡터 앞 5개 값(예시): {np.round(mem_vec[:5], 4)}")
    print("  (1536개 숫자 중 일부. 사람이 해석할 수 없지만, 의미를 담은 좌표다)\n")

    print(f"[의미 유사도] 기준 기억: {memory}\n")

    scored = [
        (text, cosine_similarity(mem_vec, vec))
        for text, vec in zip(candidates, cand_vecs)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)  # 유사도 높은 순 = 떠올릴 순서

    for rank, (text, score) in enumerate(scored, start=1):
        print(f"  {rank}위  유사도 {score:.3f}  | {text}")

    print("\n[01번과 비교 — 이게 진짜 임베딩의 힘]")
    print(" - '냥이/집사'는 '고양이/키운다'와 글자가 거의 안 겹치는데도 유사도가 높다.")
    print("   01번 장난감 embed()였다면 유사도 0이 나왔을 문장이다.")
    print(" - 글자 겹침이 아니라 '의미'로 가까움을 재기 때문에 가능한 결과다.")
    print(" - 3위(프로그래밍)는 여전히 가장 낮다 — 주제가 정말 다르면 멀다.")
    print(" - 참고: 진짜 임베딩은 무관한 문장도 유사도가 딱 0이 아니라")
    print("   보통 0.1~0.3 정도의 '약한 양수'로 나온다 (완전한 직교는 드물다).")
