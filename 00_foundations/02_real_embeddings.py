"""
02_real_embeddings.py — AI Agent 메모리 2단계: 실제 임베딩 모델 사용하기

[01번 예제와 무엇이 다른가]
01번의 장난감 embed()는 글자가 겹치는지만 확인했다.
그래서 "고양이"와 "냥이"처럼 의미는 같지만 표현이 다른 단어를 놓쳤다.
실제 임베딩 모델은 학습을 통해 이런 의미 관계를 더 잘 잡아낸다.

여기서는 OpenAI의 text-embedding-3-small 모델로 텍스트를 벡터화한다.
사고방식은 01번과 같다: 텍스트를 벡터로 바꾸고, 코사인 유사도로 비교한다.
달라진 점은 embed()가 직접 만든 규칙이 아니라 실제 모델을 호출한다는 것이다.

[준비물]
  1) pip install -r requirements.txt  (openai, numpy, python-dotenv)
  2) 프로젝트 루트에 .env 파일 생성 후 OPENAI_API_KEY=sk-... 기입
     (.env.example 참고. .env는 .gitignore 처리되어 커밋되지 않는다)

[비용/속도 메모]
text-embedding-3-small은 저렴하지만 유료 API다.
네트워크를 사용하므로 01번 예제보다 느릴 수 있다.
실제 서비스에서는 같은 텍스트를 반복 호출하지 않도록 벡터를 저장해 둔다.
"""

import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일의 OPENAI_API_KEY를 환경변수로 불러온다.
load_dotenv()

MODEL = "text-embedding-3-small"  # 1536차원 임베딩을 만드는 학습용 기본 모델

# API 키가 없으면 모델을 호출할 수 없으므로 먼저 확인한다.
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit(
        "OPENAI_API_KEY가 없습니다. 프로젝트 루트에 .env 파일을 만들고 "
        "OPENAI_API_KEY=sk-... 를 넣어주세요 (.env.example 참고)."
    )

client = OpenAI()  # 환경변수의 OPENAI_API_KEY를 사용한다.


def embed(texts: list[str]) -> np.ndarray:
    """여러 문장을 한 번의 API 호출로 임베딩한다.

    문장을 리스트로 묶어 보내면 API 호출 횟수가 줄어 더 효율적이다.
    반환은 (문장 수, 1536) 모양의 numpy 배열.
    """
    resp = client.embeddings.create(model=MODEL, input=texts)
    # 응답 순서가 바뀌어도 입력 순서와 맞도록 index 기준으로 정렬한다.
    ordered = sorted(resp.data, key=lambda item: item.index)
    return np.array([item.embedding for item in ordered])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도를 numpy로 계산한다."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 01번의 장난감 임베딩이 놓쳤던 표현 차이를 실제 모델로 다시 확인한다.
    memory = "사용자는 고양이를 반려동물로 키운다"
    candidates = [
        "그 사람은 냥이를 키우는 집사다",        # 같은 의미, 표현은 다름
        "강아지를 반려동물로 키운다",            # 비슷하지만 동물이 다름
        "나는 파이썬 프로그래밍 언어를 배운다",  # 관련 없는 주제
    ]

    # 기준 기억과 후보 문장을 한 번에 임베딩한다. 첫 번째 벡터가 기준 기억이다.
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
    scored.sort(key=lambda pair: pair[1], reverse=True)  # 유사도가 높은 순서

    for rank, (text, score) in enumerate(scored, start=1):
        print(f"  {rank}위  유사도 {score:.3f}  | {text}")

    print("\n[01번과 비교 — 이게 진짜 임베딩의 힘]")
    print(" - '냥이/집사'는 '고양이/키운다'와 글자가 거의 안 겹치는데도 유사도가 높다.")
    print("   01번 장난감 embed()였다면 유사도 0이 나왔을 문장이다.")
    print(" - 글자 겹침이 아니라 '의미'로 가까움을 재기 때문에 가능한 결과다.")
    print(" - 3위(프로그래밍)는 여전히 가장 낮다 — 주제가 정말 다르면 멀다.")
    print(" - 참고: 진짜 임베딩은 무관한 문장도 유사도가 딱 0이 아니라")
    print("   보통 0.1~0.3 정도의 '약한 양수'로 나온다 (완전한 직교는 드물다).")
