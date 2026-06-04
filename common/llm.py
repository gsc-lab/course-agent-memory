"""
llm.py — OpenAI 호출 헬퍼

세대 예제마다 OpenAI 클라이언트 생성, API 키 확인, 호출 코드를 반복하지 않도록
한 곳에 모았다.

  - chat()  : 요약, 사실 추출, LLM 판정에 사용한다.
  - embed() : 텍스트를 임베딩 벡터로 바꿀 때 사용한다.

API 키 확인은 실제 호출 시점에만 한다.
그래서 이 파일을 import하는 것만으로는 프로그램이 멈추지 않는다.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # .env의 OPENAI_API_KEY를 환경변수로 불러온다.

CHAT_MODEL = "gpt-4o-mini"  # 요약·판정에 쓰는 기본 챗 모델
EMBED_MODEL = "text-embedding-3-small"  # 1536차원 임베딩 모델

_client = None


def _get_client():
    """OpenAI 클라이언트를 처음 호출할 때 한 번만 만든다."""
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY가 없습니다. 프로젝트 루트에 .env 파일을 만들고 "
                "OPENAI_API_KEY=sk-... 를 넣어주세요 (.env.example 참고)."
            )
        from openai import OpenAI

        _client = OpenAI()
    return _client


def chat(prompt: str, system: str | None = None, model: str = CHAT_MODEL,
         temperature: float = 0.0) -> str:
    """프롬프트를 보내고 응답 텍스트만 돌려준다."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _get_client().chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()


def embed(texts: list[str], model: str = EMBED_MODEL):
    """여러 문장을 한 번에 임베딩하고 numpy 배열로 돌려준다."""
    import numpy as np

    resp = _get_client().embeddings.create(model=model, input=texts)
    # 응답 순서가 바뀌어도 입력 순서와 맞도록 정렬한다.
    ordered = sorted(resp.data, key=lambda item: item.index)
    return np.array([item.embedding for item in ordered])
