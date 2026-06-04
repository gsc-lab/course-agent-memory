"""
llm.py — OpenAI 호출 공용 래퍼 (챗 / 임베딩)

세대 예제마다 OpenAI 클라이언트 생성·키 확인·호출 코드를 반복하지 않도록 한 곳에
모았다. gen2(요약)·gen4(사실 추출)·capstone이 chat()을, gen3 이후가 embed()를 쓴다.

키 확인은 '실제로 호출할 때' 한 번만 한다(지연 초기화) — import만으로는 키가 없어도
죽지 않으므로, 채점기의 keyword 모드처럼 LLM이 필요 없는 코드는 영향받지 않는다.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # .env의 OPENAI_API_KEY를 환경변수로 올린다

CHAT_MODEL = "gpt-4o-mini"  # 저렴한 기본 챗 모델 (요약·판정용). 필요 시 교체.
EMBED_MODEL = "text-embedding-3-small"  # 1536차원 임베딩 (gen3~)

_client = None


def _get_client():
    """OpenAI 클라이언트를 한 번만 만들어 재사용한다(지연 초기화)."""
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
    """프롬프트 하나를 보내 응답 텍스트를 받는다."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _get_client().chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content.strip()


def embed(texts: list[str], model: str = EMBED_MODEL):
    """여러 문장을 한 번의 호출로 임베딩해 (문장 수, 차원) numpy 배열로 돌려준다."""
    import numpy as np

    resp = _get_client().embeddings.create(model=model, input=texts)
    ordered = sorted(resp.data, key=lambda item: item.index)  # 입력 순서 보장
    return np.array([item.embedding for item in ordered])
