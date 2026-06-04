"""
gen2 — 2세대 메모리: 단기 + 컨텍스트 윈도우(CW) 한도 관리

[1세대의 어떤 실패를 푸는가]
1세대는 대화를 무한정 쌓아 CW를 넘겼다(데모1). 2세대는 컨텍스트를 한도 안으로
'관리'한다. 두 가지 고전 기법:

  A) 슬라이딩 윈도우 : 최근 N턴만 남기고 옛 메시지는 버린다. CW는 일정하게 유지되지만
                      버려진 옛 사실은 그대로 사라진다.
  B) 러닝 요약(압축) : 버릴 옛 메시지를 '요약'으로 눌러 담는다. 요약 + 최근 N턴만
                      유지 → CW는 일정하면서도 옛 내용의 '요지'는 남는다.

[그런데 새로 생기는 한계]
요약은 **비가역 압축**이다. 원본을 요약으로 갈아끼우는 순간, 요약이 빠뜨린 디테일은
영영 복구할 수 없다(원본이 없으니까). 토큰 예산이 빠듯해 더 세게 압축할수록 이름·숫자
같은 구체 정보가 먼저 날아간다. → "원본을 밖에 보관했다가 필요할 때 검색"하자는
3세대(영속 Vector DB)의 동기가 된다.

[준비물] OpenAI 키(.env). 요약에 실제 LLM을 호출한다.
"""

import pathlib
import sys

# repo 루트를 sys.path에 추가(세대 예제 공통 부트스트랩).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import llm, scoring
from common.scenario import PROBES, conversation

CW_LIMIT = 500  # gen1과 같은 가상 CW 한도(토큰). 2세대는 이 안에 머무는 게 목표.
WINDOW = 4      # 컨텍스트에 남길 최근 턴 수 (= 최근 2번의 주고받음)


def est_tokens(text: str) -> int:
    """토큰 수 대략 추정(한국어 ~2글자=1토큰 근사)."""
    return max(1, len(text) // 2)


def join(turns: list) -> str:
    return "\n".join(f"{role}: {text}" for role, text in turns)


def answer_from_context(context: str, probe: dict) -> str:
    """주어진 컨텍스트(윈도우 또는 요약)만 보고 답한다.

    gen1과 같은 시뮬레이션: 필요한 사실이 컨텍스트에 남아 있으면 답할 수 있고,
    없으면 못 한다. (진짜 에이전트라면 이 컨텍스트를 LLM에 넣어 답한다)
    """
    hit = next((kw for kw in probe["expected"] if kw in context), None)
    if hit is None:
        return "음... 그건 기억에 없어요."
    return f"네, '{hit}' 관련해서 기억하고 있어요."


def summarize(turns: list, instruction: str) -> str:
    """오래된 대화를 LLM으로 요약(압축)한다."""
    prompt = (
        f"다음 대화를 {instruction}\n"
        "사용자에 대한 사실(이름·거주지·반려동물 등)을 최대한 보존해줘.\n\n"
        f"{join(turns)}"
    )
    return llm.chat(prompt)


def grade_and_report(title: str, context: str) -> None:
    answers = [answer_from_context(context, probe) for probe in PROBES]
    results = scoring.grade(PROBES, answers)
    print(f"  컨텍스트 크기: {est_tokens(context)} 토큰 (한도 {CW_LIMIT} 이내)")
    scoring.print_report(title, results)


def demo_sliding_window() -> None:
    """A) 슬라이딩 윈도우 — 최근 N턴만. 옛 사실은 버려진다."""
    print("[데모 A] 슬라이딩 윈도우 — 최근 4턴만 유지\n")
    window = conversation()[-WINDOW:]
    print("  남은 컨텍스트:")
    print("   " + join(window).replace("\n", "\n   ") + "\n")
    grade_and_report("윈도우만 (옛 대화 버림)", join(window))
    print("  ↳ 옛 세션(코코·서울·사료)이 잘려나가 Q1·Q3을 못 답한다.\n")


def demo_running_summary() -> None:
    """B) 러닝 요약 — 버릴 옛 대화를 요약으로 눌러 담는다."""
    print("[데모 B] 러닝 요약 — 옛 대화는 요약 + 최근 4턴 유지\n")
    turns = conversation()
    older, recent = turns[:-WINDOW], turns[-WINDOW:]
    summary = summarize(older, "핵심 사실 위주로 2~3문장으로 요약해줘.")
    print(f"  옛 대화 {len(older)}턴 → 요약:")
    print(f"   \"{summary}\"\n")
    context = f"[이전 요약] {summary}\n{join(recent)}"
    grade_and_report("요약 + 윈도우", context)
    print("  ↳ 요약이 옛 요지를 보존해 윈도우보다 더 많이 답한다(요약 > 윈도우).\n")


def demo_aggressive_summary() -> None:
    """[한계] 압축을 더 세게 하면 디테일이 사라진다(비가역)."""
    print("[한계 시연] 토큰 예산이 빠듯하면 더 세게 압축 → 디테일 소실\n")
    summary = summarize(conversation(), "딱 한 문장으로 아주 짧게 요약해줘.")
    print(f"  전체 대화 → 한 문장 요약:")
    print(f"   \"{summary}\"\n")
    grade_and_report("초압축 요약", summary)
    print("  ↳ 위 정답률이 떨어진 항목이 바로 '압축으로 날아간 디테일'이다.")
    print("    원본을 요약으로 갈아끼웠으니 그 정보는 되돌릴 수 없다.\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 2세대: 슬라이딩 윈도우 + 요약 (CW 한도 관리) ===\n")
    demo_sliding_window()
    demo_running_summary()
    demo_aggressive_summary()

    print("[정리] 윈도우는 옛 사실을 '버리고', 요약은 '압축'한다 — 둘 다 CW는 지키지만")
    print(" 요약은 비가역 손실이 남는다. → 3세대: 원본을 영속 Vector DB에 저장해 두고")
    print("   필요할 때 의미 검색으로 '원본 그대로' 되살린다(손실 없는 회상).")
