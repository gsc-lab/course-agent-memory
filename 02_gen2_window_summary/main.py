"""
gen2 — 2세대 메모리: 슬라이딩 윈도우 + 요약

[1세대의 어떤 실패를 푸는가]
1세대는 대화가 길어지면 오래된 메시지가 CW 밖으로 밀려났다.
2세대는 모델 입력을 한도 안에 유지하기 위해 대화를 관리한다.

대표적인 방법은 두 가지다.

  A) 슬라이딩 윈도우
     최근 N턴만 남긴다. 입력 크기는 안정적이지만 오래된 사실은 사라진다.

  B) 러닝 요약
     오래된 대화를 짧은 요약으로 압축하고, 최근 N턴과 함께 넣는다.
     입력 크기를 줄이면서도 과거 내용의 요지는 남길 수 있다.

[그런데 새로 생기는 한계]
요약은 되돌릴 수 없는 압축이다.
요약문에 빠진 이름, 숫자, 사건 같은 디테일은 원본이 없으면 복구할 수 없다.
그래서 3세대에서는 원본을 외부 저장소(Vector DB)에 보관하고,
필요할 때 검색해서 다시 가져오는 방식을 사용한다.

[준비물]
OpenAI 키가 있으면 실제 LLM으로 요약한다.
키가 없거나 호출에 실패하면 고정 요약 예시로 흐름을 계속 보여준다.
"""

import pathlib
import sys

# 번호로 시작하는 폴더는 패키지 이름으로 쓰기 어렵다.
# 그래서 repo 루트를 import 경로에 추가해 common 패키지를 불러온다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import llm, scoring
from common.scenario import PROBES, conversation

# 실습용 CW 한도. 2세대의 목표는 이 한도 안에 입력을 맞추는 것이다.
CW_LIMIT = 500

# 최근 4턴만 그대로 남긴다. 사용자/에이전트가 두 번 주고받은 분량이다.
WINDOW = 4

FALLBACK_SUMMARY = (
    "지훈은 고양이 코코를 키우고, 코코가 사료를 잘 먹지 않아 걱정했다. "
    "그는 서울에서 부산으로 이사했다."
)

# 초압축 요약의 한계를 안정적으로 보여주기 위한 예시다.
# 주제는 남아 있지만 이름, 장소, 사건 같은 답변용 디테일은 빠져 있다.
LOSSY_SUMMARY = "사용자는 이사, 반려동물, 프로그래밍에 관해 이야기했다."


def est_tokens(text: str) -> int:
    """문자 길이로 토큰 수를 대략 추정한다."""
    return max(1, len(text) // 2)


def join(turns: list[tuple[str, str]]) -> str:
    """대화 턴 목록을 모델 입력에 넣기 쉬운 문자열로 바꾼다."""
    return "\n".join(f"{role}: {text}" for role, text in turns)


def clue_words(question: str) -> list[str]:
    """질문과 관련된 과거 발화를 찾기 위한 단어를 고른다."""
    clues = []
    # 이 함수는 LLM 대신 쓰는 작은 검색 규칙이다.
    # 정답 키워드가 아니라 질문 표현에서 단서를 뽑는 것이 중요하다.
    if "반려동물" in question or "이름" in question:
        clues.extend(["반려동물", "고양이", "강아지", "이름"])
    if "어디" in question or "살" in question:
        clues.extend(["살고", "살아", "거주", "이사"])
    if "코코" in question:
        clues.append("코코")
    return clues


def has_specific_detail(question: str, evidence: str) -> bool:
    """질문에 답할 만큼 구체적인 정보가 evidence에 남아 있는지 확인한다."""
    if "이름" in question:
        return "이름은" in evidence or "코코" in evidence
    if "어디" in question or "살" in question:
        return any(place in evidence for place in ["서울", "부산"]) and (
            "살" in evidence or "이사" in evidence
        )
    if "코코" in question:
        return "사료" in evidence or "잘 안 먹" in evidence
    return True


def answer_from_context(context: str, probe: dict) -> str:
    """주어진 모델 입력만 사용해 질문에 답한다."""
    clues = clue_words(probe["question"])
    # 최신 정보가 더 중요하므로 뒤에서부터 관련 줄을 찾는다.
    evidence = next(
        (
            line.strip()
            for line in reversed(context.splitlines())
            if any(clue in line for clue in clues)
        ),
        None,
    )
    if evidence is None:
        return "음... 그건 기억에 없어요."
    if not has_specific_detail(probe["question"], evidence):
        return f'관련 이야기는 남아 있지만, 구체적인 답은 없어요 — "{evidence}"'
    return f'네, 기억나요 — "{evidence}" 라는 내용이 남아 있어요.'


def summarize(turns: list[tuple[str, str]], instruction: str, fallback: str) -> str:
    """오래된 대화를 요약한다. LLM을 쓸 수 없으면 고정 예시를 사용한다."""
    # gen2의 핵심은 "원본을 요약으로 압축한다"는 구조다.
    # OpenAI 키가 있는 환경에서는 실제 요약 결과를 볼 수 있다.
    prompt = (
        f"다음 대화를 {instruction}\n"
        "사용자에 대한 사실(이름·거주지·반려동물 등)을 최대한 보존해줘.\n\n"
        f"{join(turns)}"
    )
    try:
        return llm.chat(prompt)
    except SystemExit:
        print("  (OPENAI_API_KEY가 없어 고정 요약 예시를 사용합니다.)")
    except Exception as exc:
        print(f"  (LLM 요약 실패: {exc.__class__.__name__}. 고정 요약 예시를 사용합니다.)")
    return fallback


def grade_and_report(title: str, context: str) -> None:
    answers = [answer_from_context(context, probe) for probe in PROBES]
    results = scoring.grade(PROBES, answers)
    tokens = est_tokens(context)
    status = "이내" if tokens <= CW_LIMIT else "초과"
    print(f"  컨텍스트 크기: {tokens} 토큰 (한도 {CW_LIMIT} {status})")
    scoring.print_report(title, results)


def demo_sliding_window() -> None:
    """A) 슬라이딩 윈도우: 최근 메시지만 남기고 오래된 메시지는 버린다."""
    print("[데모 A] 슬라이딩 윈도우 — 최근 4턴만 유지\n")
    window = conversation()[-WINDOW:]
    print("  남은 컨텍스트:")
    print("   " + join(window).replace("\n", "\n   ") + "\n")
    grade_and_report("윈도우만 (옛 대화 버림)", join(window))
    print("  ↳ 옛 세션(코코·서울·사료)이 잘려나가 Q1·Q3을 못 답한다.\n")


def demo_running_summary() -> None:
    """B) 러닝 요약: 오래된 대화를 요약으로 압축한다."""
    print("[데모 B] 러닝 요약 — 옛 대화는 요약 + 최근 4턴 유지\n")
    turns = conversation()
    # 오래된 부분은 요약하고, 최근 부분은 원문 그대로 남긴다.
    older, recent = turns[:-WINDOW], turns[-WINDOW:]
    summary = summarize(older, "핵심 사실 위주로 2~3문장으로 요약해줘.", FALLBACK_SUMMARY)
    print(f"  옛 대화 {len(older)}턴 → 요약:")
    print(f"   \"{summary}\"\n")
    context = f"[이전 요약] {summary}\n{join(recent)}"
    grade_and_report("요약 + 윈도우", context)
    print("  ↳ 요약이 옛 요지를 보존해 윈도우보다 더 많이 답한다(요약 > 윈도우).\n")


def demo_aggressive_summary() -> None:
    """[한계] 압축을 너무 세게 하면 구체적인 정보가 빠진다."""
    print("[한계 시연] 압축을 너무 세게 하면 디테일이 사라진다\n")
    summary = LOSSY_SUMMARY
    print("  전체 대화 → 아주 짧은 요약:")
    print(f"   \"{summary}\"\n")
    grade_and_report("초압축 요약", summary)
    print("  ↳ 위 정답률이 떨어진 항목이 바로 '압축으로 날아간 디테일'이다.")
    print("    원본을 요약으로 갈아끼웠으니 그 정보는 되돌릴 수 없다.\n")


if __name__ == "__main__":
    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 2세대: 슬라이딩 윈도우 + 요약 (CW 한도 관리) ===\n")
    demo_sliding_window()
    demo_running_summary()
    demo_aggressive_summary()

    print("[정리] 윈도우는 오래된 메시지를 버리고, 요약은 오래된 메시지를 압축한다.")
    print(" 둘 다 CW 한도 관리에는 도움이 되지만, 요약에는 정보 손실이 생길 수 있다.")
    print(" → 3세대: 원본을 Vector DB에 저장해 두고 필요할 때 의미 검색으로 다시 가져온다.")
