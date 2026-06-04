"""
gen1 — 1세대 메모리: 단기 메모리만 (대화를 컨텍스트에 그대로 쌓기)

[이 세대의 방식]
가장 단순한 메모리. 주고받은 메시지를 전부 컨텍스트 윈도우(CW)에 누적해 둔다.
"지금 이 대화" 안에서는 잘 작동한다 — 방금 한 말을 기억하니까.

[그런데 두 가지가 무너진다]
1) CW 초과 : 대화가 길어질수록 토큰이 계속 늘지만, 모델의 컨텍스트 윈도우는
             고정이다. 언젠가 한도를 넘어 옛 메시지가 잘려 나간다.
2) 휘발성  : 메모리가 '대화(프로세스) 안'에만 있다. 재시작하면 전부 사라진다.

이 파일은 같은 공통 시나리오로 위 두 한계를 눈으로 보여준다.

[참고] 진짜 에이전트라면 누적한 대화를 LLM에 넣어 답하게 한다. 여기서는 LLM
호출 없이(stdlib만으로) '필요한 사실이 메모리에 남아있는가'로 답변 가능성을
시뮬레이션한다 — 메모리에 있으면 LLM도 답할 수 있고, 없으면 못 답하기 때문이다.
"""

import pathlib
import sys

# 번호 폴더(01_gen1_...)는 `python -m`으로 못 돌리므로, repo 루트를 sys.path에 넣어
# common 패키지를 import할 수 있게 한다. (세대 예제 공통 부트스트랩)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import scoring
from common.scenario import PROBES, conversation

# 가상 컨텍스트 윈도우 한도(토큰). 실제 모델은 8K~200K 등으로 고정돼 있다.
CW_LIMIT = 500


def est_tokens(text: str) -> int:
    """토큰 수 대략 추정(한국어 ~2글자=1토큰 근사). 실제 토크나이저와는 다르다."""
    return max(1, len(text) // 2)


class ShortTermMemory:
    """대화 메시지를 그대로 쌓아 두는 단기 메모리. (재시작하면 사라진다)"""

    def __init__(self):
        self.turns = []  # (role, text) 목록

    def add(self, role: str, text: str) -> None:
        self.turns.append((role, text))

    def text(self) -> str:
        return "\n".join(f"{role}: {text}" for role, text in self.turns)

    def total_tokens(self) -> int:
        return sum(est_tokens(text) for _, text in self.turns)

    def __len__(self) -> int:
        return len(self.turns)


def answer(memory: ShortTermMemory, probe: dict) -> str:
    """probe에 답한다(LLM 없이 시뮬레이션 — 위 docstring 참고)."""
    blob = memory.text()
    hit = next((kw for kw in probe["expected"] if kw in blob), None)
    if hit is None:
        return "음... 그건 잘 기억이 안 나요."
    evidence = next((text for _, text in memory.turns if hit in text), hit)
    return f'네, 기억나요 — "{evidence}" 라고 하셨어요.'


def demo_context_window() -> None:
    """데모 1: 단기 메모리는 무한정 쌓여 CW 한도를 넘는다."""
    print("[데모 1] 컨텍스트 윈도우 한도 — 단기 메모리는 끝없이 쌓인다")
    print(f"  가상 CW 한도: {CW_LIMIT} 토큰 (모델마다 고정값)\n")

    memory = ShortTermMemory()
    turns = conversation()
    crossed_at = None
    # 같은 대화가 계속 이어지는 '긴 세션'을 흉내 내며 누적 토큰을 추적한다.
    for round_no in range(1, 51):
        for role, text in turns:
            memory.add(role, text)
        print(f"  대화 {round_no:>2}바퀴 후: 누적 {memory.total_tokens():>4} 토큰 "
              f"(메시지 {len(memory)}개)")
        if memory.total_tokens() > CW_LIMIT:
            crossed_at = round_no
            break

    print(f"\n  → {crossed_at}바퀴째에 한도 초과. 토큰은 계속 늘지만 CW는 고정이라")
    print("     언젠가 반드시 넘친다 — 초과분(옛 메시지)은 모델에 들어가지 못한다.\n")


def demo_with_memory() -> None:
    """데모 2: 대화 직후, 메모리가 살아있으면 잘 답한다."""
    memory = ShortTermMemory()
    for role, text in conversation():
        memory.add(role, text)
    answers = [answer(memory, probe) for probe in PROBES]
    results = scoring.grade(PROBES, answers)
    scoring.print_report("데모 2 · 메모리가 살아있을 때 (대화 직후)", results)


def demo_after_restart() -> None:
    """데모 3: 프로세스 재시작 = 빈 메모리. 전부 잊는다(휘발성)."""
    memory = ShortTermMemory()  # 새 프로세스처럼 텅 빈 상태
    answers = [answer(memory, probe) for probe in PROBES]
    results = scoring.grade(PROBES, answers)
    scoring.print_report("데모 3 · 프로세스 재시작 후 (메모리 휘발)", results)


if __name__ == "__main__":
    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 1세대: 단기 메모리만 ===\n")
    demo_context_window()
    demo_with_memory()
    demo_after_restart()

    print("[한계] 1세대 단기 메모리의 두 가지 한계")
    print(" 1) CW 초과 : 대화가 길어지면 토큰이 한도를 넘어 옛 기억이 잘려 나간다.")
    print("    → 2세대(슬라이딩 윈도우 + 요약)가 '한도 안에서' 관리한다.")
    print(" 2) 휘발성  : 재시작하면 전부 사라진다(데모 3에서 정답률 0%).")
    print("    → 3세대(영속 Vector DB)가 '대화 밖'에 저장해 되살린다.")
