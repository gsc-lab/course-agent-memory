"""
gen1 — 1세대 메모리: 단기 메모리만 사용하기

[이 세대의 방식]
가장 단순한 방식이다. 대화 내용을 따로 저장하지 않고,
지금까지 주고받은 메시지를 모델 입력(context window, CW)에 그대로 넣는다.

짧은 대화에서는 잘 작동한다. 방금 한 말이 아직 모델 입력 안에 있기 때문이다.

[그런데 두 가지가 무너진다]
1) CW 초과
   대화가 길어지면 입력 토큰이 계속 늘어난다.
   하지만 CW 크기는 고정되어 있어, 오래된 메시지는 결국 입력에서 빠진다.

2) 휘발성
   기억이 파일이나 DB에 저장되지 않는다.
   프로그램을 다시 시작하면 이전 대화는 남아 있지 않다.

이 파일은 같은 공통 시나리오로 위 두 한계를 직접 보여준다.

[참고]
실제 에이전트는 CW 안에 들어간 대화를 LLM에 넣고 답하게 한다.
여기서는 외부 API를 쓰지 않기 위해, 질문과 관련된 발화가 CW 안에 남아 있는지만
확인한다. 중요한 점은 "전체 대화"가 아니라 "모델이 지금 볼 수 있는 대화"다.
"""

import pathlib
import sys

# 번호로 시작하는 폴더는 패키지 이름으로 쓰기 어렵다.
# 그래서 repo 루트를 import 경로에 추가해 common 패키지를 불러온다.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import scoring
from common.scenario import PROBES, conversation

# 실습용 CW 한도. 실제 모델도 각자 정해진 최대 입력 토큰 수가 있다.
CW_LIMIT = 500


def est_tokens(text: str) -> int:
    """문자 길이로 토큰 수를 대략 추정한다."""
    return max(1, len(text) // 2)


class ShortTermMemory:
    """현재 실행 중인 프로그램 안에서만 유지되는 대화 목록."""

    def __init__(self):
        self.turns = []  # 예: ("user", "안녕")

    def add(self, role: str, text: str) -> None:
        self.turns.append((role, text))

    def text(self) -> str:
        return "\n".join(f"{role}: {text}" for role, text in self.turns)

    def total_tokens(self, turns: list[tuple[str, str]] | None = None) -> int:
        target = self.turns if turns is None else turns
        return sum(est_tokens(text) for _, text in target)

    def visible_turns(self, limit: int) -> list[tuple[str, str]]:
        """모델 입력에 넣을 수 있는 최신 메시지만 돌려준다."""
        visible = []
        used = 0
        for role, text in reversed(self.turns):
            cost = est_tokens(text)
            if visible and used + cost > limit:
                break
            visible.append((role, text))
            used += cost
        return list(reversed(visible))

    def __len__(self) -> int:
        return len(self.turns)


def clue_words(question: str) -> list[str]:
    """질문과 관련된 과거 발화를 찾기 위한 단어를 고른다."""
    clues = []
    if "반려동물" in question or "이름" in question:
        clues.extend(["반려동물", "고양이", "강아지", "이름"])
    if "어디" in question or "살" in question:
        clues.extend(["살고", "살아", "거주", "이사"])
    if "코코" in question:
        clues.append("코코")
    return clues


def answer(memory: ShortTermMemory, probe: dict, limit: int = CW_LIMIT) -> str:
    """모델이 볼 수 있는 최신 대화만 사용해 질문에 답한다."""
    visible = memory.visible_turns(limit)
    clues = clue_words(probe["question"])
    evidence = next(
        (
            text
            for role, text in reversed(visible)
            if role == "user" and any(clue in text for clue in clues)
        ),
        None,
    )
    if evidence is None:
        return "음... 그건 잘 기억이 안 나요."
    return f'네, 기억나요 — "{evidence}" 라고 하셨어요.'


def unrelated_followup_turns(count: int = 24) -> list[tuple[str, str]]:
    """중요한 기억을 뒤로 밀어낼 무관한 후속 대화를 만든다."""
    topics = [
        "다음 주 회의 준비",
        "점심 메뉴 후보",
        "운동 루틴 조정",
        "노트북 구매 기준",
        "여행 짐 체크리스트",
        "프로젝트 일정 정리",
    ]
    turns = []
    for i in range(1, count + 1):
        topic = topics[(i - 1) % len(topics)]
        turns.append((
            "user",
            f"{topic}에 대해 {i}번째로 이어서 이야기해보자. "
            "조건과 우선순위를 다시 정리하고 싶어.",
        ))
        turns.append((
            "assistant",
            f"좋아요. {topic}은 목적, 제약, 다음 행동 순서로 나누어 "
            "정리하면 따라가기 쉽습니다.",
        ))
    return turns


def demo_context_window() -> None:
    """데모 2: 대화가 길어지면 오래된 기억이 CW 밖으로 밀려난다."""
    print("[데모 2] CW 한도 — 오래된 기억이 모델 입력에서 빠진다")
    print(f"  실습용 CW 한도: {CW_LIMIT} 토큰")

    memory = ShortTermMemory()
    for role, text in conversation():
        memory.add(role, text)
    print(f"  핵심 대화 직후: {memory.total_tokens():>4} 토큰, 메시지 {len(memory)}개")

    for role, text in unrelated_followup_turns():
        memory.add(role, text)

    visible = memory.visible_turns(CW_LIMIT)
    hidden_count = len(memory) - len(visible)
    print(f"  후속 대화 이후 전체 누적: {memory.total_tokens():>4} 토큰, 메시지 {len(memory)}개")
    print(f"  모델 입력에 남은 최신 대화: {memory.total_tokens(visible):>4} 토큰, 메시지 {len(visible)}개")
    print(f"  모델이 볼 수 없는 오래된 메시지: {hidden_count}개\n")

    answers = [answer(memory, probe) for probe in PROBES]
    results = scoring.grade(PROBES, answers)
    scoring.print_report("데모 2 · 긴 대화 후 모델 입력만 볼 때", results)


def demo_with_memory() -> None:
    """데모 1: 짧은 대화에서는 단기 메모리만으로도 답할 수 있다."""
    memory = ShortTermMemory()
    for role, text in conversation():
        memory.add(role, text)
    answers = [answer(memory, probe) for probe in PROBES]
    results = scoring.grade(PROBES, answers)
    scoring.print_report("데모 1 · 짧은 대화 직후", results)


def demo_after_restart() -> None:
    """데모 3: 재시작하면 프로그램 안에 있던 대화 목록이 사라진다."""
    memory = ShortTermMemory()  # 새로 실행한 프로그램처럼 빈 상태
    answers = [answer(memory, probe) for probe in PROBES]
    results = scoring.grade(PROBES, answers)
    scoring.print_report("데모 3 · 프로그램 재시작 후", results)


if __name__ == "__main__":
    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 1세대: 단기 메모리만 ===\n")
    demo_with_memory()
    demo_context_window()
    demo_after_restart()

    print("[한계] 단기 메모리만으로는 부족한 이유")
    print(" 1) CW 초과 : 긴 대화에서는 오래된 메시지가 모델 입력에서 빠진다.")
    print("    → 2세대는 슬라이딩 윈도우와 요약으로 입력 크기를 관리한다.")
    print(" 2) 휘발성  : 프로그램을 다시 시작하면 이전 대화가 남아 있지 않다.")
    print("    → 3세대는 Vector DB에 기억을 저장해 다음 실행에서도 다시 찾는다.")
