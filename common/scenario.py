"""
scenario.py — 모든 세대 예제가 공유하는 대화와 평가 질문

[왜 공통 시나리오인가]
세대마다 다른 대화로 시연하면 "무엇이 좋아졌는지" 비교가 불가능하다.
같은 멀티세션 대화 + 같은 평가 질문을 모든 세대가 처리하면,
'같은 입력'에 대해 세대별로 답이 어떻게 달라지는지 한눈에 보인다.

이 파일은 로직 없이 데이터만 제공한다:
  - SESSIONS : 시간 순서로 진행되는 멀티세션 대화
  - PROBES   : 평가 질문 + 기대 정답 키워드
채점 로직은 common/scoring.py가 맡는다. 이 파일은 '정답'만 들고 있다.

[시나리오가 의도적으로 심어 둔 함정]
  - 휘발성   : 여러 세션의 대화는 단기 메모리만으로 보존하기 어렵다.
  - 모순/갱신: S1의 서울 거주가 S3에서 부산 거주로 바뀐다.
  - 노이즈   : S4의 파이썬 이야기처럼 질문과 무관한 사실도 섞인다.
"""

# 각 세션은 시간 순서대로 진행된다.
# turns는 ("user" 또는 "assistant", 발화문) 쌍의 목록이다.
SESSIONS = [
    {
        "id": "S1",
        "note": "기본 사실(semantic) — 이름 · 거주지 · 반려동물",
        "turns": [
            ("user", "안녕, 나는 지훈이야. 지금 서울에 살고, 고양이 한 마리 키워 — 이름은 코코."),
            ("assistant", "반가워요 지훈님! 서울에서 코코랑 지내는군요. 코코는 몇 살이에요?"),
        ],
    },
    {
        "id": "S2",
        "note": "엔티티 누적(episodic) — 코코에 얽힌 사건",
        "turns": [
            ("user", "요즘 코코가 사료를 잘 안 먹어서 걱정이야."),
            ("assistant", "저런, 코코가 입맛이 없나 봐요. 평소와 달라진 점이 있었나요?"),
        ],
    },
    {
        "id": "S3",
        "note": "모순/갱신 — 거주지가 서울에서 부산으로 바뀜",
        "turns": [
            ("user", "참, 나 지난주에 서울에서 부산으로 이사했어."),
            ("assistant", "부산으로 이사하셨군요! 새 동네는 좀 어때요?"),
        ],
    },
    {
        "id": "S4",
        "note": "무관 노이즈 — 질문과 상관없는 사실",
        "turns": [
            ("user", "요즘 파이썬 공부도 시작했어."),
            ("assistant", "좋네요, 파이썬은 입문용으로 아주 좋은 언어예요."),
        ],
    },
]

# 세대별 메모리 성능을 확인하기 위한 평가 질문이다.
# expected는 정답 판정에 필요한 키워드다.
# note는 이 질문이 어떤 학습 포인트를 보여주는지 설명한다.
PROBES = [
    {
        "question": "내 반려동물 이름이 뭐였지?",
        "expected": ["코코"],
        "note": "1세대는 재시작하면 답하지 못한다(휘발성). 3세대부터 검색으로 회복.",
    },
    {
        "question": "나 지금 어디 살아?",
        "expected": ["부산"],  # 최신 사실. '서울'은 갱신 전 과거값이다.
        "note": "4세대 모순 해결 전에는 '서울/부산'을 혼동한다 (temporal: 최신값=부산).",
    },
    {
        "question": "코코한테 요즘 무슨 일이 있었지?",
        "expected": ["사료"],
        "note": "5세대 엔티티 검색이 코코 관련 기억(사료)을 모아온다.",
    },
]


def conversation():
    """전체 대화를 시간 순서의 (role, text) 목록으로 돌려준다."""
    return [turn for session in SESSIONS for turn in session["turns"]]


def user_messages():
    """사용자 발화만 시간 순서대로 돌려준다."""
    return [text for role, text in conversation() if role == "user"]


if __name__ == "__main__":
    import sys

    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 공통 시나리오: 멀티세션 대화 ===\n")
    for session in SESSIONS:
        print(f"[{session['id']}] {session['note']}")
        for role, text in session["turns"]:
            who = "사용자  " if role == "user" else "에이전트"
            print(f"  {who}: {text}")
        print()

    print("=== 평가 질문 ===\n")
    for i, probe in enumerate(PROBES, start=1):
        print(f"  Q{i}. {probe['question']}")
        print(f"      기대 키워드: {probe['expected']}")
        print(f"      ({probe['note']})")

    print(
        f"\n총 {len(SESSIONS)}개 세션 · 사용자 발화 {len(user_messages())}개 · "
        f"평가 질문 {len(PROBES)}개."
    )
