"""
03_context_window_and_memory.py — AI Agent 메모리 0단계: "왜 메모리 관리가 필요한가"

[01·02와 무엇이 다른가]
01·02는 "기억을 어떻게 찾을까(임베딩·유사도)"를 다뤘다 — 이는 뒤에 나오는
RAG(3세대)의 토대다. 이 파일은 그보다 앞서는 질문, "애초에 왜 메모리를
'관리'해야 하는가"에 답한다. 1세대의 첫 실패가 바로 '컨텍스트 윈도우(CW)
토큰 한도 초과'인데, 그 실패가 왜·언제 일어나는지를 여기서 먼저 이해한다.

[이 파일의 목표]
  1) LLM의 max_tokens 와 모델의 컨텍스트 윈도우(CW)가 무슨 관계인지.
  2) CW 관점에서 '언제' 에이전트 메모리 관리가 필요해지는지.
  3) 실무에서 가장 먼저 쓰는 간단한 기법(슬라이딩 윈도우)을 직접 돌려보기.

[핵심 한 줄]
  컨텍스트 윈도우 = 한 번의 요청에서 모델이 볼 수 있는 토큰 총량.
  그리고 그 안에 '입력(프롬프트+누적 대화)'과 '출력(생성)'이 같이 들어가야 한다.
      입력 토큰 + 출력 토큰(max_tokens) ≤ 컨텍스트 윈도우
  대화가 길어지면 입력이 계속 불어나 이 부등식이 깨진다 → 메모리 관리가 필요.

[토큰 카운팅에 LLM을 써야 할까? — 실무 관점]
  쓰지 않는다. 토큰을 세려고 LLM을 호출하면 네트워크 지연·비용·장애점이 붙는다.
  토큰화는 결정론적 규칙이라 '로컬 토크나이저'로 오프라인·즉시·무료로 센다.
  OpenAI 계열 모델은 tiktoken이 표준이다 (Anthropic은 별도 count_tokens API).
  요청을 보내기 '전에' 길이를 알아야 자르거나 요약할 수 있으니, 로컬 카운팅은 필수다.

[준비물]
  pip install tiktoken   (requirements.txt에 포함. 없으면 거친 근사로 대체 실행됨)
"""

import sys

# gpt-4o-mini 기준 값. 모델마다 CW가 다르다(예: 일부 모델은 수십만~백만 토큰).
MODEL = "gpt-4o-mini"
CONTEXT_WINDOW = 128_000  # 이 모델이 한 요청에서 볼 수 있는 토큰 총량(입력+출력)

# 챗 형식은 메시지마다 역할(role) 등 보이지 않는 토큰이 조금 더 붙는다.
# 정확한 값은 모델/포맷마다 다르므로, 여기서는 OpenAI 문서의 근사치를 쓴다.
TOKENS_PER_MESSAGE = 3  # 메시지 1개당 대략의 부가 토큰
REPLY_PRIMING = 3       # 답변 생성을 시작할 때 붙는 대략의 토큰


# ---------------------------------------------------------------------------
# 1) 토큰 카운팅 — 로컬 토크나이저(tiktoken). LLM 호출이 아니다.
# ---------------------------------------------------------------------------
try:
    import tiktoken

    try:
        _enc = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        # tiktoken 버전이 모델명을 몰라도 인코딩 이름으로 직접 가져올 수 있다.
        _enc = tiktoken.get_encoding("o200k_base")  # gpt-4o 계열 인코딩

    def count_tokens(text: str) -> int:
        """문자열의 토큰 수를 로컬에서 정확히 센다(네트워크 호출 없음)."""
        return len(_enc.encode(text))

    _COUNT_MODE = "tiktoken(정확)"

except ModuleNotFoundError:
    # tiktoken이 없을 때를 위한 '거친 근사'. 실무에서는 쓰지 말고 tiktoken을 설치한다.
    # 한국어/영어가 섞이면 대략 글자 수와 비슷한 수준이라 4글자≈3토큰 정도로 잡는다.
    def count_tokens(text: str) -> int:
        return max(1, round(len(text) * 0.75))

    _COUNT_MODE = "근사(tiktoken 미설치 — 'pip install tiktoken' 권장)"


def count_chat_tokens(messages: list[tuple[str, str]]) -> int:
    """(role, text) 메시지 목록 전체가 입력으로 소비하는 토큰 수(부가 토큰 포함)."""
    total = REPLY_PRIMING
    for _role, text in messages:
        total += TOKENS_PER_MESSAGE + count_tokens(text)
    return total


# ---------------------------------------------------------------------------
# 2) 실무에서 가장 먼저 쓰는 기법 — 슬라이딩 윈도우
#
# "입력 토큰 예산"에 맞춰 '최근' 메시지부터 채워 넣고, 예산을 넘기면 버린다.
# (gen2의 '윈도우 + 요약'에서 이 윈도우 부분이 바로 이것이다.)
# ---------------------------------------------------------------------------

def fit_to_budget(
    messages: list[tuple[str, str]], input_budget: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """최근 메시지부터 예산 안에 들어가는 만큼만 남긴다.

    반환: (남긴 메시지[시간순], 버린 메시지[시간순])
    """
    kept_reversed: list[tuple[str, str]] = []
    running = REPLY_PRIMING
    # 최신 메시지가 가장 중요하다고 보고 '뒤에서부터' 채운다.
    for role, text in reversed(messages):
        cost = TOKENS_PER_MESSAGE + count_tokens(text)
        if running + cost > input_budget:
            break  # 더 넣으면 예산 초과 → 여기서 멈춘다(앞쪽은 모두 버려짐)
        kept_reversed.append((role, text))
        running += cost
    kept = list(reversed(kept_reversed))
    dropped = messages[: len(messages) - len(kept)]
    return kept, dropped


# ---------------------------------------------------------------------------
# 3) 시연용 대화 — scenario.py 대신, 한눈에 보이는 짧은 문자열로 직접 작성.
#
# 일부러 '초반'에 중요한 사실(땅콩 알레르기)을 심어 둔다.
# 슬라이딩 윈도우가 오래된 메시지를 버리면 이 사실이 사라지는 장면을 보기 위함.
# ---------------------------------------------------------------------------
CONVERSATION = [
    ("user", "안녕! 나 이번 주말 부산 여행 가는데 일정 좀 짜줘."),
    ("assistant", "좋아요! 1박 2일인가요, 2박 3일인가요?"),
    ("user", "1박 2일. 아 참, 나 땅콩 알레르기 있으니까 맛집 추천할 때 꼭 빼줘."),
    ("assistant", "알겠어요. 해운대 근처 숙소부터 잡을까요?"),
    ("user", "응 좋아. 첫째 날 저녁 맛집부터 추천해줘."),
    ("assistant", "좋아요, 동선과 예산에 맞춰 추천해 드릴게요."),
]


if __name__ == "__main__":
    # Windows 콘솔에서 한글/특수문자가 깨지지 않도록 UTF-8로 출력한다.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"모델: {MODEL}  |  토큰 카운팅: {_COUNT_MODE}\n")

    # === 목표 1: max_tokens 와 컨텍스트 윈도우(CW)의 관계 =====================
    print("[1] max_tokens 와 컨텍스트 윈도우(CW)는 다른 개념이다")
    print(f"  - 컨텍스트 윈도우(CW): {CONTEXT_WINDOW:,} 토큰")
    print("      = 한 요청에서 모델이 볼 수 있는 총량. '입력 + 출력'이 여기 같이 들어간다.")
    print("  - max_tokens: 내가 정하는 '출력(생성)'의 상한일 뿐. CW를 늘려주지 않는다.")
    print("  - 불변식:  입력 토큰 + max_tokens ≤ CW")
    example_input, example_max = 120_000, 16_000
    total = example_input + example_max
    print(f"  - 예) 입력 {example_input:,} + max_tokens {example_max:,} "
          f"= {total:,}  >  CW {CONTEXT_WINDOW:,}  → 요청 거부/출력 잘림")
    print("    (흔한 오해: 'max_tokens가 모델의 기억 용량'이 아니다. 출력 상한일 뿐.)\n")

    # === 목표 2: CW 관점에서 '언제' 메모리 관리가 필요한가 ====================
    print("[2] 대화가 쌓일수록 '입력' 토큰이 불어난다 (API는 매 턴 전체 history를 다시 보냄)")
    cumulative: list[tuple[str, str]] = []
    for role, text in CONVERSATION:
        cumulative.append((role, text))
        who = "user " if role == "user" else "asst "
        print(f"  {len(cumulative)}턴 후 입력 토큰: {count_chat_tokens(cumulative):>4}  "
              f"| 방금 {who}: {text}")
    full_tokens = count_chat_tokens(CONVERSATION)
    print(f"  → 전체 대화 입력 토큰: {full_tokens}")
    print("  '입력 + max_tokens'가 CW에 가까워지는 순간이 메모리 관리가 필요한 시점이다.\n")

    # === 목표 3: 가장 흔한 기법 — 슬라이딩 윈도우 ============================
    # 실제 CW는 128K라 이 짧은 대화로는 절대 안 넘친다.
    # 그래서 '한도를 일부러 작게 잡은' 가상 예산으로, 대규모 대화에서 벌어질 일을 축소 재현한다.
    DEMO_INPUT_BUDGET = 60  # 교육용으로 작게 잡은 가상 입력 예산(토큰)
    print(f"[3] 슬라이딩 윈도우: 입력 예산을 {DEMO_INPUT_BUDGET}토큰으로 가정(교육용 축소)")
    kept, dropped = fit_to_budget(CONVERSATION, DEMO_INPUT_BUDGET)
    print(f"  남긴 메시지({count_chat_tokens(kept)}토큰, 최근 {len(kept)}개):")
    for role, text in kept:
        print(f"    · [{role}] {text}")
    print(f"  버린 메시지({len(dropped)}개):")
    for role, text in dropped:
        print(f"    × [{role}] {text}")

    # === [한계] → 다음 세대를 부르는 실패 장면 ===============================
    print("\n[한계] 슬라이딩 윈도우는 단순하지만 '비가역 손실'이 있다")
    dropped_texts = " ".join(t for _r, t in dropped)
    if "알레르기" in dropped_texts:
        print("  - 방금 초반의 '땅콩 알레르기' 사실이 잘려나갔다.")
        print("    이 상태로 맛집을 추천하면 알레르기를 무시한다 — 오래됐지만 '중요한' 사실을 잃음.")
    print("  - 1세대(단기 메모리): 한도를 넘기거나 재시작하면 그냥 소실된다.")
    print("  - 2세대(윈도우+요약): 버리는 대신 '요약'으로 압축 — 하지만 요약도 디테일을 잃는다.")
    print("  - 3세대(RAG): 원본을 외부에 저장해 두고, 필요할 때만 검색해 떠올린다(01·02의 임베딩).")
