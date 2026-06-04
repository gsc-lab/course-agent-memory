"""
scoring.py — probe 답변을 자동 채점하는 공용 채점기

[왜 채점기가 필요한가]
각 세대가 '기억을 잘 하는지'를 말로만 비교하면 주관적이다. 재시작 후 같은
질문(probe)을 던지고 **답이 맞았는지 기계가 판정**해야, 세대별 'probe 정답률'로
객관적인 비교가 된다. 이 정답률이 과정 피날레(평가 단계)의 핵심 잣대다.

[두 가지 채점 방식]
  - keyword(기본): 답에 기대 키워드가 들어있나 문자열 검사. 무료·즉시·결정적.
                   세대 데모는 단순한 probe라 이걸로 충분하다.
  - llm(선택)     : "이 답이 기대정답과 같은 뜻인가?"를 LLM에 물음. 표현이 달라도
                   잡아내지만 API 비용·비결정성이 있다. capstone 평가에서만 켠다.

scenario.py가 '정답(expected 키워드)'을 들고 있고, 이 파일은 '판정 로직'만 맡는다.
"""

# capstone에서 llm 채점을 켤 때 쓰는 모델(저렴한 chat 모델). 필요 시 교체.
LLM_JUDGE_MODEL = "gpt-4o-mini"


def keyword_match(answer: str, expected: list[str]) -> bool:
    """answer에 expected 키워드가 하나라도 (대소문자 무시) 들어가면 정답."""
    low = answer.lower()
    return any(kw.lower() in low for kw in expected)


def llm_judge(question: str, expected: list[str], answer: str) -> bool:
    """LLM이 answer의 정답 여부를 판정한다(capstone 평가용, openai 필요).

    keyword 방식이 표현 차이에 약할 때 쓴다. 호출이 일어나므로 기본값은 아니다.
    """
    from openai import OpenAI  # 지연 import — keyword 모드는 openai 없이도 동작

    client = OpenAI()
    prompt = (
        "다음 답변이 질문에 대해 사실상 올바른지 판단해줘.\n"
        f"질문: {question}\n"
        f"정답에 해당하는 핵심: {', '.join(expected)}\n"
        f"답변: {answer}\n\n"
        "올바르면 'YES', 아니면 'NO'만 출력해."
    )
    resp = client.chat.completions.create(
        model=LLM_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("Y")


def grade(probes: list[dict], answers: list[str], mode: str = "keyword") -> list[dict]:
    """probes와 answers(같은 길이·순서)를 채점해 결과 딕셔너리 리스트로 돌려준다."""
    if len(probes) != len(answers):
        raise ValueError("probes와 answers의 길이가 다릅니다.")

    results = []
    for probe, answer in zip(probes, answers):
        if mode == "keyword":
            passed = keyword_match(answer, probe["expected"])
        elif mode == "llm":
            passed = llm_judge(probe["question"], probe["expected"], answer)
        else:
            raise ValueError(f"알 수 없는 채점 모드: {mode!r} (keyword|llm)")
        results.append(
            {
                "question": probe["question"],
                "expected": probe["expected"],
                "answer": answer,
                "passed": passed,
            }
        )
    return results


def accuracy(results: list[dict]) -> float:
    """probe 정답률(0.0~1.0)."""
    if not results:
        return 0.0
    return sum(r["passed"] for r in results) / len(results)


def print_report(title: str, results: list[dict]) -> None:
    """채점 결과를 사람이 보기 좋게 출력한다."""
    print(f"[{title}]")
    for r in results:
        mark = "O" if r["passed"] else "X"
        print(f"  {mark}  Q: {r['question']}")
        print(f"      A: {r['answer']}")
    print(f"  → probe 정답률: {accuracy(results):.0%} "
          f"({sum(r['passed'] for r in results)}/{len(results)})\n")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # keyword 채점 자체 점검(데모): 정답/오답 한 개씩.
    demo_probes = [
        {"question": "반려동물 이름?", "expected": ["코코"]},
        {"question": "어디 살아?", "expected": ["부산"]},
    ]
    demo_answers = ["당신의 고양이 이름은 코코예요.", "잘 모르겠어요."]
    results = grade(demo_probes, demo_answers, mode="keyword")
    print_report("scoring 자체 점검 (keyword)", results)
