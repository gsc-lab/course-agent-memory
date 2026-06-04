"""
scoring.py — 평가 질문 답변을 채점하는 공용 채점기

[왜 채점기가 필요한가]
각 세대가 "기억을 잘 하는지"를 말로만 비교하면 주관적이다.
같은 질문을 던지고 답이 맞았는지 같은 기준으로 판정해야,
세대별 정답률을 비교할 수 있다.

[두 가지 채점 방식]
  - keyword(기본): 답에 기대 키워드가 들어 있는지 문자열로 확인한다.
                   빠르고 비용이 없어서 기본 데모에 적합하다.
  - llm(선택)     : 답이 기대 정답과 같은 뜻인지 LLM에게 판정시킨다.
                   표현이 달라도 잡아낼 수 있지만 API 비용이 든다.

scenario.py가 정답 키워드를 들고 있고, 이 파일은 판정 로직만 맡는다.
"""

# LLM 채점을 켤 때 쓰는 기본 모델. 필요하면 바꿀 수 있다.
LLM_JUDGE_MODEL = "gpt-4o-mini"


def keyword_match(answer: str, expected: list[str]) -> bool:
    """답변에 기대 키워드가 하나라도 들어 있으면 정답으로 본다."""
    low = answer.lower()
    return any(kw.lower() in low for kw in expected)


def llm_judge(question: str, expected: list[str], answer: str) -> bool:
    """LLM이 답변의 정답 여부를 판정한다."""
    from openai import OpenAI  # keyword 모드에서는 openai가 필요 없도록 여기서 import한다.

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
    """평가 질문과 답변을 같은 순서로 받아 채점한다."""
    if len(probes) != len(answers):
        raise ValueError("평가 질문과 답변의 길이가 다릅니다.")

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
    """정답률(0.0~1.0)."""
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
    print(f"  → 정답률: {accuracy(results):.0%} "
          f"({sum(r['passed'] for r in results)}/{len(results)})\n")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # keyword 채점이 어떻게 동작하는지 보여주는 간단한 점검 예시.
    demo_probes = [
        {"question": "반려동물 이름?", "expected": ["코코"]},
        {"question": "어디 살아?", "expected": ["부산"]},
    ]
    demo_answers = ["당신의 고양이 이름은 코코예요.", "잘 모르겠어요."]
    results = grade(demo_probes, demo_answers, mode="keyword")
    print_report("scoring 자체 점검 (keyword)", results)
