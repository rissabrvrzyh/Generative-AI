from dataclasses import dataclass
import ollama


@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]  # at least one must appear in response
    must_be_json: bool = False


def evaluate_prompt(system: str, cases: list[EvalCase]) -> dict:
    """Run a prompt against test cases and return pass rate + details."""
    results = []

    for case in cases:
        resp = ollama.chat(
            model="qwen3:8b",
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": case.input_text
                }
            ],
            think=False,
            options={
                "temperature": 0,
                "num_predict": 256
            }
        )

        text = resp["message"]["content"].strip()

        # Check keyword hit
        keyword_hit = any(
            kw.lower() in text.lower()
            for kw in case.expected_keywords
        )

        # Check JSON validity if required
        json_valid = True

        if case.must_be_json:
            try:
                import json
                json.loads(text)
            except json.JSONDecodeError:
                json_valid = False

        passed = keyword_hit and json_valid

        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response_preview": text[:80],
        })

    pass_rate = sum(
        r["passed"] for r in results
    ) / len(results)

    return {
        "pass_rate": pass_rate,
        "results": results
    }


# Test a classification prompt
CLASSIFY_SYSTEM = """Classify the AI task as one of:
CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.

Return ONLY the category word.
Do not provide explanations."""


test_cases = [
    EvalCase(
        "Predict whether an email is spam.",
        ["CLASSIFICATION"]
    ),
    EvalCase(
        "Write a product description for headphones.",
        ["GENERATION"]
    ),
    EvalCase(
        "Find the most relevant documents for a query.",
        ["RETRIEVAL"]
    ),
    EvalCase(
        "Convert this sentence to a vector.",
        ["EMBEDDING"]
    ),
    EvalCase(
        "Label customer reviews as positive or negative.",
        ["CLASSIFICATION"]
    ),
]


report = evaluate_prompt(CLASSIFY_SYSTEM, test_cases)

print(f"Pass rate: {report['pass_rate']:.0%}")

for r in report["results"]:
    status = "PASS" if r["passed"] else "FAIL"

    print(
        f" [{status}] {r['input']!r} → {r['response_preview']!r}"
    )