from dataclasses import dataclass, asdict
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

MODEL = "openrouter/free"


# ============================================================
# 1. Compare three code-review prompts
# ============================================================

BASIC_REVIEW = """You are a code review assistant.
Review the given code and identify obvious problems.
Give a short explanation and suggest improvements."""

INTERMEDIATE_REVIEW = """You are a software engineer performing a
code review.

For every issue you find:
1. Identify the problem.
2. Explain why it matters.
3. Suggest a practical fix.

Pay attention to correctness, readability, security,
error handling, and performance."""

EXPERT_REVIEW = """You are a senior Python engineer reviewing code
for production use.

Analyze the code systematically for:
- correctness and hidden bugs
- security vulnerabilities
- input validation
- exception and resource handling
- performance and scalability
- maintainability and API design
- concurrency problems where applicable

Do not report stylistic preferences as bugs.

For every significant issue use:
Issue:
Impact:
Evidence:
Recommended fix:

Include corrected code only when it materially improves
the solution. Prioritize concrete, actionable findings."""


CODE_SAMPLES = [
    """
def divide(a, b):
    return a / b
""",

    """
def read_user_file(filename):
    with open(filename, "r") as f:
        return f.read()
""",

    """
def get_user(user_id):
    import requests
    url = f"http://api/users/{user_id}"
    return requests.get(url).json()
""",

    """
def find_user(users, target):
    for user in users:
        if user["name"] == target:
            return user
    return None
""",

    """
def run_query(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    return database.execute(query)
"""
]


def review_code(system_prompt: str, code: str) -> str:

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Review this code:\n\n{code}"
            }
        ]
    )

    content = response.choices[0].message.content

    return content if content else "No response returned."


def evaluate_review_prompt(
    name: str,
    system_prompt: str,
    snippets: list[str]
):

    print(f"\n{'=' * 70}")
    print(f"{name.upper()} REVIEW")
    print(f"{'=' * 70}")

    outputs = []

    for number, snippet in enumerate(snippets, start=1):

        result = review_code(
            system_prompt,
            snippet
        )

        outputs.append(result)

        print(f"\n--- Code Snippet {number} ---")
        print(result[:1200])

    return outputs


print("\n### EXERCISE 1 ###")

basic_results = evaluate_review_prompt(
    "Basic",
    BASIC_REVIEW,
    CODE_SAMPLES
)

intermediate_results = evaluate_review_prompt(
    "Intermediate",
    INTERMEDIATE_REVIEW,
    CODE_SAMPLES
)

expert_results = evaluate_review_prompt(
    "Expert",
    EXPERT_REVIEW,
    CODE_SAMPLES
)


print("\n\nPROMPT QUALITY COMPARISON")
print("-" * 70)
print("Basic       : focuses mainly on obvious issues.")
print("Intermediate: adds structured issue analysis and practical fixes.")
print("Expert      : checks security, correctness, performance,")
print("              maintainability, and production-level concerns.")


# ============================================================
# 2. PromptLibrary
# ============================================================

@dataclass
class PromptTemplate:

    name: str
    version: int
    template: str
    last_evaluation: str | None = None


class PromptLibrary:

    def __init__(self):
        self.templates = {}

    def add(
        self,
        name: str,
        template: str,
        version: int = 1
    ):

        self.templates[name] = PromptTemplate(
            name=name,
            version=version,
            template=template
        )

    def get(self, name: str) -> PromptTemplate:

        if name not in self.templates:
            raise KeyError(
                f"Prompt '{name}' does not exist."
            )

        return self.templates[name]

    def record_evaluation(
        self,
        name: str,
        evaluation_id: str
    ):

        self.get(name).last_evaluation = evaluation_id

    def save(self, filename: str):

        data = {
            name: asdict(prompt)
            for name, prompt in self.templates.items()
        }

        Path(filename).write_text(
            json.dumps(
                data,
                indent=2
            ),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, filename: str):

        library = cls()

        raw = json.loads(
            Path(filename).read_text(
                encoding="utf-8"
            )
        )

        for name, values in raw.items():

            library.templates[name] = PromptTemplate(
                **values
            )

        return library

    def show(self):

        for prompt in self.templates.values():

            print(
                f"{prompt.name} "
                f"v{prompt.version} "
                f"| last evaluation: "
                f"{prompt.last_evaluation}"
            )


print("\n\n### EXERCISE 2 ###")

library = PromptLibrary()

library.add(
    "basic_review",
    BASIC_REVIEW,
    version=1
)

library.add(
    "intermediate_review",
    INTERMEDIATE_REVIEW,
    version=1
)

library.add(
    "expert_review",
    EXPERT_REVIEW,
    version=1
)

library.record_evaluation(
    "basic_review",
    "eval-001"
)

library.record_evaluation(
    "intermediate_review",
    "eval-001"
)

library.record_evaluation(
    "expert_review",
    "eval-001"
)

library.save("prompt_library.json")

print("Saved prompt library:")
library.show()

loaded_library = PromptLibrary.load(
    "prompt_library.json"
)

print("\nLoaded prompt library:")
loaded_library.show()


# ============================================================
# 3. Automatic JSON repair
# ============================================================

def remove_code_fence(text: str) -> str:

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    return cleaned.strip()


def repair_json_with_model(text: str) -> dict:

    repair_prompt = f"""Repair the following invalid JSON.

Return ONLY valid JSON.
Do not use markdown.
Do not explain anything.

Invalid JSON:
{text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": repair_prompt
            }
        ]
    )

    repaired = response.choices[0].message.content

    if not repaired:
        raise ValueError(
            "Model did not return repaired JSON."
        )

    repaired = remove_code_fence(repaired)

    return json.loads(repaired)


def safe_json_parse(text: str) -> dict:

    # Attempt 1: direct parsing
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # Attempt 2: remove markdown fences
    cleaned = remove_code_fence(text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    # Attempt 3: ask the model to repair it
    return repair_json_with_model(cleaned)


print("\n\n### EXERCISE 3 ###")

json_examples = [

    '{"name": "Alice", "score": 95}',

    '''```json
    {"name": "Bob", "score": 88}
    ```''',

    '{"name": "Charlie", "score": 91,}'
]

for sample in json_examples:

    try:

        parsed = safe_json_parse(sample)

        print("\nOriginal:")
        print(sample)

        print("Parsed:")
        print(parsed)

    except Exception as error:

        print(
            f"\nJSON repair failed: {error}"
        )


# ============================================================
# 4. CoT-style LLM evaluation ranking
# ============================================================

RANKING_SYSTEM = """You are evaluating several language models.

Given evaluation scores for 10 model-task results across
three tasks:

1. Calculate the average score for each model.
2. Compare the models using the same calculation for every input.
3. Rank models from highest average score to lowest.
4. Write exactly two sentences recommending a model based
   on the calculated results.

Return the final result using this format:

Ranking:
1. MODEL - average score
2. MODEL - average score
...

Recommendation:
Sentence one.
Sentence two.

Show the calculations briefly before the final ranking.
Do not change or invent any scores."""


EVALUATION_DATA = [

    """
Model Alpha:
Task A = 90
Task B = 85
Task C = 88

Model Beta:
Task A = 84
Task B = 92
Task C = 87

Model Gamma:
Task A = 88
Task B = 86
Task C = 91
""",

    """
Model Alpha:
Task A = 70
Task B = 95
Task C = 82

Model Beta:
Task A = 88
Task B = 84
Task C = 90

Model Gamma:
Task A = 92
Task B = 80
Task C = 86
""",

    """
Model Alpha:
Task A = 96
Task B = 91
Task C = 94

Model Beta:
Task A = 90
Task B = 93
Task C = 92

Model Gamma:
Task A = 85
Task B = 97
Task C = 89
"""
]


def rank_models(score_text: str) -> str:

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=700,
        messages=[
            {
                "role": "system",
                "content": RANKING_SYSTEM
            },
            {
                "role": "user",
                "content": score_text
            }
        ]
    )

    result = response.choices[0].message.content

    return result if result else "No response returned."


print("\n\n### EXERCISE 4 ###")

for index, dataset in enumerate(
    EVALUATION_DATA,
    start=1
):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"Evaluation Input {index}"
    )

    print(
        "=" * 70
    )

    print(
        rank_models(dataset)
    )