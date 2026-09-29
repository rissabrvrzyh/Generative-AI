import ollama
import json
import re

SYSTEM = """You are a data extractor.

Extract information from the given text and return ONLY a JSON object.

Schema:
{
    "company": "string",
    "founded": integer or null,
    "products": ["string"],
    "headquarters": "string or null",
    "is_public": boolean
}

Rules:
- Return ONLY valid JSON.
- No markdown.
- No explanation.
- No code fences.
"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]


def extract_company_info(text: str) -> dict:

    resp = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM
            },
            {
                "role": "user",
                "content": text
            }
        ],
        think=False,
        options={
            "temperature": 0,
            "num_predict": 512
        }
    )

    raw = resp["message"]["content"].strip()

    print("Raw response:")
    print(raw)
    print()

    # Remove markdown code fences if present
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    # Find JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    if not match:
        raise ValueError(
            f"Qwen tidak menghasilkan JSON yang valid.\n"
            f"Raw output: {raw!r}"
        )

    json_text = match.group(0)

    return json.loads(json_text)


for text in texts:
    info = extract_company_info(text)

    print("Parsed result:")
    print(json.dumps(info, indent=2))
    print("=" * 60)