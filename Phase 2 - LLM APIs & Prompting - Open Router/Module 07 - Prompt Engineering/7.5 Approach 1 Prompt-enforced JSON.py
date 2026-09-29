from openai import OpenAI
import os, json
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

SYSTEM = """You are a data extractor. Extract information and
return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.
Schema:
{
"company": string,
"founded": integer or null,
"products": [string],
"headquarters": string or null,
"is_public": boolean
}"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]


def extract_company_info(text: str) -> dict:

    resp = client.chat.completions.create(
        model="openrouter/free",
        max_tokens=256,
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
    )

    message = resp.choices[0].message

    # Check whether the model returned text
    if message.content is None:
        print("Model did not return text.")
        print("Finish reason:", resp.choices[0].finish_reason)
        print("Full response:")
        print(resp)
        return {}

    raw = message.content.strip()

    # Strip any accidental markdown fences
    raw = (
        raw.removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )

    return json.loads(raw)


for text in texts:

    info = extract_company_info(text)

    if info:
        print(json.dumps(info, indent=2))
        print()