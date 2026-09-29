import ollama
import json

response = ollama.chat(
    model="qwen3:8b",
    format="json",
    think=False,
    messages=[
        {
            "role": "system",
            "content": """Extract entities.

Return JSON with this schema:
{
    "people": ["string"],
    "organizations": ["string"],
    "locations": ["string"]
}"""
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
        }
    ],
    options={
        "temperature": 0
    }
)

result = json.loads(response["message"]["content"])

print(result)