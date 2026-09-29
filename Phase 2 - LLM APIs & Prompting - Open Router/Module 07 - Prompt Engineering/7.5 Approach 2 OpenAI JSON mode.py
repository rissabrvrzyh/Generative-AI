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

response = client.chat.completions.create(
    model="openrouter/free",
    response_format={"type": "json_object"},  # enforces valid JSON
    messages=[
        {
            "role": "system",
            "content": """Extract entities. Return JSON with
this schema:
{"people": [string], "organizations": [string], "locations":
[string]}"""
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
        }
    ]
)

result = json.loads(response.choices[0].message.content)

print(result)

# {'people': ['Elon Musk'], 'organizations': ['SpaceX', 'Tesla'], 'locations': ['Hawthorne, California']}