from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

stream = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=512,
    stream=True,
    messages=[
        {
            "role": "user",
            "content": "Explain embeddings in 3 bullet points."
        }
    ]
)

for chunk in stream:
    delta = chunk.choices[0].delta.content

    if delta:
        print(delta, end="", flush=True)

print()