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

response = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=1024,
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical assistant."
        },
        {
            "role": "user",
            "content": "What is the difference between RAG and fine-tuning?"
        }
    ]
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")