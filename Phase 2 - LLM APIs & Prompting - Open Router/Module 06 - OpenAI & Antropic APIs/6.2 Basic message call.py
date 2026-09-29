import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

message = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "What is retrieval-augmented generation?"
        }
    ]
)

# Response text
print(message.choices[0].message.content)

# Usage stats
print(f"Input tokens: {message.usage.prompt_tokens}")
print(f"Output tokens: {message.usage.completion_tokens}")