import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key dari file .env
load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

# Create OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# Send message
message = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=512,
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical writer. Answer in plain English, no jargon."
        },
        {
            "role": "user",
            "content": "Explain what a vector database does."
        }
    ]
)

# Print response
print(message.choices[0].message.content)

# Print token usage
print(f"Input tokens: {message.usage.prompt_tokens}")
print(f"Output tokens: {message.usage.completion_tokens}")