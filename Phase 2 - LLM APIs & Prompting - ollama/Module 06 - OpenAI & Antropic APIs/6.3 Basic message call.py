import ollama

response = ollama.chat(
    model="qwen3:8b",
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

print(response["message"]["content"])