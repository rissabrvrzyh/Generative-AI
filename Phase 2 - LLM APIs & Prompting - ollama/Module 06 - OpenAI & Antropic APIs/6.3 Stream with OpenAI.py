import ollama

stream = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "Explain embeddings in 3 bullet points."
        }
    ],
    stream=True
)

for chunk in stream:
    delta = chunk["message"]["content"]

    if delta:
        print(delta, end="", flush=True)

print()