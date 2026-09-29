import ollama

response = ollama.chat(
    model="qwen3:8b",
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

print(response["message"]["content"])

# Usage stats
print(f"Input tokens: {response['prompt_eval_count']}")
print(f"Output tokens: {response['eval_count']}")
print(
    f"Total tokens: "
    f"{response['prompt_eval_count'] + response['eval_count']}"
)