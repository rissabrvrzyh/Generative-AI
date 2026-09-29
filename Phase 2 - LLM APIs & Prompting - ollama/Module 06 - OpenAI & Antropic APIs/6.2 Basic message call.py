import ollama

message = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "What is retrieval-augmented generation?"
        }
    ]
)

# Response text
print(message["message"]["content"])

# Usage stats
print(f"Input tokens: {message['prompt_eval_count']}")
print(f"Output tokens: {message['eval_count']}")