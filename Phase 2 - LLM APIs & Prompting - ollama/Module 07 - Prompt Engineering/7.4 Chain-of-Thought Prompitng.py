import ollama

COT_SYSTEM = """You are a careful reasoning assistant.

Solve problems step by step before giving the final answer.
Check your calculations and reasoning carefully.

For each problem:
1. Identify the important information.
2. Work through the problem step by step.
3. Verify the result.
4. Give the final answer clearly.

Keep the reasoning concise and focused."""

test_inputs = [
    "A store gives a 20% discount on a $50 item. What is the final price?",
    "If a train travels 120 km in 2 hours, what is its average speed?",
    "A student has 80 points and gains 15 points, then loses 10 points. What is the final score?"
]

for text in test_inputs:
    resp = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": COT_SYSTEM
            },
            {
                "role": "user",
                "content": text
            }
        ],
        options={
            "temperature": 0
        }
    )

    print(f"Question: {text}")
    print(f"Answer:\n{resp['message']['content']}\n")
    print("-" * 60)