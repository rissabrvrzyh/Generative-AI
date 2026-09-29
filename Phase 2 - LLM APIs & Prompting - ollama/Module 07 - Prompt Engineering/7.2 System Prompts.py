import ollama

# Weak system prompt
weak_prompt = "You are an AI assistant."

# Strong system prompt
strong_prompt = """
You are a senior Python engineer reviewing
code for a production AI pipeline.

Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters

Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.

Format:
Return your review as a numbered list.
Each item: Issue → Impact → Fix.
"""

user_message = """
Review this Python code:

def calculate_average(numbers):
    total = 0

    for number in numbers:
        total += number

    return total / len(numbers)
"""

# =========================
# Weak Prompt
# =========================
response_weak = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": weak_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
)

print("===== WEAK SYSTEM PROMPT =====")
print(response_weak["message"]["content"])


# =========================
# Strong Prompt
# =========================
response_strong = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": strong_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
)

print("\n===== STRONG SYSTEM PROMPT =====")
print(response_strong["message"]["content"])