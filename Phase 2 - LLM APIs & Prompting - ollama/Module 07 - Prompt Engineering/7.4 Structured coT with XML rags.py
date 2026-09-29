import ollama
import re

SYSTEM = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

resp = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": SYSTEM
        },
        {
            "role": "user",
            "content": "A RAG pipeline retrieves 5 documents, each 400 tokens. The query is 50 tokens. The model has a 4096 token limit for context. How many tokens remain for the response?"
        }
    ],
    options={
        "temperature": 0,
        "num_predict": 512
    }
)

text = resp["message"]["content"]

# Extract sections
thinking = re.search(
    r"<thinking>(.*?)</thinking>",
    text,
    re.DOTALL
)

answer = re.search(
    r"<answer>(.*?)</answer>",
    text,
    re.DOTALL
)

print(
    "Reasoning:",
    thinking.group(1).strip() if thinking else "not found"
)

print(
    "Answer:",
    answer.group(1).strip() if answer else "not found"
)