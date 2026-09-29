import ollama


# Count tokens before sending
messages = [
    {
        "role": "system",
        "content": "You are a concise assistant."
    },
    {
        "role": "user",
        "content": "Explain the transformer architecture."
    }
]

response = ollama.chat(
    model="qwen3:8b",
    messages=messages
)

print(f"Input tokens: {response['prompt_eval_count']}")
print(f"Output tokens: {response['eval_count']}")

input_tokens = response["prompt_eval_count"]
output_tokens = response["eval_count"]


# Cost estimator
# Qwen is running locally, so API cost = $0
def estimate_cost(
    input_tokens: int,
    output_tokens: int
) -> float:
    """Return estimated cost in USD."""
    return 0.0


cost = estimate_cost(
    input_tokens=input_tokens,
    output_tokens=output_tokens
)

print(f"Estimated cost: ${cost:.6f}")


# Context window check
# Set this according to the model configuration you are using.
CONTEXT_LIMIT = 32768


def fits_in_context(
    token_count: int,
    reserve_for_output: int = 2048
) -> bool:
    return token_count + reserve_for_output <= CONTEXT_LIMIT


print(
    f"Fits in context: "
    f"{fits_in_context(input_tokens)}"
)