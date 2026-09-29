from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# Count tokens before sending
# OpenRouter does not provide Anthropic's count_tokens endpoint,
# so we use the actual token usage returned by the API.

response = client.chat.completions.create(
    model="openrouter/free",
    max_tokens=512,
    messages=[
        {
            "role": "system",
            "content": "You are a concise assistant."
        },
        {
            "role": "user",
            "content": "Explain the transformer architecture."
        }
    ]
)

input_tokens = response.usage.prompt_tokens
output_tokens = response.usage.completion_tokens

print(f"Input tokens used: {input_tokens}")
print(f"Output tokens used: {output_tokens}")
print(f"Total tokens used: {response.usage.total_tokens}")


# Cost estimator
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    # per 1M tokens
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")

    p = PRICING[model]

    return (
        input_tokens * p["input"] +
        output_tokens * p["output"]
    ) / 1_000_000


cost = estimate_cost(
    "gpt-4o",
    input_tokens=input_tokens,
    output_tokens=output_tokens
)

print(f"Estimated cost: ${cost:.6f}")


# Context window limits
CONTEXT_LIMITS = {
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-1.5-pro": 1_000_000,
}


def fits_in_context(
    model: str,
    token_count: int,
    reserve_for_output: int = 2048
) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit


# Example context check
fits = fits_in_context(
    "gpt-4o",
    input_tokens
)

print(f"Fits in context: {fits}")