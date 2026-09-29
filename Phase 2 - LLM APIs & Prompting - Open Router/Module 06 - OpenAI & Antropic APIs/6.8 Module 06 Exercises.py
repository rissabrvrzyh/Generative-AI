import asyncio
import os
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI, RateLimitError


# ============================================================
# Configuration
# ============================================================

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

API_KEY = os.environ["OPENROUTER_API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


# ============================================================
# 1. Retry on Rate Limit
# ============================================================

class MockRateLimitError(Exception):
    """
    Custom exception used only for testing the retry mechanism.
    """
    pass


def retry_on_rate_limit(
    client,
    messages,
    max_retries=5
):
    """
    Send a request and retry when a rate-limit error occurs.
    Uses exponential backoff between attempts.
    """

    for attempt in range(max_retries + 1):

        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                max_tokens=512,
                messages=messages
            )

            return response

        except (RateLimitError, MockRateLimitError) as error:

            if attempt == max_retries:
                print("Maximum retry attempts reached.")
                raise error

            delay = (2 ** attempt) + random.uniform(0, 0.5)

            print(
                f"Rate limit detected. "
                f"Retry {attempt + 1}/{max_retries} "
                f"after {delay:.2f} seconds..."
            )

            time.sleep(delay)


# ----------------------------
# Mock client for testing
# ----------------------------

class MockCompletions:

    def __init__(self, failures=2):
        self.calls = 0
        self.failures = failures

    def create(self, **kwargs):

        self.calls += 1

        if self.calls <= self.failures:
            raise MockRateLimitError(
                "Simulated 429 rate limit error"
            )

        return {
            "status": "success",
            "message": "Request succeeded after retry."
        }


class MockChat:

    def __init__(self, failures=2):
        self.completions = MockCompletions(failures)


class MockClient:

    def __init__(self, failures=2):
        self.chat = MockChat(failures)


print("\n=== Exercise 1: Retry Test ===")

mock_client = MockClient(failures=2)

try:

    mock_result = retry_on_rate_limit(
        mock_client,
        [
            {
                "role": "user",
                "content": "Test retry mechanism."
            }
        ],
        max_retries=5
    )

    print("Mock retry test passed.")
    print(mock_result)

except (RateLimitError, MockRateLimitError) as error:

    print(f"Retry test failed: {error}")


# ============================================================
# 2. Token Budget Manager
# ============================================================

class BudgetExceeded(Exception):
    """
    Raised when the token budget has been exceeded.
    """
    pass


class TokenBudgetManager:

    def __init__(self, maximum_tokens: int):
        self.maximum_tokens = maximum_tokens
        self.used_tokens = 0

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int
    ):

        total_for_request = (
            input_tokens + output_tokens
        )

        new_total = (
            self.used_tokens + total_for_request
        )

        if new_total > self.maximum_tokens:

            raise BudgetExceeded(
                f"Token budget exceeded: "
                f"{new_total}/{self.maximum_tokens}"
            )

        self.used_tokens = new_total

    def remaining(self) -> int:
        return self.maximum_tokens - self.used_tokens

    def reset(self):
        self.used_tokens = 0

    def __str__(self):
        return (
            f"Token usage: "
            f"{self.used_tokens}/{self.maximum_tokens}"
        )


print("\n=== Exercise 2: Token Budget ===")

budget = TokenBudgetManager(
    maximum_tokens=1000
)

try:

    budget.add_usage(
        input_tokens=300,
        output_tokens=200
    )

    print(budget)

    budget.add_usage(
        input_tokens=250,
        output_tokens=150
    )

    print(budget)

    print(
        f"Remaining tokens: "
        f"{budget.remaining()}"
    )

except BudgetExceeded as error:

    print(error)


# ============================================================
# 3. Compare Models Concurrently
# ============================================================

async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


async def run_model(
    prompt: str,
    model_name: str
) -> dict:

    start_time = time.perf_counter()

    try:

        response = await async_client.chat.completions.create(
            model=model_name,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "model": model_name,
            "response_text": (
                response.choices[0].message.content
            ),
            "input_tokens": (
                response.usage.prompt_tokens
            ),
            "output_tokens": (
                response.usage.completion_tokens
            ),
            "latency_ms": round(elapsed_ms, 2)
        }

    except Exception as error:

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "model": model_name,
            "response_text": f"ERROR: {error}",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round(elapsed_ms, 2)
        }


async def compare_models_async(
    prompt: str,
    models: list[str]
) -> pd.DataFrame:

    tasks = [
        run_model(prompt, model)
        for model in models
    ]

    results = await asyncio.gather(*tasks)

    return pd.DataFrame(
        results,
        columns=[
            "model",
            "response_text",
            "input_tokens",
            "output_tokens",
            "latency_ms"
        ]
    )


def compare_models(
    prompt: str,
    models: list[str]
) -> pd.DataFrame:

    return asyncio.run(
        compare_models_async(
            prompt,
            models
        )
    )


print("\n=== Exercise 3: Model Comparison ===")

models_to_test = [
    "openrouter/free"
]

comparison_result = compare_models(
    "Explain RAG in two sentences.",
    models_to_test
)

print(
    comparison_result.to_string(
        index=False
    )
)


# ============================================================
# 4. Stream Response to File
# ============================================================

def stream_to_file(
    prompt: str,
    output_path: str
):

    stream = client.chat.completions.create(
        model="openrouter/free",
        max_tokens=512,
        stream=True,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output_file = Path(output_path)

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in stream:

            if not chunk.choices:
                continue

            text = (
                chunk.choices[0]
                .delta
                .content
            )

            if text:

                file.write(text)
                file.flush()

                print(
                    text,
                    end="",
                    flush=True
                )

    print(
        f"\n\nResponse saved to: "
        f"{output_file}"
    )


print("\n=== Exercise 4: Streaming ===")

stream_to_file(
    "Explain how retrieval augmented generation works.",
    "stream_output.txt"
)