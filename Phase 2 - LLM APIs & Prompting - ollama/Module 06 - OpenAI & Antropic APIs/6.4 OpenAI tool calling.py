import ollama
import json


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model identifier."
                    }
                },
                "required": ["model_name"]
            }
        }
    }
]


def get_model_info(model_name: str) -> dict:
    db = {
        "qwen3:8b": {
            "context_k": "Local model",
            "cost_input": 0
        },
        "gpt-4o": {
            "context_k": 128,
            "cost_input": 2.50
        },
        "claude-sonnet-4-5": {
            "context_k": 200,
            "cost_input": 3.00
        },
    }

    return db.get(model_name, {"error": "unknown model"})


messages = [
    {
        "role": "user",
        "content": "What is qwen3:8b's context window?"
    }
]


# First request
response = ollama.chat(
    model="qwen3:8b",
    tools=tools,
    messages=messages
)


# Check whether Qwen wants to call a tool
if response["message"].get("tool_calls"):

    tool_call = response["message"]["tool_calls"][0]

    name = tool_call["function"]["name"]
    args = tool_call["function"]["arguments"]

    print(f"Tool called: {name}({args})")

    # Execute the function
    result = get_model_info(**args)

    print(f"Tool result: {result}")

    # Append assistant message
    messages.append(response["message"])

    # Append tool result
    messages.append({
        "role": "tool",
        "content": json.dumps(result)
    })

    # Second request
    final = ollama.chat(
        model="qwen3:8b",
        messages=messages,
        tools=tools
    )

    print("\nFinal answer:")
    print(final["message"]["content"])

else:
    print(response["message"]["content"])