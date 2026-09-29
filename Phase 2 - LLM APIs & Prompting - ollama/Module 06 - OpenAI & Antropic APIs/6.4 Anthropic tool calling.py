import ollama
import json


# 1. Define the tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window size and cost per 1K tokens for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model identifier, e.g. 'qwen3:8b' or 'gpt-4o'."
                    }
                },
                "required": ["model_name"]
            }
        }
    }
]


# 2. The actual function the tool will call
def get_model_info(model_name: str) -> dict:
    db = {
        "qwen3:8b": {
            "context_k": "Check Ollama/model documentation",
            "cost_input": 0,
            "cost_output": 0
        },
        "gpt-4o": {
            "context_k": 128,
            "cost_input": 2.50,
            "cost_output": 10.00
        },
        "gemini-1.5-pro": {
            "context_k": 1000,
            "cost_input": 1.25,
            "cost_output": 5.00
        },
    }

    return db.get(
        model_name,
        {"error": f"Unknown model: {model_name}"}
    )


# 3. First API call
messages = [
    {
        "role": "user",
        "content": "How large is the context window of qwen3:8b?"
    }
]

response = ollama.chat(
    model="qwen3:8b",
    messages=messages,
    tools=tools
)


# 4. Check if model wants to use a tool
if response["message"].get("tool_calls"):

    for tool_call in response["message"]["tool_calls"]:

        tool_name = tool_call["function"]["name"]
        tool_input = tool_call["function"]["arguments"]

        print(f"Tool called: {tool_name}({tool_input})")

        # 5. Execute the function
        result = get_model_info(**tool_input)

        print(f"Tool result: {result}")

        # Add assistant's tool call to conversation
        messages.append(response["message"])

        # 6. Send tool result back to the model
        messages.append({
            "role": "tool",
            "content": json.dumps(result)
        })

    final = ollama.chat(
        model="qwen3:8b",
        messages=messages,
        tools=tools
    )

    print("\nFinal answer:")
    print(final["message"]["content"])

else:
    print(response["message"]["content"])