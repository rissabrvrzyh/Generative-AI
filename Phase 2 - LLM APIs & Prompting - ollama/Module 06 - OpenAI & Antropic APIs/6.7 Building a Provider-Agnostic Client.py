from abc import ABC, abstractmethod
from dataclasses import dataclass
import ollama


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class BaseLLMClient(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        ...


class OllamaClient(BaseLLMClient):

    def __init__(self, model: str = "qwen3:8b"):
        self.model = model

    def chat(
        self,
        messages,
        system="",
        max_tokens=1024,
        temperature=0.7
    ) -> ChatResponse:

        api_messages = []

        if system:
            api_messages.append({
                "role": "system",
                "content": system
            })

        api_messages += [
            {
                "role": m.role,
                "content": m.content
            }
            for m in messages
        ]

        resp = ollama.chat(
            model=self.model,
            messages=api_messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        )

        return ChatResponse(
            text=resp["message"]["content"],
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
            model=self.model,
        )


# Usage
client: BaseLLMClient = OllamaClient()

msgs = [
    ChatMessage(
        role="user",
        content="What is a vector database?"
    )
]

result = client.chat(
    msgs,
    system="Be concise."
)

print(result.text)
print(
    f"Tokens: "
    f"{result.input_tokens} in, "
    f"{result.output_tokens} out"
)