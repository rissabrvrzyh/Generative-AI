from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)


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


class OpenRouterClient(BaseLLMClient):

    def __init__(self, model: str = "openrouter/free"):
        self.model = model
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
        )

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
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )

        return ChatResponse(
            text=resp.choices[0].message.content,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            model=self.model,
        )


# Usage - same code, different backend:
client: BaseLLMClient = OpenRouterClient()

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
    f"Cost estimate: {result.input_tokens} in, "
    f"{result.output_tokens} out"
)