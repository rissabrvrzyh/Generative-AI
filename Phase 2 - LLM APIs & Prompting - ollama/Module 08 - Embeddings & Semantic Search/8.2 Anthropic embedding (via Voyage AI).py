import ollama
import numpy as np


EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> list[float]:
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text
    )

    return response["embeddings"][0]


texts = [
    "Python is a programming language.",
    "Python is commonly used for software development.",
    "Cats are popular household pets.",
]


for text in texts:
    embedding = get_embedding(text)

    print(f"Text: {text}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print()