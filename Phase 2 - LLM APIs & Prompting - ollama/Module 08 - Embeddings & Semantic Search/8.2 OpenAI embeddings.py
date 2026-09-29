import ollama
import numpy as np


def embed(
    texts: list[str],
    model: str = "nomic-embed-text"
) -> np.ndarray:
    """Embed a list of texts. Returns array of shape (n, dim)."""

    response = ollama.embed(
        model=model,
        input=texts
    )

    return np.array(
        response["embeddings"],
        dtype=np.float32
    )


texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]


embeddings = embed(texts)

print(
    f"Shape: {embeddings.shape}"
)

print(
    f"Norm of first vector: "
    f"{np.linalg.norm(embeddings[0]):.4f}"
)