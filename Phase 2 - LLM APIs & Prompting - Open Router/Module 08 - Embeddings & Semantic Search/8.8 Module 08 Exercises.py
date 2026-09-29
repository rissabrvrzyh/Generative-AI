import hashlib
import math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

EMBED_MODEL = "openai/text-embedding-3-small"


# ============================================================
# EMBEDDING HELPER
# ============================================================

def embed_texts(
    texts: list[str],
    model: str = EMBED_MODEL
) -> np.ndarray:
    """Embed a list of texts."""

    response = client.embeddings.create(
        input=texts,
        model=model
    )

    vectors = sorted(
        response.data,
        key=lambda x: x.index
    )

    return np.array(
        [v.embedding for v in vectors],
        dtype=np.float32
    )


# ============================================================
# EXERCISE 1
# DUPLICATE DETECTOR
# ============================================================

@dataclass
class DuplicatePair:
    doc_id_1: str
    doc_id_2: str
    similarity: float


class DuplicateDetector:
    """
    Detect near-duplicate documents using cosine similarity.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        model: str = EMBED_MODEL
    ):
        self.threshold = threshold
        self.model = model

    def find_duplicates(
        self,
        documents: list[tuple[str, str]]
    ) -> list[DuplicatePair]:
        """
        documents:
            List of (document_id, document_text)

        Returns:
            Pairs whose cosine similarity is >= threshold.
        """

        if len(documents) < 2:
            return []

        ids = [
            doc_id
            for doc_id, _ in documents
        ]

        texts = [
            text
            for _, text in documents
        ]

        embeddings = embed_texts(
            texts,
            model=self.model
        )

        # Normalize embeddings
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        normalized = (
            embeddings /
            np.where(norms == 0, 1, norms)
        )

        # Cosine similarity matrix
        similarity_matrix = normalized @ normalized.T

        duplicates = []

        for i in range(len(documents)):

            for j in range(i + 1, len(documents)):

                similarity = float(
                    similarity_matrix[i][j]
                )

                if similarity >= self.threshold:

                    duplicates.append(
                        DuplicatePair(
                            doc_id_1=ids[i],
                            doc_id_2=ids[j],
                            similarity=similarity
                        )
                    )

        return sorted(
            duplicates,
            key=lambda x: x.similarity,
            reverse=True
        )


# ============================================================
# 50-DOCUMENT CORPUS
# ============================================================

CORPUS_50 = [
    ("d01", "Retrieval-Augmented Generation combines retrieval with language model generation."),
    ("d02", "RAG combines information retrieval with language model generation."),
    ("d03", "Retrieval-Augmented Generation retrieves external documents before generating answers."),
    ("d04", "Vector databases store numerical embeddings for semantic search."),
    ("d05", "Vector databases store high-dimensional embeddings for similarity search."),
    ("d06", "A vector database enables efficient similarity search over embeddings."),
    ("d07", "Fine-tuning adapts a pretrained model using task-specific training data."),
    ("d08", "Fine tuning adapts pretrained language models to specialized tasks."),
    ("d09", "Fine-tuning trains a pretrained model on a curated task-specific dataset."),
    ("d10", "Prompt engineering designs prompts to guide language model outputs."),
    ("d11", "Prompt engineering involves designing inputs that guide LLM behavior."),
    ("d12", "Careful prompt design can improve the output of language models."),
    ("d13", "Cosine similarity measures the angle between two vectors."),
    ("d14", "Cosine similarity compares vectors by measuring the angle between them."),
    ("d15", "The cosine similarity score is commonly used for text embeddings."),
    ("d16", "Transformers use self-attention to model relationships between tokens."),
    ("d17", "The transformer architecture uses self-attention across sequence tokens."),
    ("d18", "Self-attention allows transformers to model relationships between tokens."),
    ("d19", "Agents can use language models to plan and execute multi-step tasks."),
    ("d20", "LLM agents can plan multiple steps and call external tools."),
    ("d21", "Language model agents use tools to perform multi-step tasks."),
    ("d22", "Chunking divides documents into smaller pieces for retrieval."),
    ("d23", "Document chunking splits large documents into smaller retrieval units."),
    ("d24", "Chunking strategies divide documents into smaller sections for RAG."),
    ("d25", "HNSW is an algorithm used for approximate nearest-neighbor search."),
    ("d26", "HNSW enables fast approximate nearest-neighbor vector search."),
    ("d27", "Approximate nearest-neighbor search can use HNSW indexes."),
    ("d28", "Embeddings represent text as numerical vectors."),
    ("d29", "Text embeddings convert language into numerical vector representations."),
    ("d30", "Embedding models transform text into vectors for machine learning."),
    ("d31", "Semantic search retrieves documents based on meaning rather than exact keywords."),
    ("d32", "Semantic search finds documents using their meaning and embeddings."),
    ("d33", "Embedding-based search can retrieve semantically related documents."),
    ("d34", "BM25 is a keyword-based ranking algorithm for information retrieval."),
    ("d35", "BM25 ranks documents according to keyword frequency and document length."),
    ("d36", "The BM25 algorithm is widely used for lexical document retrieval."),
    ("d37", "Hybrid search combines semantic and keyword-based retrieval."),
    ("d38", "Hybrid retrieval combines vector similarity with lexical search."),
    ("d39", "A hybrid search system blends semantic similarity and keyword scores."),
    ("d40", "Reranking can improve the ordering of retrieved search results."),
    ("d41", "A reranker reorders retrieved documents according to relevance."),
    ("d42", "Reranking retrieved candidates can improve search relevance."),
    ("d43", "RAG systems retrieve relevant context before generating responses."),
    ("d44", "RAG retrieves relevant context and provides it to a language model."),
    ("d45", "Retrieval augmented systems use external context during generation."),
    ("d46", "SQLite is a lightweight relational database stored in a local file."),
    ("d47", "SQLite provides a lightweight local relational database."),
    ("d48", "Caching embeddings can avoid repeated API calls."),
    ("d49", "Embedding caches store vectors so repeated texts do not require new API calls."),
    ("d50", "Local embedding caches reduce redundant embedding API requests."),
]


# ============================================================
# RUN EXERCISE 1
# ============================================================

print("\n" + "=" * 70)
print("EXERCISE 1 - DUPLICATE DETECTOR")
print("=" * 70)

detector = DuplicateDetector(
    threshold=0.95
)

duplicates = detector.find_duplicates(
    CORPUS_50
)

print(
    f"\nFound {len(duplicates)} near-duplicate pairs."
)

for pair in duplicates:
    print(
        f"{pair.doc_id_1} <-> "
        f"{pair.doc_id_2} | "
        f"similarity={pair.similarity:.4f}"
    )


# ============================================================
# EXERCISE 2
# HYBRID SEARCH
# ============================================================

@dataclass
class HybridResult:
    doc_id: str
    text: str
    semantic_score: float
    keyword_score: float
    final_score: float


class HybridSearch:
    """
    Combines semantic similarity with a simple BM25-style
    keyword score.

    final_score =
        alpha * semantic_score
        + (1 - alpha) * keyword_score
    """

    def __init__(
        self,
        documents: list[tuple[str, str]],
        alpha: float = 0.5
    ):
        if not 0 <= alpha <= 1:
            raise ValueError(
                "alpha must be between 0 and 1."
            )

        self.documents = documents
        self.alpha = alpha

        self.ids = [
            doc_id
            for doc_id, _ in documents
        ]

        self.texts = [
            text
            for _, text in documents
        ]

        self.tokens = [
            self.tokenize(text)
            for text in self.texts
        ]

        self.embeddings = embed_texts(
            self.texts
        )

        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True
        )

        self.embeddings = (
            self.embeddings /
            np.where(norms == 0, 1, norms)
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    def keyword_score(
        self,
        query_tokens: list[str],
        document_tokens: list[str]
    ) -> float:

        if not query_tokens:
            return 0.0

        doc_length = len(document_tokens)

        if doc_length == 0:
            return 0.0

        # Count term frequency
        score = 0.0

        for term in query_tokens:

            tf = document_tokens.count(term)

            if tf > 0:

                # Simple BM25-style term score
                score += (
                    tf /
                    (
                        tf
                        + 1.5
                        * (
                            0.5
                            + 0.5
                            * doc_length
                            / 20
                        )
                    )
                )

        # Normalize approximately to 0-1
        return min(
            score / len(query_tokens),
            1.0
        )

    def search(
        self,
        query: str,
        k: int = 5
    ) -> list[HybridResult]:

        query_embedding = embed_texts(
            [query]
        )[0]

        query_norm = np.linalg.norm(
            query_embedding
        )

        if query_norm == 0:
            return []

        query_embedding = (
            query_embedding / query_norm
        )

        # Semantic similarity
        semantic_scores = (
            self.embeddings
            @ query_embedding
        )

        # Keyword scores
        query_tokens = self.tokenize(
            query
        )

        keyword_scores = np.array([
            self.keyword_score(
                query_tokens,
                doc_tokens
            )
            for doc_tokens in self.tokens
        ])

        # Final hybrid score
        final_scores = (
            self.alpha
            * semantic_scores
            +
            (1 - self.alpha)
            * keyword_scores
        )

        top_idx = np.argsort(
            final_scores
        )[::-1][:k]

        return [
            HybridResult(
                doc_id=self.ids[int(i)],
                text=self.texts[int(i)],
                semantic_score=float(
                    semantic_scores[i]
                ),
                keyword_score=float(
                    keyword_scores[i]
                ),
                final_score=float(
                    final_scores[i]
                )
            )
            for i in top_idx
        ]


print("\n" + "=" * 70)
print("EXERCISE 2 - HYBRID SEARCH")
print("=" * 70)

hybrid = HybridSearch(
    CORPUS_50,
    alpha=0.7
)

query = "RAG retrieval documents"

results = hybrid.search(
    query,
    k=5
)

print(
    f"\nQuery: {query}"
)

print(
    f"Alpha: {hybrid.alpha}"
)

for result in results:

    print(
        f"\n[{result.doc_id}]"
    )

    print(
        f"Semantic: {result.semantic_score:.4f}"
    )

    print(
        f"Keyword:  {result.keyword_score:.4f}"
    )

    print(
        f"Final:    {result.final_score:.4f}"
    )

    print(
        f"Text: {result.text}"
    )


# ============================================================
# EXERCISE 3
# VECTOR STORE WITH DELETE AND UPDATE
# ============================================================

@dataclass
class Document:
    id: str
    text: str
    embedding: Optional[np.ndarray] = field(
        default=None,
        repr=False
    )


class VectorStore:
    """
    In-memory vector store with:
    - add
    - search
    - delete
    - update
    """

    def __init__(
        self,
        model: str = EMBED_MODEL
    ):
        self.model = model
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def _rebuild_matrix(self):
        if not self._documents:

            self._matrix = None

            return

        self._matrix = np.array(
            [
                doc.embedding
                for doc in self._documents
            ],
            dtype=np.float32
        )

    def add_documents(
        self,
        documents: list[Document]
    ):

        embeddings = embed_texts(
            [
                doc.text
                for doc in documents
            ],
            model=self.model
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        embeddings = (
            embeddings /
            np.where(norms == 0, 1, norms)
        )

        for doc, embedding in zip(
            documents,
            embeddings
        ):
            doc.embedding = embedding
            self._documents.append(doc)

        self._rebuild_matrix()

    def delete(
        self,
        doc_id: str
    ) -> None:
        """Delete document and rebuild matrix."""

        original_count = len(
            self._documents
        )

        self._documents = [
            doc
            for doc in self._documents
            if doc.id != doc_id
        ]

        if len(self._documents) == original_count:

            raise KeyError(
                f"Document '{doc_id}' not found."
            )

        self._rebuild_matrix()

    def update(
        self,
        doc_id: str,
        new_text: str
    ) -> None:
        """Update document text and embedding."""

        target = None

        for doc in self._documents:

            if doc.id == doc_id:
                target = doc
                break

        if target is None:

            raise KeyError(
                f"Document '{doc_id}' not found."
            )

        new_embedding = embed_texts(
            [new_text],
            model=self.model
        )[0]

        norm = np.linalg.norm(
            new_embedding
        )

        if norm != 0:

            new_embedding = (
                new_embedding / norm
            )

        target.text = new_text
        target.embedding = new_embedding

        self._rebuild_matrix()

    def search(
        self,
        query: str,
        k: int = 5
    ):

        if not self._documents:

            raise RuntimeError(
                "No documents indexed."
            )

        query_embedding = embed_texts(
            [query],
            model=self.model
        )[0]

        norm = np.linalg.norm(
            query_embedding
        )

        if norm == 0:
            return []

        query_embedding = (
            query_embedding / norm
        )

        scores = (
            self._matrix
            @ query_embedding
        )

        k = min(
            k,
            len(self._documents)
        )

        top_idx = np.argsort(
            scores
        )[::-1][:k]

        return [
            (
                self._documents[int(i)],
                float(scores[i])
            )
            for i in top_idx
        ]

    @property
    def size(self):
        return len(
            self._documents
        )


print("\n" + "=" * 70)
print("EXERCISE 3 - DELETE AND UPDATE")
print("=" * 70)

store = VectorStore()

store.add_documents([
    Document(
        "d01",
        "RAG combines retrieval with language model generation."
    ),
    Document(
        "d02",
        "Vector databases store embeddings."
    ),
    Document(
        "d03",
        "Fine-tuning adapts models to specific tasks."
    ),
])

print(
    f"\nInitial size: {store.size}"
)

store.delete("d02")

print(
    f"After deleting d02: {store.size}"
)

store.update(
    "d03",
    "Fine-tuning trains a pretrained model on task-specific data."
)

print(
    "Updated d03 successfully."
)

print(
    f"Matrix shape: {store._matrix.shape}"
)

results = store.search(
    "How does fine-tuning work?",
    k=2
)

print("\nSearch after update/delete:")

for doc, score in results:

    print(
        f"[{score:.4f}] "
        f"{doc.id}: "
        f"{doc.text}"
    )


# ============================================================
# EXERCISE 4
# SQLITE EMBEDDING CACHE
# ============================================================

class EmbeddingCache:

    def __init__(
        self,
        database_path: str = "embedding_cache.db"
    ):
        self.database_path = database_path

        self.connection = sqlite3.connect(
            database_path
        )

        self._create_table()

    def _create_table(self):

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)

        self.connection.commit()

    @staticmethod
    def make_key(
        text: str,
        model: str
    ) -> str:

        raw = (
            text
            + "::"
            + model
        ).encode("utf-8")

        return hashlib.sha256(
            raw
        ).hexdigest()

    def get(
        self,
        text: str,
        model: str
    ) -> Optional[np.ndarray]:

        key = self.make_key(
            text,
            model
        )

        row = self.connection.execute(
            """
            SELECT embedding
            FROM embeddings
            WHERE cache_key = ?
            """,
            (key,)
        ).fetchone()

        if row is None:
            return None

        return np.frombuffer(
            row[0],
            dtype=np.float32
        ).copy()

    def set(
        self,
        text: str,
        model: str,
        embedding: np.ndarray
    ):

        key = self.make_key(
            text,
            model
        )

        self.connection.execute(
            """
            INSERT OR REPLACE INTO embeddings
            (
                cache_key,
                text,
                model,
                embedding
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                key,
                text,
                model,
                embedding.astype(
                    np.float32
                ).tobytes()
            )
        )

        self.connection.commit()

    def close(self):

        self.connection.close()


cache = EmbeddingCache()


def embed_with_cache(
    texts: list[str],
    model: str = EMBED_MODEL
) -> np.ndarray:
    """
    Embed texts using a SQLite cache.

    Cached texts do not trigger another API request.
    """

    results: list[Optional[np.ndarray]] = [
        None
        for _ in texts
    ]

    missing_texts = []
    missing_indices = []

    # Check cache
    for index, text in enumerate(texts):

        cached = cache.get(
            text,
            model
        )

        if cached is not None:

            results[index] = cached

        else:

            missing_texts.append(text)

            missing_indices.append(index)

    # Only call API for missing texts
    if missing_texts:

        print(
            f"API request: "
            f"{len(missing_texts)} new text(s)"
        )

        new_embeddings = embed_texts(
            missing_texts,
            model=model
        )

        for index, text, embedding in zip(
            missing_indices,
            missing_texts,
            new_embeddings
        ):

            cache.set(
                text,
                model,
                embedding
            )

            results[index] = embedding

    else:

        print(
            "All embeddings loaded from cache."
        )

    return np.array(
        results,
        dtype=np.float32
    )


print("\n" + "=" * 70)
print("EXERCISE 4 - SQLITE EMBEDDING CACHE")
print("=" * 70)


cache_texts = [
    "What is RAG?",
    "What is a vector database?",
    "What is RAG?",
]


print("\nFirst call:")

embeddings_1 = embed_with_cache(
    cache_texts
)

print(
    f"Shape: {embeddings_1.shape}"
)


print("\nSecond call:")

embeddings_2 = embed_with_cache(
    cache_texts
)

print(
    f"Shape: {embeddings_2.shape}"
)


print(
    "\nFirst and second results are identical:",
    np.allclose(
        embeddings_1,
        embeddings_2
    )
)


cache.close()