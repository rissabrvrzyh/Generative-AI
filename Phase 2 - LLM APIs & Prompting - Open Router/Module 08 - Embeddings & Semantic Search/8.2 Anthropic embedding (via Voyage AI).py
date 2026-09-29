# pip install voyageai numpy python-dotenv

import voyageai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

vo = voyageai.Client(
    api_key=os.environ["VOYAGE_API_KEY"]
)

result = vo.embed(
    [
        "What is RAG?",
        "Explain vector databases."
    ],
    model="voyage-3",
    input_type="document",
)

embeddings = np.array(
    result.embeddings,
    dtype=np.float32
)

print(f"Shape: {embeddings.shape}")
print(f"Token usage: {result.total_tokens}")