import ollama
import base64
from pathlib import Path


# Describe image from local file
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()

    response = ollama.chat(
        model="qwen2.5vl:7b",
        messages=[
            {
                "role": "user",
                "content": "What is in this image?",
                "images": [b64]
            }
        ]
    )

    return response["message"]["content"]


# Usage
text = describe_image_file("gambar.jpg")
print(text)