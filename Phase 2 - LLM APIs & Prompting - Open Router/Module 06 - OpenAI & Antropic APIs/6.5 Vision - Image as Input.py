from openai import OpenAI
import os, base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=r"C:\Users\MSI-PC\Downloads\Generative AI - Open Router\.env"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

# Option A: URL (fastest)
def describe_image_url(url: str) -> str:
    response = client.chat.completions.create(
        model="openrouter/free",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url
                    }
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image."
                }
            ]
        }]
    )

    # Check response
    if not response.choices:
        print("No choices returned.")
        print(response)
        return ""

    message = response.choices[0].message

    if message.content is None:
        print("Model returned no text content.")
        print("Response:")
        print(response)
        return ""

    return message.content


# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    media_type = f"image/{ext}"

    response = client.chat.completions.create(
        model="openrouter/free",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}"
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                }
            ]
        }]
    )

    # Check response
    if not response.choices:
        print("No choices returned.")
        print(response)
        return ""

    message = response.choices[0].message

    if message.content is None:
        print("Model returned no text content.")
        print("Response:")
        print(response)
        return ""

    return message.content


# Usage:
text = describe_image_url(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Sunrise_over_the_sea.jpg/1280px-Sunrise_over_the_sea.jpg"
)

print("\nImage description:")
print(text)


# Example local image:
# text = describe_image_file("gambar.jpg")
# print("\nImage description:")
# print(text)