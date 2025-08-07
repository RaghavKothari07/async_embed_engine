import requests

endpoint = "http://localhost:8000/embed"
payload = {
    "texts": [
        "AI is transforming the world.",
        "Fast embedding pipelines are critical for scale."
    ],
    "model": "text-embedding-3-large"
}
response = requests.post(endpoint, json=payload)
print(response.json())
