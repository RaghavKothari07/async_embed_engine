from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import openai
import asyncio
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class TextBatch(BaseModel):
    texts: List[str]
    model: str = "text-embedding-3-large"

async def get_embedding(text: str, model: str) -> List[float]:
    try:
        response = await openai.Embedding.acreate(
            input=text,
            model=model
        )
        return response['data'][0]['embedding']
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}")

@app.post("/embed")
async def embed_batch(batch: TextBatch):
    try:
        results = await asyncio.gather(*[
            get_embedding(text, batch.model) for text in batch.texts
        ])
        return {"embeddings": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
