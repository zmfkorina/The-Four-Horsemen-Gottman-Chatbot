from fastapi import FastAPI, Body
from pydantic import BaseModel
from app.rag import ask

app = FastAPI(title="Relationship RAG (local)")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_endpoint(payload: Query = Body(...)):
    out = ask(payload.question)
    return out