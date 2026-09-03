"""
Phase 7: FastAPI backend for the TCS RAG chatbot.

Run: uvicorn app:app --reload --port 8000
Then test at: http://localhost:8000/docs  (interactive Swagger UI)
"""

import chromadb
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION_NAME = "tcs_knowledge"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_K = 3
DISTANCE_THRESHOLD = 1.1

REFUSAL_MESSAGE = (
    "I don't have information about that. I can only help with questions "
    "about TCS's services, tracking, policies, and offices."
)

SYSTEM_INSTRUCTION = """You are a helpful assistant for TCS (The Courier Service), a Pakistani courier and logistics company.
Answer the user's question using ONLY the context provided below.
If the answer is not contained in the context, respond exactly with:
"I don't have information about that. I can only help with questions about TCS's services, tracking, policies, and offices."
Do not make up information. Do not answer questions unrelated to TCS.
"""

app = FastAPI(title="TCS Chatbot API")

# Allow a frontend running on a different port (e.g. React on :3000) to call this API.
# For a public deployment, replace "*" with your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup, not per-request (this is the expensive part)
print("Loading embedding model and ChromaDB...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)
print("Ready.")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    refused: bool


def retrieve_context(query: str, top_k: int = TOP_K):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0], results["metadatas"][0], results["distances"][0]


def build_prompt(query: str, chunks: list[str]) -> str:
    context_text = "\n\n---\n\n".join(chunks)
    return f"""{SYSTEM_INSTRUCTION}

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""


def call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


@app.get("/")
def root():
    return {"status": "TCS Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.message.strip()

    chunks, metadatas, distances = retrieve_context(query)
    sources = list({m["source"] for m in metadatas})

    if distances[0] > DISTANCE_THRESHOLD:
        return ChatResponse(answer=REFUSAL_MESSAGE, sources=[], refused=True)

    prompt = build_prompt(query, chunks)
    answer = call_ollama(prompt)

    return ChatResponse(answer=answer, sources=sources, refused=False)