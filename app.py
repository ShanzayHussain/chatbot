"""
Phase 7 (updated): FastAPI backend for the TCS RAG chatbot.
Now uses Groq's hosted API to run an open-source Llama model,
instead of a locally-running Ollama instance — makes public
deployment possible without needing to rent a GPU/RAM-heavy server.

Run: uvicorn app:app --reload --port 8000
Then test at: http://localhost:8000/docs  (interactive Swagger UI)
"""

import os

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads GROQ_API_KEY from a local .env file

DB_DIR = "chroma_db"
COLLECTION_NAME = "tcs_knowledge"
GROQ_MODEL = "qwen/qwen3.6-27b"  # open-source model (Alibaba), confirmed available on this Groq account via /v1/models

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

# Allow the frontend to call this API. For production, replace "*" with
# your actual deployed frontend URL (e.g. "https://your-app.vercel.app").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model and ChromaDB...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=DB_DIR)

try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Loaded existing collection with {collection.count()} chunks.")
except Exception:
    print("Collection not found — building it from chunks.json now...")
    import json

    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"], "word_count": c["word_count"]} for c in chunks]
    embeddings = embed_model.encode(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Built collection with {collection.count()} chunks.")

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
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


import re


def call_groq(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
        reasoning_effort="none",  # Qwen is a reasoning model; this suppresses the <think> block
    )
    raw_answer = response.choices[0].message.content

    # Fallback safety net: strip any <think>...</think> block that slips through anyway,
    # in case this Groq model/version ignores the reasoning_effort parameter.
    cleaned = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
    return cleaned


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
    answer = call_groq(prompt)

    return ChatResponse(answer=answer, sources=sources, refused=False)