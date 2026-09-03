"""
Phase 5: Full RAG pipeline.
User question -> embed -> retrieve top chunks from ChromaDB
-> build prompt with context -> send to Ollama -> print answer.

Run: python rag_pipeline.py
"""

import chromadb
import requests
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION_NAME = "tcs_knowledge"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_K = 3  # how many chunks to retrieve per query

# Based on real testing: in-scope questions scored 0.70-0.85 best-match distance,
# out-of-scope questions scored 1.41-1.76. 1.1 sits cleanly in the gap between them.
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


def load_pipeline():
    print("Loading embedding model and ChromaDB...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


def retrieve_context(query, model, collection, top_k=TOP_K):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return chunks, metadatas, distances


def build_prompt(query, chunks):
    context_text = "\n\n---\n\n".join(chunks)
    prompt = f"""{SYSTEM_INSTRUCTION}

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
    return prompt


def call_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    model, collection = load_pipeline()
    print(f"\nReady! Ask questions about TCS (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        chunks, metadatas, distances = retrieve_context(query, model, collection)

        print(f"\n[debug] top match distances: {[round(d, 3) for d in distances]}")
        print(f"[debug] sources used: {[m['source'] for m in metadatas]}")

        # Hard guardrail: if even the best match is too dissimilar, refuse
        # WITHOUT calling the LLM at all. Faster, and can't be talked around
        # by clever prompting since the LLM never even sees the question.
        if distances[0] > DISTANCE_THRESHOLD:
            print(f"\nBot: {REFUSAL_MESSAGE}")
            print("[debug] refused before LLM call - best distance exceeded threshold]\n")
            continue

        prompt = build_prompt(query, chunks)
        answer = call_ollama(prompt)

        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()