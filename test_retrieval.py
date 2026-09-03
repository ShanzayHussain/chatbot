"""
Quick sanity check: confirm retrieval from ChromaDB works.
Run: python test_retrieval.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("tcs_knowledge")

test_queries = [
    "how do I track my TCS shipment",
    "what happens if my parcel is lost",
    "does TCS deliver on holidays",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"\n[{meta['source']}] (distance={dist:.3f})")
        print(doc[:200].strip() + "...")