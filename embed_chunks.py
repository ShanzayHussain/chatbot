"""
Phase 3: Embed chunks.json and store them in ChromaDB.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
DB_DIR = "chroma_db"          # persistent storage folder
COLLECTION_NAME = "tcs_knowledge"

def main():
    # 1. Load chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # 2. Load embedding model (small, fast, runs locally — no API needed)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Set up persistent ChromaDB client
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 4. Embed + add in one batch
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"], "word_count": c["word_count"]} for c in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in ChromaDB at ./{DB_DIR}")

if __name__ == "__main__":
    main()