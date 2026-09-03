# TCS Assistant — Domain-Scoped RAG Chatbot

A chatbot that answers questions **only within a single company's domain** — built for **TCS (The Courier Service)**, Pakistan's courier and logistics company — using a fully open-source, locally-run LLM (no Gemini/ChatGPT API).

If a question falls outside TCS's services, tracking, policies, or offices, the bot refuses rather than guessing.

---

## How it works

```
User question
    │
    ▼
Embed question (sentence-transformers, all-MiniLM-L6-v2)
    │
    ▼
Search ChromaDB for top-3 most similar chunks of TCS data
    │
    ▼
Best match too dissimilar? ──Yes──▶ Refuse (never calls the LLM)
    │ No
    ▼
Build prompt: strict system instruction + retrieved context + question
    │
    ▼
Send to Ollama (llama3.2:3b, running locally)
    │
    ▼
Return answer + which source documents were used
```

The scope restriction is enforced in **two layers**:
1. A hard numeric cutoff on retrieval similarity — if nothing in the knowledge base is actually relevant, the question never reaches the LLM at all.
2. A system prompt instructing the model to answer only from the given context, as a second line of defense.

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| LLM | **Ollama** running **Llama 3.2 (3B)** | Fully open-source, runs 100% locally — no external API, no API key |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Small, fast, free, runs on CPU |
| Vector store | **ChromaDB** (persistent, local) | Stores embedded knowledge-base chunks for retrieval |
| Backend | **FastAPI** | Exposes a `/chat` HTTP endpoint |
| Frontend | **React** (Vite) | Chat UI |

**Note on deployment:** this runs entirely on a local machine (LLM + vector DB + API + frontend all local). It is not yet deployed to a public cloud server — running Ollama publicly requires a server with enough RAM to hold the model in memory, which is a separate infrastructure step from the chatbot logic itself.

---

## Project structure

```
tcs_data/                  # Raw source text collected from tcsexpress.com
  about_us.txt
  services_overview.txt
  contact_offices.txt
  rates_shipping_policy.txt
  tracking_faqs.txt

chunk_documents.py          # Phase 2: cleans + splits tcs_data/ into chunks.json
chunks.json                 # Output: 71 chunks, one paragraph-aware chunk per entry

embed_chunks.py              # Phase 3: embeds chunks.json into ChromaDB
chroma_db/                   # Persistent vector store (created after running embed_chunks.py)

rag_pipeline.py               # Phase 5+6: CLI chat loop (retrieval + guardrail + Ollama)
app.py                         # Phase 7: FastAPI backend, exposes POST /chat

tcs-chatbot-frontend/           # Phase 8: React chat UI
```

---

## Setup & running it locally

### 1. Install Ollama and pull the model
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama serve   # skip if already running as a background service
```

### 2. Set up the Python environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# CPU-only torch first (avoids pulling large unnecessary CUDA packages)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers chromadb fastapi uvicorn requests
```

### 3. Build the knowledge base (one-time)
```bash
python chunk_documents.py     # tcs_data/*.txt  ->  chunks.json
python embed_chunks.py        # chunks.json  ->  chroma_db/
```

### 4. Run the backend
```bash
uvicorn app:app --reload --port 8000
```
Test at `http://localhost:8000/docs`.

### 5. Run the frontend
```bash
cd tcs-chatbot-frontend
npm install
npm run dev
```
Open `http://localhost:5173`.

---

## Design decisions worth knowing

- **Why Llama 3.2 3B instead of a larger model?** Tested on a machine with 7.1GB RAM — an 8B model caused heavy swapping and slow responses. Since retrieval already narrows the LLM's job to "summarize this specific context," a smaller model performs well here without needing heavy reasoning ability.
- **Why a hard distance threshold instead of relying only on prompting?** Testing showed a clean separation: genuinely in-scope questions scored 0.70–0.85 on retrieval distance, out-of-scope questions scored 1.41–1.76. A cutoff at 1.1 refuses confidently and skips the LLM call entirely for out-of-scope questions — faster, and not dependent on the model choosing to obey the system prompt.
- **Why manually collected data instead of scraping?** For a scoped demo with ~5 source pages, manual collection is faster and gives more control over data quality than building a scraper.

---

## Known limitations

- Data is a static snapshot — TCS's live tracking data (actual shipment status) is not connected; this answers from published policy/FAQ/service info only.
- Not yet deployed publicly — currently runs on `localhost` only.
- Small knowledge base (71 chunks) — retrieval quality would benefit from more source documents if scope expands.