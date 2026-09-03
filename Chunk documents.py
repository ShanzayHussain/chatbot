"""
Phase 2: Clean + Chunk TCS data for RAG pipeline.
 
Takes .txt files from tcs_data/, cleans them, and splits them into
overlapping, paragraph-aware chunks. Outputs a single chunks.json
ready for embedding in Phase 3.
"""
 
import os
import re
import json
from pathlib import Path
 
DATA_DIR = Path("tcs_data")
OUTPUT_FILE = Path("chunks.json")
 
MAX_WORDS = 200      # target chunk size
OVERLAP_WORDS = 40    # words repeated between consecutive chunks (context continuity)
 
 
def clean_text(text: str) -> str:
    """Strip excessive whitespace/blank lines, normalize spacing."""
    text = text.replace("\r\n", "\n")
    # Collapse 3+ blank lines into 2 (paragraph breaks preserved)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
 
 
def split_into_blocks(text: str):
    """Split on blank lines -> list of paragraph-like blocks."""
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks
 
 
def chunk_blocks(blocks, source_name):
    """
    Merge small blocks together up to MAX_WORDS, preserving block
    boundaries (never splitting a paragraph/QA-pair in half unless
    it alone exceeds MAX_WORDS).
    """
    chunks = []
    current_words = []
    current_blocks = []
 
    def flush():
        if current_blocks:
            chunk_text = "\n\n".join(current_blocks)
            chunks.append({
                "text": chunk_text,
                "source": source_name,
                "word_count": len(current_words),
            })
 
    for block in blocks:
        block_words = block.split()
 
        # If a single block is itself huge (e.g. a giant legal clause),
        # split it on sentence boundaries instead of dropping it whole.
        if len(block_words) > MAX_WORDS:
            flush()
            current_blocks, current_words = [], []
            sentences = re.split(r"(?<=[.!?])\s+", block)
            sub_words = []
            sub_sentences = []
            for sent in sentences:
                sub_sentences.append(sent)
                sub_words.extend(sent.split())
                if len(sub_words) >= MAX_WORDS:
                    chunks.append({
                        "text": " ".join(sub_sentences),
                        "source": source_name,
                        "word_count": len(sub_words),
                    })
                    sub_sentences, sub_words = [], []
            if sub_sentences:
                chunks.append({
                    "text": " ".join(sub_sentences),
                    "source": source_name,
                    "word_count": len(sub_words),
                })
            continue
 
        # Normal case: accumulate blocks until MAX_WORDS reached
        if len(current_words) + len(block_words) > MAX_WORDS and current_blocks:
            flush()
            # start next chunk with overlap: carry the tail of the previous chunk forward
            overlap_text = " ".join(current_words[-OVERLAP_WORDS:]) if current_words else ""
            current_blocks = [overlap_text] if overlap_text else []
            current_words = overlap_text.split() if overlap_text else []
 
        current_blocks.append(block)
        current_words.extend(block_words)
 
    flush()
    return chunks
 
 
def process_all_files():
    all_chunks = []
    for file_path in sorted(DATA_DIR.glob("*.txt")):
        raw = file_path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        blocks = split_into_blocks(cleaned)
        file_chunks = chunk_blocks(blocks, file_path.stem)
        all_chunks.extend(file_chunks)
        print(f"{file_path.name}: {len(blocks)} blocks -> {len(file_chunks)} chunks")
 
    # Assign global IDs
    for i, c in enumerate(all_chunks):
        c["id"] = f"chunk_{i:04d}"
 
    OUTPUT_FILE.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to {OUTPUT_FILE}")
    return all_chunks
 
 
if __name__ == "__main__":
    process_all_files()