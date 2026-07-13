# 04_rag_setup.py
# Purpose: Build a ChromaDB vector database from the German grammar text file.
# Uses the multilingual-e5-small embedding model (cached offline).

import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

# Disable internet (embedding model is already cached)
os.environ["HF_HUB_OFFLINE"] = "1"

# ===============================================
# CONFIGURATION
# ===============================================
TEXT_FILE = "data/almanca_gramer.txt"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

if not os.path.exists(TEXT_FILE):
    raise FileNotFoundError(f"❌ {TEXT_FILE} not found! Please create it with grammar rules.")

print("⏳ Loading and chunking grammar text...")
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Smart chunking: split by paragraphs first
chunks = []
current = []
for line in text.split("\n"):
    if line.strip():
        current.append(line.strip())
    else:
        if current:
            chunks.append(" ".join(current))
            current = []
if current:
    chunks.append(" ".join(current))

# Further split long chunks
final_chunks = []
for chunk in chunks:
    if len(chunk) > CHUNK_SIZE:
        for i in range(0, len(chunk), CHUNK_SIZE - CHUNK_OVERLAP):
            final_chunks.append(chunk[i:i + CHUNK_SIZE])
    else:
        final_chunks.append(chunk)

final_chunks = [c.strip() for c in final_chunks if c.strip()]
print(f"✅ Created {len(final_chunks)} chunks.")

print("⏳ Loading embedding model (cached)...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="german_grammar")

print("⏳ Generating embeddings...")
embeddings = embedder.encode(final_chunks, show_progress_bar=True).tolist()

collection.add(
    ids=[f"chunk_{i}" for i in range(len(final_chunks))],
    documents=final_chunks,
    embeddings=embeddings,
    metadatas=[{"source": TEXT_FILE} for _ in final_chunks],
)
print(f"✅ Vector database saved to {CHROMA_DIR}")

# Quick test
test_query = "Almanca'da artikel nasil secilir?"
query_emb = embedder.encode([test_query]).tolist()
results = collection.query(query_embeddings=query_emb, n_results=2)

print("\n🔍 Test Retrieval:")
for i, doc in enumerate(results['documents'][0]):
    print(f"   {i+1}. {doc[:150]}...")