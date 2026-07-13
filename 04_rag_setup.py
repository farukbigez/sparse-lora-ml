# src/04_rag_setup.py
# Purpose: Build a vector database (ChromaDB) from a German grammar text file.
# Uses the latest LangChain packages: langchain-text-splitters for chunking.
# ChromaDB and sentence-transformers for embeddings and vector storage.

import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitters
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import numpy as np

# ===============================================
# 1. CONFIGURATION
# ===============================================
TEXT_FILE = "data/german_grammar.txt"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # Supports both German & Turkish

# ===============================================
# 2. CHECK IF TEXT FILE EXISTS
# ===============================================
if not os.path.exists(TEXT_FILE):
    raise FileNotFoundError(f"""
    ❌ {TEXT_FILE} not found!
    Please create this file and paste German grammar rules into it.
    """)

# ===============================================
# 3. LOAD AND SPLIT TEXT (Using the new langchain-text-splitters)
# ===============================================
print(f"⏳ Loading text from {TEXT_FILE}...")
loader = TextLoader(TEXT_FILE, encoding="utf-8")
documents = loader.load()

print("⏳ Splitting text into chunks using RecursiveCharacterTextSplitter...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks.")

# ===============================================
# 4. CREATE EMBEDDINGS AND STORE IN CHROMADB (Direct chromadb usage)
# ===============================================
print("⏳ Loading embedding model (multilingual-e5-small)...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Delete old ChromaDB if exists
if os.path.exists(CHROMA_DIR):
    print(f"   Removing old {CHROMA_DIR}...")
    shutil.rmtree(CHROMA_DIR)

print("⏳ Building vector database with ChromaDB...")
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="german_grammar")

# Prepare data for ChromaDB
ids = [f"chunk_{i}" for i in range(len(chunks))]
texts = [chunk.page_content for chunk in chunks]
metadatas = [{"source": TEXT_FILE} for _ in chunks]

# Generate embeddings
embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()

# Add to ChromaDB
collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
)
print(f"✅ Vector database saved to {CHROMA_DIR}")

# ===============================================
# 5. TEST RETRIEVAL
# ===============================================
print("\n🔍 Test Retrieval (Query: 'Almanca'da artikel...'):")
test_query = "Almanca'da artikel nasil secilir?"
query_embedding = embedding_model.encode([test_query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=2)

for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"   {i+1}. {doc[:100]}... (Distance: {dist:.4f})")