# router.py - 3B model (no pruning)
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from sentence_transformers import SentenceTransformer

# ===============================================
# CONFIGURATION
# ===============================================
BASE_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"   # Use the model name, not a local path
CHROMA_DIR = "chroma_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHAT_KEYWORDS = [
    "merhaba", "nasılsın", "iyiyim", "günaydın", "selam",
    "teşekkür", "ne haber", "iyi akşamlar", "hoşçakal"
]

# ===============================================
# LOAD MODEL (3B, no pruning, float16)
# ===============================================
print("⏳ Loading 3B model (no pruning)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("✅ 3B model ready.")

# ===============================================
# LOAD RAG
# ===============================================
print("⏳ Loading RAG...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("german_grammar")
print("✅ RAG ready.")

# ===============================================
# ROUTER & GENERATION
# ===============================================
def route_query(query: str) -> str:
    q = query.lower()
    for word in CHAT_KEYWORDS:
        if word in q:
            return "chat"
    return "grammar"

def generate_answer(query: str) -> str:
    route = route_query(query)

    if route == "chat":
        prompt = f"User: {query}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

    else:  # grammar
        q_emb = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=2)
        context = "\n---\n".join(results['documents'][0])

        prompt = f"""German Grammar Rules:
{context}

User Question: {query}

Answer in Turkish with German examples:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()