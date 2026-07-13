# router.py
# Purpose: Load the pruned + LoRA 7B model, integrate RAG, and route queries.
# Routes between "chat" (casual conversation) and "grammar" (rule-based RAG).

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import chromadb
from sentence_transformers import SentenceTransformer

# ===============================================
# CONFIGURATION
# ===============================================
BASE_MODEL_PATH = "models/qwen_pruned_7b_4bit"
LORA_ADAPTER_PATH = "models/lora_on_pruned_7b_4bit"
CHROMA_DIR = "chroma_db"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Simple keyword-based router for chat vs grammar
CHAT_KEYWORDS = [
    "merhaba", "nasılsın", "iyiyim", "günaydın", "selam",
    "teşekkür", "ne haber", "iyi akşamlar", "hoşçakal"
]

# ===============================================
# LOAD MODEL (MERGED)
# ===============================================
print("⏳ Loading base pruned 7B model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("⏳ Loading LoRA adapter and merging...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()
print("✅ Model ready.")

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
        prompt = f"<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n"
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

    else:  # grammar route
        # Retrieve relevant rules from RAG
        q_emb = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=2)
        context = "\n---\n".join(results['documents'][0])

        prompt = f"""
[INST]
German Grammar Rules:
{context}

User Query: {query}

Based on the rules above, provide a clear and accurate explanation to the user.
If the rules are insufficient, use your own knowledge. Answer in Turkish with German examples.
[/INST]
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()