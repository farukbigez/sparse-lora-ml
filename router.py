# router.py
# Purpose: Load the pruned+LoRA model, integrate RAG, and route queries.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import chromadb
from sentence_transformers import SentenceTransformer
import os

# ===============================================
# 1. CONFIGURATION
# ===============================================
BASE_MODEL_PATH = "models/qwen_pruned_30percent"
LORA_ADAPTER_PATH = "models/lora_on_pruned"
CHROMA_DIR = "chroma_db"
DEVICE = torch.device("mps")

# Casual chat keywords
CHAT_KEYWORDS = [
    "merhaba", "nasılsın", "iyiyim", "günaydın", "selam",
    "teşekkür", "ne haber", "iyi akşamlar", "hoşçakal"
]

# ===============================================
# 2. LOAD MODEL (FIXED - no duplicate device_map)
# ===============================================
print("⏳ Loading base pruned model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.float16,
    device_map="mps",          # SADECE BİR KERE!
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("⏳ Loading LoRA adapter and merging...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()
print("✅ Model ready for inference.")

# ===============================================
# 3. LOAD RAG (CHROMADB)
# ===============================================
print("⏳ Loading RAG vector database...")
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("german_grammar")
print("✅ RAG ready.")

# ===============================================
# 4. ROUTER FUNCTION
# ===============================================
def route_query(query: str) -> str:
    query_lower = query.lower()
    for word in CHAT_KEYWORDS:
        if word in query_lower:
            return "chat"
    return "grammar"

# ===============================================
# 5. GENERATION FUNCTION
# ===============================================
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
        response = response.replace(prompt, "").strip()
        return response

    else:  # grammar route
        # Retrieve from RAG
        query_embedding = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=2)
        context = "\n---\n".join(results['documents'][0])
        
        prompt = f"""
[INST]
Almanca Gramer Kuralları (Rules):
{context}

Kullanıcının Sorusu (User Query): {query}

Lütfen yukarıdaki kurallara dayanarak kullanıcıya açıklayıcı ve doğru bir cevap ver. Eğer kurallar yetmiyorsa kendi bilgini kullan.
Cevabı Türkçe veya Almanca karışık olarak açıkla.
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
        response = response.replace(prompt, "").strip()
        return response