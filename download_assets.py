# download_assets.py
# Pre-download all models, tokenizers, and datasets to the HuggingFace cache.
# Run this ONCE on the master node (it has internet access).

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

print("1. Downloading Qwen2.5-7B-Instruct (Base Model)...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print("✅ 7B Model cached.")

print("2. Downloading Sentence Embedding model (multilingual-e5-small)...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
print("✅ Embedding model cached.")

print("3. Downloading Datasets (full sizes for rich training)...")
try:
    load_dataset("LEL-A/translated_german_alpaca", split="train")
    print("✅ Alpaca cached.")
except Exception:
    print("⚠️ LEL-A failed, trying fallback...")
    load_dataset("fabiochiu/alpaca-german-cleaned", split="train")

try:
    load_dataset("juancavallotti/multilingual-gec", split="train")
    print("✅ GEC cached.")
except Exception:
    print("⚠️ Multilingual-GEC fallback skipped.")

try:
    load_dataset("OpenAssistant/OASST-DE", split="train")
    print("✅ OASST cached.")
except Exception:
    print("⚠️ OASST skipped.")

print("✅ All assets are now cached locally under ~/.cache/huggingface/")
print("You can now run the cluster jobs offline.")