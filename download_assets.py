# download_assets.py
# Pre-download all necessary models, tokenizers, and datasets to HF cache.
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

print("1. Downloading Qwen2.5-3B-Instruct (Base Model)...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print("✅ Base model cached.")

print("2. Downloading Sentence Embedding model (multilingual-e5-small)...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
print("✅ Embedding model cached.")

print("3. Downloading datasets (to cache them for 01_load_up_data.py)...")
try:
    # Sadece metadata/ilk birkaç örneği indir, cache'de dosyalar oluşsun.
    load_dataset("LEL-A/translated_german_alpaca", split="train").select(range(5))
except Exception:
    print("⚠️ Skipping LEL-A (maybe unavailable), trying fallback...")
    load_dataset("fabiochiu/alpaca-german-cleaned", split="train").select(range(5))

try:
    load_dataset("juancavallotti/multilingual-gec", split="train").select(range(5))
except Exception:
    print("⚠️ Multilingual-GEC fallback skipped.")

try:
    load_dataset("OpenAssistant/OASST-DE", split="train").select(range(5))
except Exception:
    print("⚠️ OASST-DE skipped.")

print("✅ All assets are now cached locally under ~/.cache/huggingface/")