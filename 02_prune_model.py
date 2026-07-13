# 02_prune_model.py (4-bit Quantization ile Pruning)
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ===============================================
# CONFIGURATION
# ===============================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PRUNING_RATIO = 0.30
OUTPUT_DIR = "models/qwen_pruned_7b_4bit"

print("⏳ Loading 7B model with 4-bit quantization (fits 8GB VRAM)...")

# --- 4-bit Quantization Config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Model loaded in 4-bit.")

print(f"⏳ Applying global pruning (ratio: {PRUNING_RATIO*100}%)...")
params_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
        params_to_prune.append((module, "weight"))

prune.global_unstructured(
    params_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=PRUNING_RATIO,
)
for module, _ in params_to_prune:
    prune.remove(module, "weight")
print("✅ Pruning complete.")

print("⏳ Saving pruned model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Saved to {OUTPUT_DIR}")