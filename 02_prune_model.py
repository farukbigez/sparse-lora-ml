# 02_prune_model.py
# Purpose: Apply Layer-wise Unstructured Magnitude Pruning to Qwen2.5-7B.
# Prunes each Linear layer individually (instead of globally) to prevent OOM
# on GPUs with limited VRAM (e.g., 8GB).
# Output: Saved pruned model to 'models/qwen_pruned_7b_4bit'

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
PRUNING_RATIO = 0.30   # 30% sparsity
OUTPUT_DIR = "models/qwen_pruned_7b_4bit"

print("⏳ Loading 7B model with 4-bit quantization (fits 8GB VRAM)...")

# 4-bit Quantization Config to fit the RTX 2070 8GB
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

print(f"⏳ Applying layer-wise pruning (ratio: {PRUNING_RATIO*100}%)...")

# Collect all Linear layers (excluding lm_head to avoid catastrophic loss)
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
        linear_layers.append((name, module))

print(f"   Total layers to prune: {len(linear_layers)}")

# Prune each layer one by one to keep memory usage low
for idx, (name, module) in enumerate(linear_layers):
    prune.l1_unstructured(module, name="weight", amount=PRUNING_RATIO)
    prune.remove(module, "weight")  # Remove mask immediately to free memory
    
    if (idx + 1) % 50 == 0 or (idx + 1) == len(linear_layers):
        print(f"   Progress: {idx+1}/{len(linear_layers)} layers pruned.")

print("✅ Pruning complete.")

print("⏳ Saving pruned model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Saved to {OUTPUT_DIR}")