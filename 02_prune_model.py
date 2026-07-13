# 02_prune_model.py
# Global unstructured pruning (30%) on Qwen2.5-7B-Instruct.
# Runs on GPU (CUDA) using bfloat16 for memory efficiency.

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===============================================
# CONFIGURATION
# ===============================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PRUNING_RATIO = 0.30   # 30% sparsity
OUTPUT_DIR = "models/qwen_pruned_7b"

print("⏳ Loading 7B model on GPU (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,   # Use bf16 for Ampere GPUs
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Model loaded.")

# Collect all Linear layers except lm_head
params_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
        params_to_prune.append((module, "weight"))

print(f"⏳ Pruning {len(params_to_prune)} layers globally (ratio = {PRUNING_RATIO*100}%)...")
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