# src/02_prune_model.py
# Purpose: Apply Layer-wise Unstructured Magnitude Pruning to Qwen2.5-3B.
# FIX: Prunes each Linear layer individually (instead of globally) to avoid
# OOM (Out of Memory) kills on M1 Pro 16GB systems.
# Output: Saved pruned model to 'models/qwen_pruned_30percent'

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

# ===============================================
# 0. ENVIRONMENT FIXES
# ===============================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===============================================
# 1. CONFIGURATION
# ===============================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
PRUNING_RATIO = 0.30  # 30% sparsity per layer
OUTPUT_DIR = "models/qwen_pruned_30percent"

# ===============================================
# 2. LOAD MODEL ON CPU (LOW MEMORY)
# ===============================================
print("⏳ Loading base model on CPU with low_memory mode...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Model loaded on CPU.")

# ===============================================
# 3. APPLY LAYER-WISE PRUNING (LOW MEMORY FOOTPRINT)
# ===============================================
print(f"⏳ Applying layer-wise pruning with ratio: {PRUNING_RATIO*100}%")
print("   (This prunes one layer at a time, preventing memory spikes)")

# Count layers for progress tracking
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
        linear_layers.append((name, module))

print(f"   Total linear layers to prune: {len(linear_layers)}")

# Prune each layer individually
for idx, (name, module) in enumerate(linear_layers):
    # Apply pruning to this module's weight
    prune.l1_unstructured(module, name="weight", amount=PRUNING_RATIO)
    # Remove the mask (make pruning permanent) immediately to save memory
    prune.remove(module, "weight")

    # Print progress every 50 layers
    if (idx + 1) % 50 == 0 or (idx + 1) == len(linear_layers):
        print(f"   Progress: {idx+1}/{len(linear_layers)} layers pruned.")

print("✅ Layer-wise pruning applied and masks removed.")

# ===============================================
# 4. SAVE PRUNED MODEL
# ===============================================
print(f"⏳ Saving pruned model to '{OUTPUT_DIR}'...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Pruned model saved successfully.")

# ===============================================
# 5. CLEAN UP
# ===============================================
del model
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
print("🧹 Memory cleaned up.")