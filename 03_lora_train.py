# 03_lora_train.py
# LoRA fine‑tuning on the pruned 7B model.
# Uses rank 64, bf16, batch size 4, 3 epochs.

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ===============================================
# CONFIGURATION
# ===============================================
BASE_MODEL_PATH = "models/qwen_pruned_7b"
LORA_OUTPUT_DIR = "models/lora_on_pruned_7b"
TRAIN_DATA = "data/train.jsonl"
TEST_DATA = "data/test.jsonl"

# LoRA hyperparameters (upgraded for 7B)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]   # All attention layers

# Training hyperparameters (GPU‑optimized)
BATCH_SIZE = 4
GRAD_ACCUM = 2                # Effective batch = 8
LEARNING_RATE = 3e-4
EPOCHS = 3
MAX_SEQ_LENGTH = 512

print(f"⏳ Loading pruned model from {BASE_MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("⏳ Loading dataset...")
dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "test": TEST_DATA})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print(f"✅ Train: {len(tokenized['train'])} samples")
print(f"✅ Test:  {len(tokenized['test'])} samples")

print("⏳ Applying LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # Should show ~15‑20M parameters

training_args = TrainingArguments(
    output_dir=LORA_OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    bf16=True,                     # Use bf16 on Ampere
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
)

print("🚀 Starting training (3 epochs, ~2‑4 hours on a single A100)...")
trainer.train()

model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print(f"✅ LoRA adapter saved to {LORA_OUTPUT_DIR}")