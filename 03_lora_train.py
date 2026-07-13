# src/03_lora_train.py
# Purpose: Fine-tune the pruned model on German instruction data using LoRA.
# Output: LoRA adapter saved to 'models/lora_on_pruned'

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
import os
import gc

# ===============================================
# 1. CONFIGURATION
# ===============================================
BASE_MODEL_PATH = "models/qwen_pruned_30percent"
LORA_OUTPUT_DIR = "models/lora_on_pruned"
TRAIN_DATA_PATH = "data/train.jsonl"
TEST_DATA_PATH = "data/test.jsonl"
DEVICE = torch.device("mps")

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
EPOCHS = 1
MAX_SEQ_LENGTH = 512

# ===============================================
# 2. LOAD PRUNED MODEL AND TOKENIZER
# ===============================================
print(f"⏳ Loading pruned model from {BASE_MODEL_PATH} onto MPS...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.float16,
    device_map="mps",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Pruned model loaded on MPS.")

# ===============================================
# 3. LOAD JSONL DATASET
# ===============================================
print("⏳ Loading training data...")
dataset = load_dataset("json", data_files={"train": TRAIN_DATA_PATH, "test": TEST_DATA_PATH})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print(f"✅ Train size: {len(tokenized_dataset['train'])}, Test size: {len(tokenized_dataset['test'])}")

# ===============================================
# 4. APPLY LORA
# ===============================================
print("⏳ Applying LoRA configuration...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===============================================
# 5. TRAINING ARGUMENTS & TRAINER (FIXED)
# ===============================================
training_args = TrainingArguments(
    output_dir=LORA_OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=20,
    max_steps=500,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# ===============================================
# 6. START TRAINING
# ===============================================
print("⏳ Starting LoRA training")
trainer.train()

# ===============================================
# 7. SAVE LORA ADAPTER
# ===============================================
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print(f"✅ LoRA adapter saved to {LORA_OUTPUT_DIR}")

del model
gc.collect()
torch.mps.empty_cache()
