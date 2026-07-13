# 01_load_up_data.py
# Build a large instruction dataset from multiple sources for training.
# Uses 50k+ samples for better generalization.

import json
import os
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Disable internet access for safety (data is cached from download_assets.py)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")

# --- CLUSTER UPGRADE: Use much more data ---
MAX_ALPACA_SAMPLES = 30000   # General instruction
MAX_GEC_SAMPLES = 30000      # Grammar correction
MAX_OASST_SAMPLES = 5000     # Dialogue
TEST_SPLIT_RATIO = 0.1

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Formatting functions (same as before, but with longer samples)
# ------------------------------------------------------------------
def format_alpaca_sample(example):
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()
    if not instruction or not output:
        return None
    if example.get("input") and example["input"].strip():
        instruction = f"{instruction}\n{example['input'].strip()}"
    return {"text": f"### Instruction: {instruction}\n### Response: {output}"}

def format_gec_sample(example):
    if example.get("lang") != "de":
        return None
    correct = example.get("sentence", "").strip()
    corrupted = example.get("modified", "").strip()
    if not correct or not corrupted or len(correct) > 250:
        return None
    text = f"### Instruction: Korrigiere den folgenden deutschen Satz: {corrupted}\n### Response: Korrigierter Satz: {correct}"
    return {"text": text}

def format_oasst_sample(example):
    conversation = example.get("conversation", [])
    if not conversation or len(conversation) < 2:
        return None
    user_msg = None
    assistant_msg = None
    for turn in conversation:
        if turn.get("role") == "prompter" and user_msg is None:
            user_msg = turn.get("text", "").strip()
        elif turn.get("role") == "assistant" and assistant_msg is None:
            assistant_msg = turn.get("text", "").strip()
        if user_msg and assistant_msg:
            break
    if not user_msg or not assistant_msg:
        return None
    return {"text": f"### Instruction: {user_msg}\n### Response: {assistant_msg}"}

# ------------------------------------------------------------------
# Load and format each dataset
# ------------------------------------------------------------------
print("⏳ Loading Alpaca...")
alpaca_formatted = []
try:
    alpaca = load_dataset("LEL-A/translated_german_alpaca", split="train")
    alpaca = alpaca.shuffle(seed=42).select(range(min(len(alpaca), MAX_ALPACA_SAMPLES)))
    for s in alpaca:
        f = format_alpaca_sample(s)
        if f:
            alpaca_formatted.append(f)
    print(f"✅ Alpaca: {len(alpaca_formatted)} samples")
except Exception as e:
    print(f"⚠️ Alpaca failed: {e}")

print("⏳ Loading GEC...")
gec_formatted = []
try:
    gec = load_dataset("juancavallotti/multilingual-gec", split="train")
    gec = gec.filter(lambda x: x.get("lang") == "de")
    gec = gec.shuffle(seed=42).select(range(min(len(gec), MAX_GEC_SAMPLES)))
    for s in gec:
        f = format_gec_sample(s)
        if f:
            gec_formatted.append(f)
    print(f"✅ GEC: {len(gec_formatted)} samples")
except Exception as e:
    print(f"⚠️ GEC failed: {e}")

print("⏳ Loading OASST...")
oasst_formatted = []
try:
    oasst = load_dataset("OpenAssistant/OASST-DE", split="train")
    oasst = oasst.shuffle(seed=42).select(range(min(len(oasst), MAX_OASST_SAMPLES)))
    for s in oasst:
        f = format_oasst_sample(s)
        if f:
            oasst_formatted.append(f)
    print(f"✅ OASST: {len(oasst_formatted)} samples")
except Exception as e:
    print(f"⚠️ OASST failed: {e}")

# Merge and split
combined = alpaca_formatted + gec_formatted + oasst_formatted
random.shuffle(combined)
train_data, test_data = train_test_split(combined, test_size=TEST_SPLIT_RATIO, random_state=42)

# Save JSONL
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(train_data, TRAIN_FILE)
save_jsonl(test_data, TEST_FILE)

print(f"✅ Total Train: {len(train_data)} samples")
print(f"✅ Total Test:  {len(test_data)} samples")