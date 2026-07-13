# src/01_load_up_data.py
# Purpose: Build a high-quality instruction dataset for a German Teaching Assistant.
# Uses 3 working datasets: Alpaca-German, Multilingual-GEC, and OASST-DE.
# Output: JSONL files ready for MLX LoRA fine-tuning.

import json
import os
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ===============================================
# 1. CONFIGURATION
# ===============================================
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")

# Limits to keep dataset manageable for M1 Pro (16GB RAM)
MAX_ALPACA_SAMPLES = 10000      # General instruction
MAX_GEC_SAMPLES = 10000         # Grammar correction
MAX_OASST_SAMPLES = 3000        # Conversation
TEST_SPLIT_RATIO = 0.1

os.makedirs(DATA_DIR, exist_ok=True)

# ===============================================
# 2. FORMATTING FUNCTIONS
# ===============================================

def format_alpaca_sample(example):
    """
    Format from 'LEL-A/translated_german_alpaca'.
    Fields: instruction, input (optional), output.
    """
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()

    if not instruction or not output:
        return None

    # If there's an input field, append it to instruction
    if example.get("input") and example["input"].strip():
        instruction = f"{instruction}\n{example['input'].strip()}"

    text = f"### Instruction: {instruction}\n### Response: {output}"
    return {"text": text}

def format_gec_sample(example):
    """
    Format from 'juancavallotti/multilingual-gec'.
    Fields: sentence (correct), modified (corrupted), lang.
    Only keep German examples (lang == 'de').
    """
    if example.get("lang") != "de":
        return None

    correct = example.get("sentence", "").strip()
    corrupted = example.get("modified", "").strip()

    if not correct or not corrupted or len(correct) > 150 or len(corrupted) > 150:
        return None

    instruction = f"Korrigiere den folgenden deutschen Satz: {corrupted}"
    output = f"Korrigierter Satz: {correct}"

    text = f"### Instruction: {instruction}\n### Response: {output}"
    return {"text": text}

def format_oasst_sample(example):
    """
    Format from 'OpenAssistant/OASST-DE'.
    Has a 'conversation' field which is a list of {role, text} dicts.
    We extract the first user prompt and the assistant response.
    """
    conversation = example.get("conversation", [])
    if not conversation or len(conversation) < 2:
        return None

    # Find first prompter and assistant messages
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

    if len(user_msg) > 200 or len(assistant_msg) > 300:
        return None

    instruction = user_msg
    output = assistant_msg

    text = f"### Instruction: {instruction}\n### Response: {output}"
    return {"text": text}

# ===============================================
# 3. LOAD DATASET 1: German Alpaca
# ===============================================
print("⏳ Loading German Alpaca (LEL-A/translated_german_alpaca)...")
alpaca_formatted = []
try:
    alpaca_dataset = load_dataset("LEL-A/translated_german_alpaca", split="train")
    print(f"   Raw size: {len(alpaca_dataset)}")

    alpaca_dataset = alpaca_dataset.shuffle(seed=42).select(
        range(min(len(alpaca_dataset), MAX_ALPACA_SAMPLES))
    )

    for sample in alpaca_dataset:
        formatted = format_alpaca_sample(sample)
        if formatted:
            alpaca_formatted.append(formatted)

    print(f"✅ Alpaca formatted: {len(alpaca_formatted)} valid samples.")
except Exception as e:
    print(f"⚠️ Error loading Alpaca: {e}")
    alpaca_formatted = []

# ===============================================
# 4. LOAD DATASET 2: Multilingual GEC (German)
# ===============================================
print("⏳ Loading Multilingual GEC (juancavallotti/multilingual-gec)...")
gec_formatted = []
try:
    gec_dataset = load_dataset("juancavallotti/multilingual-gec", split="train")
    print(f"   Raw size: {len(gec_dataset)}")

    # Filter only German examples
    gec_dataset = gec_dataset.filter(lambda x: x.get("lang") == "de")
    print(f"   German subset size: {len(gec_dataset)}")

    gec_dataset = gec_dataset.shuffle(seed=42).select(
        range(min(len(gec_dataset), MAX_GEC_SAMPLES))
    )

    for sample in gec_dataset:
        formatted = format_gec_sample(sample)
        if formatted:
            gec_formatted.append(formatted)

    print(f"✅ GEC formatted: {len(gec_formatted)} valid samples.")
except Exception as e:
    print(f"⚠️ Error loading GEC: {e}")
    gec_formatted = []

# ===============================================
# 5. LOAD DATASET 3: OpenAssistant German
# ===============================================
print("⏳ Loading OpenAssistant German (OpenAssistant/OASST-DE)...")
oasst_formatted = []
try:
    oasst_dataset = load_dataset("OpenAssistant/OASST-DE", split="train")
    print(f"   Raw size: {len(oasst_dataset)}")

    oasst_dataset = oasst_dataset.shuffle(seed=42).select(
        range(min(len(oasst_dataset), MAX_OASST_SAMPLES))
    )

    for sample in oasst_dataset:
        formatted = format_oasst_sample(sample)
        if formatted:
            oasst_formatted.append(formatted)

    print(f"✅ OASST formatted: {len(oasst_formatted)} valid samples.")
except Exception as e:
    print(f"⚠️ Error loading OASST: {e}")
    oasst_formatted = []

# ===============================================
# 6. MERGE, SHUFFLE, AND SPLIT
# ===============================================
combined_data = alpaca_formatted + gec_formatted + oasst_formatted

if not combined_data:
    raise RuntimeError("❌ No data loaded! Please check your internet connection.")

print(f"🔄 Total combined samples: {len(combined_data)}")

random.shuffle(combined_data)

train_data, test_data = train_test_split(
    combined_data,
    test_size=TEST_SPLIT_RATIO,
    random_state=42
)

print(f"📊 Train samples: {len(train_data)}")
print(f"📊 Test samples: {len(test_data)}")

# ===============================================
# 7. SAVE TO JSONL
# ===============================================
def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved: {filepath}")

save_jsonl(train_data, TRAIN_FILE)
save_jsonl(test_data, TEST_FILE)

# ===============================================
# 8. DISPLAY SAMPLE
# ===============================================
print("\n🔍 Sample Training Entry (First item):")
if train_data:
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

print("\n✅ Data preparation finished successfully!")