# -*- coding: utf-8 -*-
import os
import json
import torch
import random
from PIL import Image
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from peft import get_peft_model, LoraConfig
from roboflow import Roboflow
from torch.optim import AdamW
from transformers import get_scheduler

# 🚀 Fix Windows multiprocessing issue
if os.name == "nt":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

# ✅ CUDA Check
print("CUDA Available:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Install `bitsandbytes` if missing
try:
    import bitsandbytes as bnb
    print("BitsAndBytes Version:", bnb.__version__)
except ImportError:
    os.system("pip install bitsandbytes")

# ✅ Load API Keys
load_dotenv()
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
if not ROBOFLOW_API_KEY:
    raise ValueError("⚠️ Please set ROBOFLOW_API_KEY in .env file!")

# ✅ Load Dataset
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("roboflow-jvuqo").project("pallet-load-manifest-json")
version = project.version(2)
dataset = version.download("jsonl")

# ✅ System Prompt
SYSTEM_MESSAGE = """You are a Vision Language Model specialized in extracting structured data from images.
Your task is to analyze the provided image and extract relevant information into a well-structured JSON format.
Provide only the JSON output based on the extracted information. Avoid explanations or comments."""

# ✅ Dataset Formatting
def format_data(image_directory_path, entry):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "image", "image": os.path.join(image_directory_path, entry["image"])},
            {"type": "text", "text": entry["prefix"]},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": entry["suffix"]}]}
    ]

# ✅ Custom Dataset Class
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries(jsonl_file_path)

    def _load_entries(self, jsonl_file_path):
        with open(jsonl_file_path, 'r') as file:
            return [json.loads(line) for line in file]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)
        return image, entry, format_data(self.image_directory_path, entry)

# ✅ Load Datasets
train_dataset = JSONLDataset(f"{dataset.location}/train/annotations.jsonl", f"{dataset.location}/train")
valid_dataset = JSONLDataset(f"{dataset.location}/valid/annotations.jsonl", f"{dataset.location}/valid")

# ✅ Model Config
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
USE_QLORA = True

# ✅ Update BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# ✅ Load Model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto",
    quantization_config=bnb_config if USE_QLORA else None,
    torch_dtype=torch.float16
)
model = get_peft_model(model, lora_config)
model.to(DEVICE)
model.train()

# ✅ Image Preprocessing
MIN_PIXELS, MAX_PIXELS = 256 * 28 * 28, 640 * 28 * 28
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

# ✅ Collate Function
def train_collate_fn(batch):
    _, _, examples = zip(*batch)
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    model_inputs = processor(text=texts, return_tensors="pt", padding=True)
    labels = model_inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return model_inputs["input_ids"], model_inputs["attention_mask"], labels

# ✅ Data Loaders (Windows Fix: `num_workers=0`)
BATCH_SIZE = 2
NUM_WORKERS = 0 if os.name == "nt" else 2

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=train_collate_fn, num_workers=NUM_WORKERS, shuffle=True)

# ✅ Training Config
EPOCHS = 20
LR = 2e-4
ACCUMULATION_STEPS = 4

# ✅ Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=len(train_loader) * EPOCHS,
)

# ✅ Training Loop with Epochs
def train():
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss = loss / ACCUMULATION_STEPS  # Normalize loss for gradient accumulation
            loss.backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} completed. Average Loss: {avg_loss:.4f}")

        # ✅ Save Checkpoint
        save_path = f"checkpoints/qwen2.5-epoch{epoch+1}"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")

# ✅ Run Training
if __name__ == "__main__":
    train()
