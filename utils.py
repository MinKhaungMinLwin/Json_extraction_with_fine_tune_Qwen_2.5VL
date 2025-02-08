import os
import json
from torch.utils.data import Dataset
from roboflow import Roboflow

# ✅ Load dataset from Roboflow
def get_roboflow_dataset(api_key):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("pallet-load-manifest-json")
    return project.version(2).download("jsonl")

# ✅ Custom Dataset Class
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path, image_directory_path):
        self.image_directory_path = image_directory_path
        self.entries = [json.loads(line) for line in open(jsonl_file_path, 'r')]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        return entry["image"], entry, entry["suffix"]

# ✅ Collate Function
def train_collate_fn(batch):
    _, _, labels = zip(*batch)
    return labels
