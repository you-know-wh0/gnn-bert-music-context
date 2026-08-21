import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class MusicCapsDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        self.rows = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row['caption'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label_vector'], dtype=torch.float32),
            'caption': row['caption']
        }