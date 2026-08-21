import torch
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sklearn.metrics import f1_score, average_precision_score

from bert_encoder import BertTagClassifier
from dataset import MusicCapsDataset

MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
MAX_LENGTH = 128
CHECKPOINT = 'checkpoints/bert_epoch6.pt'
with open('data/musiccaps/label_vocab.json') as f:
    label_vocab = json.load(f)
num_labels = len(label_vocab)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
test_ds = MusicCapsDataset('data/musiccaps/test.jsonl', tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
model = BertTagClassifier(num_labels=num_labels, model_name=MODEL_NAME).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

all_preds, all_labels, all_probs, all_captions = [], [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu()

        all_probs.append(probs)
        all_preds.append((probs > 0.5).int())
        all_labels.append(labels)
        all_captions.extend(batch['caption'])

preds = torch.cat(all_preds).numpy()
labels = torch.cat(all_labels).numpy()
probs = torch.cat(all_probs).numpy()

macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
auc_pr = average_precision_score(labels, probs, average='macro')

print(f"Macro-F1: {macro_f1:.4f}")
print(f"Micro-F1: {micro_f1:.4f}")
print(f"AUC-PR:   {auc_pr:.4f}")

import os
os.makedirs('results', exist_ok=True)
with open('results/bert_metrics.json', 'w') as f:
    json.dump({'macro_f1': macro_f1, 'micro_f1': micro_f1, 'auc_pr': auc_pr}, f, indent=2)

print("\n--- 5 example predictions ---")
for i in range(5):
    true_tags = [label_vocab[j] for j, v in enumerate(labels[i]) if v == 1]
    pred_tags = [label_vocab[j] for j, v in enumerate(preds[i]) if v == 1]
    print(f"\nCaption: {all_captions[i][:100]}...")
    print(f"True tags: {true_tags}")
    print(f"Predicted tags: {pred_tags}")