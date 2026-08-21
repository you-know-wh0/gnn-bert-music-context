# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sklearn.metrics import f1_score
import json
import os

from bert_encoder import BertTagClassifier
from dataset import MusicCapsDataset

MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
EPOCHS = 6
LR = 2e-5
MAX_LENGTH = 128

with open('data/musiccaps/label_vocab.json') as f:
    label_vocab = json.load(f)
num_labels = len(label_vocab)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
train_ds = MusicCapsDataset('data/musiccaps/train.jsonl', tokenizer, MAX_LENGTH)
val_ds = MusicCapsDataset('data/musiccaps/val.jsonl', tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print("Using device:", device)

model = BertTagClassifier(num_labels=num_labels, model_name=MODEL_NAME).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': [], 'val_micro_f1': []}
best_val_f1 = -1
best_epoch = -1

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
            probs = torch.sigmoid(logits).cpu()
            all_preds.append((probs > 0.5).int())
            all_labels.append(labels.cpu())
    avg_val_loss = total_val_loss / len(val_loader)

    preds_cat = torch.cat(all_preds).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(labels_cat, preds_cat, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_cat, preds_cat, average='micro', zero_division=0)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_macro_f1'].append(macro_f1)
    history['val_micro_f1'].append(micro_f1)

    print(f"Epoch {epoch+1}/{EPOCHS} — train_loss: {avg_train_loss:.4f}  val_loss: {avg_val_loss:.4f}  val_macro_f1: {macro_f1:.4f}  val_micro_f1: {micro_f1:.4f}")

    torch.save(model.state_dict(), f'checkpoints/bert_epoch{epoch+1}.pt')

    if macro_f1 > best_val_f1:
        best_val_f1 = macro_f1
        best_epoch = epoch + 1

with open('results/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\nTraining done. Best epoch by val_macro_f1: {best_epoch} (f1={best_val_f1:.4f})")
print("Use this epoch's checkpoint in evaluate.py")