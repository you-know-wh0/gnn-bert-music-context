import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class FMASegmentDataset(Dataset):
    def __init__(self, split_file, processed_dir, genre_to_idx=None):
        self.processed_dir = processed_dir
        
        # Load split JSON
        with open(split_file, 'r', encoding='utf-8') as f:
            self.split_data = json.load(f)
            
        # Get unique genres (excluding empty)
        if genre_to_idx is None:
            unique_genres = sorted(list(set(
                item["genre"] for item in self.split_data.values() if item["genre"]
            )))
            self.genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        else:
            self.genre_to_idx = genre_to_idx
            
        self.idx_to_genre = {i: g for g, i in self.genre_to_idx.items()}
        
        # Build samples list: check both JSON genre and existence of processed npz
        self.samples = []
        for track_id, info in self.split_data.items():
            genre = info["genre"]
            if not genre: # Skip if no top genre
                continue
                
            npz_path = os.path.join(processed_dir, f"{track_id}.npz")
            if os.path.exists(npz_path):
                self.samples.append((npz_path, self.genre_to_idx[genre], track_id))
                
    def __len__(self):
        # We segment each track into 6 parts
        return len(self.samples) * 6
        
    def __getitem__(self, index):
        track_index = index // 6
        segment_index = index % 6
        
        npz_path, genre_idx, track_id = self.samples[track_index]
        
        # Load npz
        data = np.load(npz_path)
        mel_seg = data["mel"][segment_index] # shape: [128, frames]
        
        # Ensure exact time frame length of 215 by padding/truncating
        target_frames = 215
        if mel_seg.shape[1] < target_frames:
            pad_width = target_frames - mel_seg.shape[1]
            mel_seg = np.pad(mel_seg, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_seg.shape[1] > target_frames:
            mel_seg = mel_seg[:, :target_frames]
        
        # Add channel dimension: [1, 128, 215]
        mel_tensor = torch.tensor(mel_seg, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(genre_idx, dtype=torch.long)
        
        return mel_tensor, label_tensor


class MelSpectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(MelSpectrogramCNN, self).__init__()
        
        # Input shape: [batch_size, 1, 128, 215]
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # shape: [16, 64, 107]
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # shape: [32, 32, 53]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # shape: [64, 16, 26]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # shape: [128, 4, 4]
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    # Test directory config
    split_file = "data/splits/training.json"
    processed_dir = "data/processed/audio_features"
    
    print("Initializing FMASegmentDataset...")
    dataset = FMASegmentDataset(split_file, processed_dir)
    print(f"Dataset initialized with {len(dataset)} segment samples.")
    print("Genre classes mapping:", dataset.genre_to_idx)
    
    if len(dataset) == 0:
        print("No samples found! Make sure you ran feature extraction and split generation first.")
    else:
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Instantiate Model
        num_classes = len(dataset.genre_to_idx)
        model = MelSpectrogramCNN(num_classes=num_classes)
        
        # Test a forward pass
        inputs, labels = next(iter(dataloader))
        print("Batch input shape:", inputs.shape)
        print("Batch labels shape:", labels.shape)
        
        outputs = model(inputs)
        print("Model output shape:", outputs.shape)
        
        # Run one dummy training epoch
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        print(f"Running test training epoch on device: {device}...")
        loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch finished. Loss: {loss:.4f}, Accuracy: {acc:.4f}")
