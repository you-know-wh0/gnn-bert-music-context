import json
import matplotlib.pyplot as plt
import os

with open('results/training_history.json') as f:
    history = json.load(f)

os.makedirs('plots', exist_ok=True)

epochs = range(1, len(history['train_loss']) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BERT Training/Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('plots/bert_training_curve.png', dpi=150)
print("Saved plots/bert_training_curve.png")