import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from audio_features import load_audio, extract_mel_spectrogram, extract_chroma, normalize_features

# Config
SAMPLE_FILE = "data/raw/fma_medium/fma_medium/000/000002.mp3"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_plots():
    print(f"Loading sample audio file: {SAMPLE_FILE}")
    if not os.path.exists(SAMPLE_FILE):
        print(f"Sample file {SAMPLE_FILE} not found!")
        return
        
    y, sr = load_audio(SAMPLE_FILE, duration=30)
    
    # Extract features
    print("Extracting features...")
    mel = extract_mel_spectrogram(y, sr)
    chroma = extract_chroma(y, sr)
    
    # Normalize features
    mel_norm = normalize_features(mel)
    chroma_norm = normalize_features(chroma)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Original Mel Spectrogram
    img1 = librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', ax=axes[0, 0])
    axes[0, 0].set_title("Original Log-Mel Spectrogram")
    fig.colorbar(img1, ax=axes[0, 0], format="%+2.0f dB")
    
    # 2. Normalized Mel Spectrogram
    img2 = librosa.display.specshow(mel_norm, sr=sr, x_axis='time', y_axis='mel', ax=axes[0, 1])
    axes[0, 1].set_title("Normalized Log-Mel Spectrogram")
    fig.colorbar(img2, ax=axes[0, 1])
    
    # 3. Original Chroma Feature
    img3 = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[1, 0])
    axes[1, 0].set_title("Original Chroma STFT")
    fig.colorbar(img3, ax=axes[1, 0])
    
    # 4. Normalized Chroma Feature
    img4 = librosa.display.specshow(chroma_norm, sr=sr, x_axis='time', y_axis='chroma', ax=axes[1, 1])
    axes[1, 1].set_title("Normalized Chroma STFT")
    fig.colorbar(img4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "mel_chroma_comparison.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Plots successfully generated and saved to: {output_path}")

if __name__ == "__main__":
    generate_plots()
