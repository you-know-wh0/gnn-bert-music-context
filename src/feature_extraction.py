import os
import glob
import numpy as np

from audio_features import load_audio, extract_mfcc, extract_mel_spectrogram, normalize_features


# Paths
AUDIO_DIR = "data/raw/fma_medium/fma_medium"
OUTPUT_DIR = "data/processed/audio_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_features_from_file(file_path):
    """
    Load one audio file and extract MFCC and Mel spectrogram features.
    """

    # Load audio
    y, sr = load_audio(file_path)

    # Extract MFCC
    mfcc = extract_mfcc(y, sr)

    # Extract Mel spectrogram
    mel = extract_mel_spectrogram(y, sr)

    return mfcc, mel


def process_audio_files(max_files=5):
    """
    Process a small number of audio files for testing.
    """

    audio_files = glob.glob(
        os.path.join(AUDIO_DIR, "**", "*.mp3"),
        recursive=True
    )

    print(f"Found {len(audio_files)} audio files.")

    # Process only a few files while testing
    audio_files = audio_files[:max_files]

    for file_path in audio_files:

        try:
            print(f"\nProcessing: {file_path}")

            mfcc, mel = extract_features_from_file(file_path)
            mfcc = normalize_features(mfcc)
            mel = normalize_features(mel)

            # Get filename without extension
            filename = os.path.splitext(
                os.path.basename(file_path)
            )[0]

            # Save extracted features
            output_path = os.path.join(
                OUTPUT_DIR,
                filename + ".npz"
            )

            np.savez(
                output_path,
                mfcc=mfcc,
                mel=mel,
                sample_rate=22050
            )

            print("MFCC shape:", mfcc.shape)
            print("Mel shape:", mel.shape)
            print("Saved to:", output_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    process_audio_files(max_files=5)


    test_file = "data/processed/audio_features/000002.npz"

    data = np.load(test_file)

    print("Keys:", data.files)
    print("MFCC shape:", data["mfcc"].shape)
    print("Mel shape:", data["mel"].shape)
    print("Sample rate:", data["sample_rate"])
    print("MFCC mean:", data["mfcc"].mean())
    print("MFCC std:", data["mfcc"].std())