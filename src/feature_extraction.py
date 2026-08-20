import os
import glob
import numpy as np

from audio_features import load_audio, extract_mel_spectrogram, extract_chroma, normalize_features


# Paths
AUDIO_DIR = "data/raw/fma_medium/fma_medium"
OUTPUT_DIR = "data/processed/audio_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def segment_features(feature_matrix, num_segments=6):
    """
    Segment a feature matrix of shape [num_features, total_frames] along the time dimension.
    """
    total_frames = feature_matrix.shape[1]
    frames_per_segment = total_frames // num_segments
    segments = []
    for i in range(num_segments):
        start = i * frames_per_segment
        end = start + frames_per_segment
        segments.append(feature_matrix[:, start:end])
    return np.array(segments)


def extract_features_from_file(file_path):
    """
    Load one audio file, extract Mel spectrogram and Chroma features,
    normalize them per track, and segment them.
    """
    # Load full track (30s duration)
    y, sr = load_audio(file_path, duration=30)

    # Extract features for the entire track
    mel = extract_mel_spectrogram(y, sr)
    chroma = extract_chroma(y, sr)

    # Normalize per track
    mel_norm = normalize_features(mel)
    chroma_norm = normalize_features(chroma)

    # Segment into 6 segments (5s each for a 30s track)
    mel_segmented = segment_features(mel_norm, num_segments=6)
    chroma_segmented = segment_features(chroma_norm, num_segments=6)

    return mel_segmented, chroma_segmented


def process_audio_files(max_files=None):
    """
    Process audio files to extract features. If max_files is None, process all found files.
    """

    audio_files = glob.glob(
        os.path.join(AUDIO_DIR, "**", "*.mp3"),
        recursive=True
    )

    print(f"Found {len(audio_files)} audio files.")

    if max_files is not None:
        audio_files = audio_files[:max_files]
        print(f"Limiting execution to {max_files} files.")

    total_files = len(audio_files)
    print(f"Starting processing of {total_files} files...")

    for idx, file_path in enumerate(audio_files):
        try:
            # Print detailed log for small runs, or every 100 files for large runs
            is_verbose = (max_files is not None and max_files <= 10) or (idx % 100 == 0) or (idx == total_files - 1)
            
            if is_verbose:
                print(f"[{idx}/{total_files}] Processing: {file_path}")

            mel_seg, chroma_seg = extract_features_from_file(file_path)

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
                mel=mel_seg,
                chroma=chroma_seg,
                sample_rate=22050
            )

            if is_verbose:
                print(f"  -> Saved shapes - Mel: {mel_seg.shape}, Chroma: {chroma_seg.shape} to {output_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    # If this is run directly, process all files
    import sys
    max_to_process = int(sys.argv[1]) if len(sys.argv) > 1 else None
    process_audio_files(max_files=max_to_process)

    # Validate output on one file if exists
    test_file = "data/processed/audio_features/000002.npz"
    if os.path.exists(test_file):
        data = np.load(test_file)
        print("\n--- Validation ---")
        print("Keys:", data.files)
        print("Segmented Mel shape:", data["mel"].shape)
        print("Segmented Chroma shape:", data["chroma"].shape)