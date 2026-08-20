import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def load_audio(file_path, sr=22050, duration=5):
    """
    Load an audio file and resample it to the target sample rate.
    """

    y, sr = librosa.load(
        file_path,
        sr=sr,
        mono=True,
        duration=duration
    )

    return y, sr


def extract_mfcc(y, sr, n_mfcc=40):
    """
    Extract Mel-Frequency Cepstral Coefficients.
    """

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    return mfcc

def extract_mel_spectrogram(y, sr, n_mels=128):
    """
    Extract a Mel spectrogram from an audio waveform.
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db    

def plot_mel_spectrogram(mel_db, sr):
    """
    Display a Mel spectrogram.
    """

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()

def normalize_features(feature):
    """
    Normalize extracted audio features.
    """

    mean = np.mean(feature)
    std = np.std(feature)

    normalized = (feature - mean) / (std + 1e-8)

    return normalized