"""GPU feature extraction: mp3 -> log-mel (128) + chroma (12), 6 segments x 215 frames.

Decoding stays on CPU (soundfile); STFT, filterbanks, dB conversion and
normalisation run on the GPU. Replaces the librosa CPU path, whose
chroma_stft segfaults in this environment.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

AUDIO_DIR = "data/raw/fma_medium"
STORE = "data/processed/store"
SR, N_FFT, HOP = 22050, 2048, 512
SEGS, MELS, CHROMA, FRAMES = 6, 128, 12, 215
TOTAL_FRAMES = SEGS * FRAMES
CLIP = 30 * SR


def decode(path):
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        if y.shape[0] < sr:
            return None
        return y.mean(axis=1), sr
    except Exception:
        return None


def filterbanks(device):
    mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=MELS)
    chroma = librosa.filters.chroma(sr=SR, n_fft=N_FFT, n_chroma=CHROMA)
    return (torch.tensor(mel, device=device), torch.tensor(chroma, device=device))


def power_to_db(x, top_db=80.0):
    """Match librosa.power_to_db(ref=np.max) per item in the batch."""
    x = torch.clamp(x, min=1e-10)
    db = 10.0 * torch.log10(x)
    ref = db.amax(dim=(1, 2), keepdim=True)
    return torch.clamp(db - ref, min=-top_db)


def zscore(x):
    return (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-8)


def to_segments(x, bins):
    return x[:, :, :TOTAL_FRAMES].reshape(x.shape[0], bins, SEGS, FRAMES).permute(0, 2, 1, 3)


@torch.no_grad()
def process_batch(waves, srs, mel_fb, chroma_fb, window, device):
    out = torch.zeros(len(waves), CLIP, device=device)
    for sr in set(srs):
        idx = [i for i, s in enumerate(srs) if s == sr]
        block = torch.zeros(len(idx), max(len(waves[i]) for i in idx), device=device)
        for j, i in enumerate(idx):
            block[j, :len(waves[i])] = torch.from_numpy(waves[i]).to(device)
        res = AF.resample(block, sr, SR) if sr != SR else block
        n = min(res.shape[1], CLIP)
        out[idx, :n] = res[:, :n]

    spec = torch.stft(out, N_FFT, HOP, window=window, center=True,
                      pad_mode="constant", return_complex=True).abs() ** 2

    mel = zscore(power_to_db(mel_fb @ spec))
    raw = chroma_fb @ spec
    chroma = zscore(raw / (raw.abs().amax(dim=1, keepdim=True) + 1e-8))
    return (to_segments(mel, MELS).half().cpu().numpy(),
            to_segments(chroma, CHROMA).half().cpu().numpy())


def main(batch_size=48, readers=16):
    device = torch.device("cuda")
    os.makedirs(STORE, exist_ok=True)
    files = sorted(__import__("glob").glob(f"{AUDIO_DIR}/**/*.mp3", recursive=True))
    ids = [os.path.basename(f)[:-4] for f in files]
    print(f"{len(files)} files | device {torch.cuda.get_device_name(0)}", flush=True)

    mel_arr = np.lib.format.open_memmap(f"{STORE}/mel_f16.npy", mode="w+", dtype=np.float16,
                                        shape=(len(files), SEGS, MELS, FRAMES))
    ch_arr = np.lib.format.open_memmap(f"{STORE}/chroma_f16.npy", mode="w+", dtype=np.float16,
                                       shape=(len(files), SEGS, CHROMA, FRAMES))
    mel_fb, chroma_fb = filterbanks(device)
    window = torch.hann_window(N_FFT, device=device)

    failed, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(readers) as pool:
        for start in range(0, len(files), batch_size):
            chunk = files[start:start + batch_size]
            decoded = list(pool.map(decode, chunk))
            good = [(i, d) for i, d in enumerate(decoded) if d is not None]
            failed += [ids[start + i] for i, d in enumerate(decoded) if d is None]
            if not good:
                continue
            rows = [start + i for i, _ in good]
            mel, chroma = process_batch([d[0] for _, d in good], [d[1] for _, d in good],
                                        mel_fb, chroma_fb, window, device)
            mel_arr[rows] = mel
            ch_arr[rows] = chroma
            done += len(chunk)
            if start % (batch_size * 40) == 0 and done:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(files)}  {rate:.0f}/s  eta {(len(files)-done)/rate/60:.1f} min",
                      flush=True)

    mel_arr.flush(); ch_arr.flush()
    json.dump(ids, open(f"{STORE}/track_ids.json", "w"))
    json.dump(failed, open("results/extraction_failures.json", "w"), indent=2)
    print(f"done in {(time.time()-t0)/60:.1f} min | {len(failed)} failed | "
          f"{(mel_arr.nbytes + ch_arr.nbytes)/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main(batch_size=int(sys.argv[1]) if len(sys.argv) > 1 else 48)
