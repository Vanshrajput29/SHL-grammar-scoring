"""Experiment (Kaggle GPU): speed-perturbed copies of the TRAINING clips for the piece-based audio model.

Speed perturbation (as in Kaldi/ASR training) re-samples the audio so it plays 0.9x or 1.1x as fast, changing tempo and
pitch together. Each perturbed copy is cut into the same overlapping 10 s pieces as the final model (one every 5 s;
same rule as transcribe.split_pieces) and embedded with WavLM-base-plus exactly like kaggle_audio_pieces_variants/.
Only training clips are perturbed; evaluation always uses the original audio, and a clip's copies only ever join the
training side of a fold.
Output: audio_pieces_speed.npz with keys 'train_<stem>@0.9' / 'train_<stem>@1.1' -> [n_pieces, 768], plus the whole-clip
embedding of each copy under 'clip:train_<stem>@<speed>' -> [768] (mean over time in 20 s chunks, then layers).
"""
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMModel

MODEL, SR = "microsoft/wavlm-base-plus", 16000
SPEEDS = {0.9: (10, 9), 1.1: (10, 11)}        # speed: (up, down) for resample_poly; played back at 16 kHz
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


def split_pieces(x, piece=10 * SR, hop=5 * SR):     # identical rule to transcribe.split_pieces
    if len(x) <= piece:
        return [x]
    segs = [(s, s + piece) for s in range(0, len(x) - piece + 1, hop)]
    if len(x) - segs[-1][1] >= piece // 2:
        segs.append((len(x) - piece, len(x)))
    else:
        segs[-1] = (segs[-1][0], len(x))
    return [x[a:b] for a, b in segs]


fe = AutoFeatureExtractor.from_pretrained(MODEL)
model = WavLMModel.from_pretrained(MODEL).cuda().eval()


def hidden(seg):
    with torch.no_grad():
        return torch.stack(model(fe(seg, sampling_rate=SR, return_tensors="pt").input_values.cuda(),
                                 output_hidden_states=True).hidden_states)[:, 0]          # [13, T, 768]


def clip_embedding(x):                               # same as kaggle_audio/: 20 s chunks, time mean, then layer mean
    sums, n = None, 0
    for s in range(0, len(x), 20 * SR):
        seg = x[s:s + 20 * SR]
        if len(seg) < SR:
            continue
        h = hidden(seg); ls = h.sum(1).cpu().numpy()
        sums, n = (ls if sums is None else sums + ls), n + h.shape[1]
    return (sums / n).mean(0)


out = {}
files = sorted((DATA / "train").glob("*.wav"))
for i, f in enumerate(files):
    x = load_wav(f)
    for speed, (up, down) in SPEEDS.items():
        xs = resample_poly(x, up, down).astype(np.float32)
        key = f"train_{f.stem}@{speed}"
        out[key] = np.stack([hidden(p).mean(1).mean(0).cpu().numpy() for p in split_pieces(xs)]).astype(np.float32)
        out["clip:" + key] = clip_embedding(xs).astype(np.float32)
    if i % 100 == 0:
        print(i, len(files), f.name, {s: len(out[f"train_{f.stem}@{s}"]) for s in SPEEDS}, flush=True)
np.savez_compressed("/kaggle/working/audio_pieces_speed.npz", **out)
print("done:", len(files), "clips x", len(SPEEDS), "speeds")
