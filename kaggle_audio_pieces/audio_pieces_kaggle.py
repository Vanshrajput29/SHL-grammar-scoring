"""Experiment (Kaggle GPU): WavLM embeddings for every ~10 s piece of every clip.

Idea: the audio model learns from only 769 clips. Cutting each clip into ~10 s pieces (each inheriting the clip's
score) gives about 5x more training rows; the clip's prediction is the average over its pieces. Same embedding recipe
as kaggle_audio/ (WavLM-base-plus, mean over time, then over all 13 layers), but per piece. Each piece goes through
the model on its own, so nothing is ever padded. A final piece shorter than 5 s is merged into the previous one;
a clip shorter than 10 s is a single piece.
Output: audio_pieces.npz with keys '<split>_<stem>' -> array [n_pieces, 768].
Evaluation happens locally (cheap), splitting folds by CLIP so pieces of one clip never sit in both train and validation.
"""
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

MODEL, SR = "microsoft/wavlm-base-plus", 16000
PIECE, MIN_LAST = 10 * SR, 5 * SR
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


def pieces(x):
    bounds = list(range(0, len(x), PIECE)) + [len(x)]
    segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    if len(segs) > 1 and segs[-1][1] - segs[-1][0] < MIN_LAST:      # merge a short tail into the previous piece
        segs[-2:] = [(segs[-2][0], segs[-1][1])]
    return [x[a:b] for a, b in segs]


fe = AutoFeatureExtractor.from_pretrained(MODEL)
model = WavLMModel.from_pretrained(MODEL).cuda().eval()
out, n_total = {}, 0
files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
for i, f in enumerate(files):
    embs = []
    for seg in pieces(load_wav(f)):
        inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            hs = torch.stack(model(inp, output_hidden_states=True).hidden_states)    # [13, 1, T, 768]
        embs.append(hs[:, 0].mean(1).mean(0).cpu().numpy())                       # time mean, then layer mean
    out[f"{f.parent.name}_{f.stem}"] = np.stack(embs).astype(np.float32); n_total += len(embs)
    if i % 200 == 0:
        print(i, len(files), f.name, len(embs), "pieces", flush=True)
np.savez_compressed("/kaggle/working/audio_pieces.npz", **out)
print("done:", len(out), "clips,", n_total, "pieces")
