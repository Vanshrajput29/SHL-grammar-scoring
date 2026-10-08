"""Experiment (Kaggle GPU): other ways to cut clips into pieces for the audio model - 5 s and 15 s pieces, and
overlapping 10 s pieces every 5 s. Same embedding recipe as kaggle_audio_pieces/ (WavLM-base-plus, each piece on its
own, mean over time then layers). A final piece shorter than half a piece is merged into the previous one.
Output: audio_pieces_<variant>.npz with keys '<split>_<stem>' -> [n_pieces, 768].
"""
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

MODEL, SR = "microsoft/wavlm-base-plus", 16000
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
VARIANTS = {"p5": (5, 5), "p15": (15, 15), "p10hop5": (10, 5)}   # name: (piece seconds, hop seconds)


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


def pieces(x, piece_s, hop_s):
    piece, hop = piece_s * SR, hop_s * SR
    if len(x) <= piece:
        return [x]
    starts = list(range(0, len(x) - piece + 1, hop))
    segs = [(s, s + piece) for s in starts]
    if len(x) - segs[-1][1] >= piece // 2:          # leftover tail long enough to stand alone
        segs.append((len(x) - piece, len(x)) if hop < piece else (segs[-1][1], len(x)))
    else:                                           # otherwise extend the last piece to the end
        segs[-1] = (segs[-1][0], len(x))
    return [x[a:b] for a, b in segs]


fe = AutoFeatureExtractor.from_pretrained(MODEL)
model = WavLMModel.from_pretrained(MODEL).cuda().eval()
files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
out = {v: {} for v in VARIANTS}
for i, f in enumerate(files):
    x = load_wav(f)
    for v, (p_s, h_s) in VARIANTS.items():
        embs = []
        for seg in pieces(x, p_s, h_s):
            inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.cuda()
            with torch.no_grad():
                hs = torch.stack(model(inp, output_hidden_states=True).hidden_states)
            embs.append(hs[:, 0].mean(1).mean(0).cpu().numpy())
        out[v][f"{f.parent.name}_{f.stem}"] = np.stack(embs).astype(np.float32)
    if i % 200 == 0:
        print(i, len(files), f.name, {v: len(out[v][f"{f.parent.name}_{f.stem}"]) for v in VARIANTS}, flush=True)
for v in VARIANTS:
    np.savez_compressed(f"/kaggle/working/audio_pieces_{v}.npz", **out[v])
    print(v, "pieces:", sum(len(a) for a in out[v].values()))
