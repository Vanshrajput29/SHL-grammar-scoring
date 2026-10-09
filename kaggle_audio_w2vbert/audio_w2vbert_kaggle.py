"""Experiment (Kaggle GPU): w2v-BERT 2.0 embeddings for the same overlapping 10 s pieces as the final audio model (v10).

w2v-BERT 2.0 (facebook/w2v-bert-2.0, ~600M parameters) is Meta's speech encoder from the Seamless project, pre-trained on
~4.5M hours of audio (WavLM-base-plus: ~94k hours). The idea: a stronger encoder in the setup that has worked (pieces + SVR),
either instead of WavLM or averaged with it.
Same pieces as transcribe.split_pieces (10 s, one every 5 s). Each piece goes through the model on its own (no padding).
Saved per layer (mean over time, float16) so the layer choice can be checked locally:
'<split>_<stem>' -> [n_pieces, 25, 1024], and the whole clip (20 s chunks, time mean) under 'clip:<split>_<stem>' -> [25, 1024].
"""
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

MODEL, SR = "facebook/w2v-bert-2.0", 16000
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
model = Wav2Vec2BertModel.from_pretrained(MODEL).cuda().eval()


def layer_means(seg):
    """[25 layers, 1024]: hidden states averaged over time."""
    inp = fe(seg, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        hs = model(input_features=inp.input_features.cuda(), output_hidden_states=True).hidden_states
    return torch.stack(hs)[:, 0].mean(1).cpu().numpy(), torch.stack(hs).shape[2]


out = {}
files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
for i, f in enumerate(files):
    x, key = load_wav(f), f"{f.parent.name}_{f.stem}"
    out[key] = np.stack([layer_means(p)[0] for p in split_pieces(x)]).astype(np.float16)
    sums, n = 0, 0
    for s in range(0, len(x), 20 * SR):                 # whole clip, same 20 s chunking as kaggle_audio/
        seg = x[s:s + 20 * SR]
        if len(seg) < SR:
            continue
        m, t = layer_means(seg); sums, n = sums + m * t, n + t
    out["clip:" + key] = (sums / n).astype(np.float16)
    if i % 100 == 0:
        print(i, len(files), f.name, out[key].shape, flush=True)
np.savez_compressed("/kaggle/working/audio_pieces_w2vbert.npz", **out)
print("done:", len(files), "clips")
