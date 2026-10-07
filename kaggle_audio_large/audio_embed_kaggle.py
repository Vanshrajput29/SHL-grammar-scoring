"""Mean-pooled WavLM hidden states per clip (Kaggle GPU). Captures fluency/pronunciation that transcripts lose.
Output: audio_emb_large.npz with keys = '<split>_<stem>', value = [1024] (mean over time, then over layers)."""
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

MODEL = "microsoft/wavlm-large"
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
CHUNK = 16000 * 20  # 20 s windows keep memory bounded on long clips


def load_wav(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


fe = AutoFeatureExtractor.from_pretrained(MODEL)
model = WavLMModel.from_pretrained(MODEL).cuda().eval()
out = {}
files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
for i, f in enumerate(files):
    x = load_wav(f)
    sums, n = None, 0
    for s in range(0, len(x), CHUNK):
        seg = x[s:s + CHUNK]
        if len(seg) < 16000:  # skip <1 s tail
            continue
        inp = fe(seg, sampling_rate=16000, return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            hs = torch.stack(model(inp, output_hidden_states=True).hidden_states)  # [L, 1, T, 768]
        layer_sum = hs[:, 0].sum(1).cpu().numpy()
        sums = layer_sum if sums is None else sums + layer_sum
        n += hs.shape[2]
    out[f"{f.parent.name}_{f.stem}"] = (sums / n).mean(0).astype(np.float32)
    if i % 100 == 0:
        print(i, len(files), flush=True)
np.savez_compressed("/kaggle/working/audio_emb_large.npz", **out)
print("done", len(out))
