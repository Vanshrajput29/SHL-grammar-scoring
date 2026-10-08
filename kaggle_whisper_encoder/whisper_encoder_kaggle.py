"""Experiment (Kaggle GPU): Whisper's encoder as a second audio embedding next to WavLM.

Whisper large-v3-turbo's encoder was trained on far more speech than WavLM, and its features mix acoustics with
language. Per clip: split into 30 s windows (Whisper's input size), log-mel -> encoder, and average the output frames
that cover REAL audio only (Whisper pads every window to 30 s; averaging the padding would repeat the padding bug found in
the WavLM fine-tune). Embeddings for train and test are saved. Evaluated exactly like v6 (SVR(C=3, eps=0.05),
KFold(5, shuffle, 42), combined with DeBERTa-large OOF) with the Whisper embedding added to / replacing WavLM; the v6
baseline is recomputed first.
"""
import difflib, json, re, subprocess, sys, wave, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-whisper"], check=True)
import numpy as np
import pandas as pd
import torch
import whisper
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

INP = Path("/kaggle/input")
find = lambda name: next(INP.rglob(name))
DATA = find("train.csv").parent
SR, WIN = 16000, 30 * 16000
FRAMES_PER_WIN = 1500                      # encoder output frames per 30 s window (20 ms each)


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


model = whisper.load_model("turbo", device="cuda")
emb = {}
files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
for i, f in enumerate(files):
    x, sums, n = load_wav(f), 0, 0
    for s in range(0, len(x), WIN):
        seg = x[s:s + WIN]
        if len(seg) < SR:                  # skip a final piece under 1 s
            continue
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(seg)), n_mels=model.dims.n_mels).cuda()
        with torch.no_grad():
            out = model.encoder(mel[None].half() if next(model.encoder.parameters()).dtype == torch.float16 else mel[None])
        real = max(1, int(round(len(seg) / WIN * FRAMES_PER_WIN)))   # frames that cover actual audio
        sums = sums + out[0, :real].float().sum(0).cpu().numpy(); n += real
    emb[f"{f.parent.name}_{f.stem}"] = (sums / n).astype(np.float32)
    if i % 200 == 0:
        print(i, len(files), f.name, flush=True)
np.savez_compressed("/kaggle/working/whisper_enc_emb.npz", **emb)
del model; torch.cuda.empty_cache()

# ---- evaluation (same hand-made features as features.py; same model and folds as v6) ----------------------------------
train = pd.read_csv(DATA / "train.csv")
y = train.label.to_numpy(); ok = y > 0
keys = ["train_" + Path(f).stem for f in train.filename]
with zipfile.ZipFile(find("transcripts.zip")) as z:
    recs = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()}
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "er", "ah", "like"}
norm_words = lambda s: re.findall(r"[a-z0-9']+", s.lower())
gec = json.loads(find("gec.json").read_text())


def base_features(rec, pairs):
    text, seg = rec["text"], rec["segments"]
    words = re.findall(r"[a-z']+", text.lower()); n = max(len(words), 1)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    f = {"n_words": len(words), "words_per_sec": len(words) / rec["duration"],
         "mean_sentence_len": len(words) / max(len(sentences), 1), "type_token_ratio": len(set(words)) / n,
         "filler_rate": sum(w in FILLERS for w in words) / n,
         "repeat_rate": sum(a == b for a, b in zip(words, words[1:])) / n,
         "mean_word_len": np.mean([len(w) for w in words]) if words else 0.0,
         "non_ascii_rate": sum(ord(c) > 127 for c in text) / max(len(text), 1),
         "avg_logprob": np.mean([s["avg_logprob"] for s in seg]) if seg else -2.0,
         "no_speech_prob": np.mean([s["no_speech_prob"] for s in seg]) if seg else 1.0, "rms": rec["rms"]}
    edits = nw = changed = 0
    for orig, corr in pairs:
        a, b = norm_words(orig), norm_words(corr)
        e = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
                difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal")
        edits, nw, changed = edits + e, nw + len(a), changed + (e > 0)
    f.update(gec_edit_rate=edits / max(nw, 1), gec_changed_frac=changed / max(len(pairs), 1))
    return f


H = pd.DataFrame([base_features(recs[k], gec[k]) for k in keys]).to_numpy()
A_npz = np.load(find("audio_emb.npz")); A = np.stack([A_npz[k].mean(0) for k in keys])
W = np.stack([emb[k] for k in keys])
deb = train.filename.map(pd.read_csv(find("oof_deberta_large.csv")).set_index("filename").iloc[:, -1]).to_numpy()
cv = KFold(5, shuffle=True, random_state=42)
rmse = lambda p, m=slice(None): float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
combine = lambda a, t: np.where((a < 1) | np.isnan(t), a, 0.5 * a + 0.5 * np.nan_to_num(t))


def svr(X, pca=None):
    return np.clip(cross_val_predict(make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05)), X, y, cv=cv), 0, 5)


print(f"\nWhisper encoder embedding: {W.shape[1]} dims (WavLM: {A.shape[1]})")
print(f"{'audio model inputs':<40} {'audio SVR':>10} {'ENSEMBLE all':>13} {'1-5':>7} {'gate zeros<1':>13}")
for name, X, pca in [("v6: WavLM + hand-made (sanity check)", np.hstack([A, H]), None),
                     ("Whisper-enc + hand-made", np.hstack([W, H]), None),
                     ("WavLM + Whisper-enc + hand-made", np.hstack([A, W, H]), None)]:
    a = svr(X, pca); e = combine(a, deb)
    print(f"{name:<40} {rmse(a):>10.4f} {rmse(e):>13.4f} {rmse(e, ok):>7.4f} {int((a[~ok] < 1).sum()):>8}/{int((~ok).sum())}", flush=True)
print("expected v6 baseline: audio SVR 0.5359 | ensemble 0.4963")
