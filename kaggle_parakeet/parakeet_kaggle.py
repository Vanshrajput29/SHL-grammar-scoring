"""Experiment (Kaggle GPU): NVIDIA Parakeet transcripts (CC-BY-4.0) - verbatim and accurate on accented speech?

wav2vec2 failed because it couldn't understand this accented, spontaneous speech, so its "disagreements" with Whisper
were its own recognition errors. Parakeet is trained on far more varied speech. Two models:
  - nvidia/parakeet-tdt-0.6b-v2 : very accurate, punctuated
  - nvidia/parakeet-ctc-1.1b    : greedy CTC without a language model, so it cannot smooth grammar
For each: sanity-print transcripts next to Whisper's, then add (a) disagreement-with-Whisper features and
(b) hand-made text features computed on the Parakeet transcript, and evaluate exactly like v6 (SVR(C=3, eps=0.05)
on [WavLM + hand-made], KFold(5, shuffle, 42), combined with DeBERTa-large OOF). v6 baseline recomputed first.
"""
import difflib, json, re, subprocess, sys, tempfile, wave, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nemo_toolkit[asr]", "openai-whisper"], check=True)
import numpy as np
import pandas as pd
import nemo.collections.asr as nemo_asr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from whisper.normalizers import EnglishTextNormalizer
import nemo
print("nemo", nemo.__version__, flush=True)

INP = Path("/kaggle/input")
find = lambda name: next(INP.rglob(name))
DATA = find("train.csv").parent
train = pd.read_csv(DATA / "train.csv")
y = train.label.to_numpy(); ok = y > 0
keys = ["train_" + Path(f).stem for f in train.filename]
with zipfile.ZipFile(find("transcripts.zip")) as z:
    recs = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()}

# Parakeet expects 16 kHz mono files; write mono copies where a clip has more than one channel.
tmp = Path(tempfile.mkdtemp()); paths = []
for f in train.filename:
    src = DATA / "train" / f
    with wave.open(str(src)) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, src
        if w.getnchannels() == 1:
            paths.append(str(src)); continue
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels()).mean(1).astype(np.int16)
    out = tmp / f
    with wave.open(str(out), "wb") as o:
        o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000); o.writeframes(x.tobytes())
    paths.append(str(out))


def transcribe(model_name):
    m = nemo_asr.models.ASRModel.from_pretrained(model_name).cuda().eval()
    hyps = m.transcribe(paths, batch_size=16)
    if isinstance(hyps, tuple):                 # some NeMo versions return (best, all)
        hyps = hyps[0]
    texts = [h.text if hasattr(h, "text") else str(h) for h in hyps]
    del m
    return dict(zip(keys, texts))


normalise = EnglishTextNormalizer()
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "er", "ah", "like"}


def disagreement(whisper_text, other_text, p):
    a, b = normalise(whisper_text).split(), normalise(other_text).split()
    sub = miss = extra = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        la, lb = i2 - i1, j2 - j1
        if op == "replace": sub += min(la, lb); extra += max(0, lb - la); miss += max(0, la - lb)
        elif op == "insert": extra += lb
        elif op == "delete": miss += la
    n = max(len(a), 1)
    return {f"{p}_dis_sub": sub / n, f"{p}_dis_extra": extra / n, f"{p}_dis_missing": miss / n,
            f"{p}_dis_total": (sub + extra + miss) / n}


def text_feats(text, duration, p):
    words = re.findall(r"[a-z']+", text.lower()); n = max(len(words), 1)
    return {f"{p}_words_per_sec": len(words) / duration, f"{p}_filler_rate": sum(w in FILLERS for w in words) / n,
            f"{p}_repeat_rate": sum(a == b for a, b in zip(words, words[1:])) / n,
            f"{p}_type_token_ratio": len(set(words)) / n}


# ---- v6 baseline features (exact copy of features.py) ----------------------------------------------------------------
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


base = pd.DataFrame([base_features(recs[k], gec[k]) for k in keys])
A_npz = np.load(find("audio_emb.npz")); A = np.stack([A_npz[k].mean(0) for k in keys])
deb = train.filename.map(pd.read_csv(find("oof_deberta_large.csv")).set_index("filename").iloc[:, -1]).to_numpy()
cv = KFold(5, shuffle=True, random_state=42)
rmse = lambda p, m=slice(None): float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
svr = lambda X: np.clip(cross_val_predict(make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05)), X, y, cv=cv), 0, 5)
ridge = lambda X: np.clip(cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))), X, y, cv=cv), 0, 5)
combine = lambda a, t: np.where((a < 1) | np.isnan(t), a, 0.5 * a + 0.5 * np.nan_to_num(t))


def spearman(values):
    """Spearman with the score on clips scored 1-5, paired by position. (The first version used
    v[ok].corr(pd.Series(y[ok])): pandas aligned the two by index labels, which differ after filtering,
    so it paired the wrong rows and printed near-zero correlations.)"""
    return pd.Series(np.asarray(values, dtype=float)[ok]).corr(pd.Series(y[ok]), method="spearman")



def report(name, H):
    a = svr(np.hstack([A, H.to_numpy()])); e = combine(a, deb)
    print(f"{name:<38} {rmse(ridge(H.to_numpy())):>15.4f} {rmse(a):>10.4f} {rmse(e):>13.4f} {rmse(e, ok):>7.4f}", flush=True)


print(f"\n{'feature set':<38} {'hand-made Ridge':>15} {'audio SVR':>10} {'ENSEMBLE all':>13} {'1-5':>7}")
report("v6 baseline (sanity check)", base)
print("expected v6 baseline: hand-made Ridge 0.9657 | audio SVR 0.5359 | ensemble 0.4963", flush=True)

for p, model_name in [("tdt", "nvidia/parakeet-tdt-0.6b-v2"), ("ctc", "nvidia/parakeet-ctc-1.1b")]:
    try:
        texts = transcribe(model_name)
    except Exception as ex:                      # report and carry on with the other model
        print(f"\n{model_name} FAILED: {type(ex).__name__}: {ex}", flush=True); continue
    Path(f"/kaggle/working/{p}_transcripts.json").write_text(json.dumps(texts))
    print(f"\n=== {model_name} ===")
    for k in keys[:2] + [keys[i] for i in np.where(~ok)[0][:1]]:
        print(f"  {k}\n    whisper : {normalise(recs[k]['text'])[:150]}\n    {p:<8}: {normalise(texts[k])[:150]}")
    dis = pd.DataFrame([disagreement(recs[k]["text"], texts[k], p) for k in keys])
    tf = pd.DataFrame([text_feats(texts[k], recs[k]["duration"], p) for k in keys])
    new = pd.concat([dis, tf], axis=1)
    print("  Spearman with score (clips 1-5):", {c: round(spearman(new[c]), 3) for c in new.columns})
    report(f"+ {p} disagreement", pd.concat([base, dis], axis=1))
    report(f"+ {p} text features", pd.concat([base, tf], axis=1))
    report(f"+ {p} both", pd.concat([base, new], axis=1))
