"""Experiment (Kaggle GPU): does "what Whisper corrected" help predict grammar scores?

Whisper writes clean, fluent text; a CTC model without a language model (wav2vec2, Apache-2.0) writes down what it
hears, mistakes included. Where the two transcripts disagree is roughly what Whisper smoothed over. Steps:
  1. transcribe every clip with facebook/wav2vec2-large-960h-lv60-self (greedy CTC, no language model)
  2. normalise both transcripts with Whisper's EnglishTextNormalizer (numbers, contractions, spelling) so that
     "11" vs "ELEVEN" doesn't count as a disagreement
  3. disagreement features: wav2vec2 words that differ from / are missing in / are extra to Whisper's, per word
  4. evaluate exactly like v6: SVR(C=3, eps=0.05) on [WavLM + hand-made features], KFold(5, shuffle, 42), combined
     with the DeBERTa-large out-of-fold predictions; the v6 baseline is recomputed first as a sanity check
Inputs come from earlier Kaggle runs (kernel_sources), so nothing is uploaded.
"""
import difflib, json, re, subprocess, sys, wave, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-whisper"], check=True)  # only for its normaliser
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from whisper.normalizers import EnglishTextNormalizer

INP = Path("/kaggle/input")
find = lambda name: next(INP.rglob(name))
DATA = find("train.csv").parent
CTC_MODEL, SR, CHUNK = "facebook/wav2vec2-large-960h-lv60-self", 16000, 30 * 16000


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


# ---- 1. wav2vec2 transcripts (train + test) -----------------------------------------------------------------------
proc = Wav2Vec2Processor.from_pretrained(CTC_MODEL)
ctc = Wav2Vec2ForCTC.from_pretrained(CTC_MODEL).cuda().eval()
ctc_text = {}
files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
for i, f in enumerate(files):
    x, parts = load_wav(f), []
    for s in range(0, len(x), CHUNK):           # 30 s chunks keep memory bounded; a word cut at a boundary is rare
        seg = x[s:s + CHUNK]
        if len(seg) < SR // 2:
            continue
        inp = proc(seg, sampling_rate=SR, return_tensors="pt").input_values.cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ids = ctc(inp).logits.argmax(-1)
        parts.append(proc.batch_decode(ids)[0])
    ctc_text[f"{f.parent.name}_{f.stem}"] = " ".join(parts)
    if i % 100 == 0:
        print(i, len(files), f.name, ctc_text[f"{f.parent.name}_{f.stem}"][:90], flush=True)
Path("/kaggle/working/ctc_transcripts.json").write_text(json.dumps(ctc_text))
del ctc; torch.cuda.empty_cache()

# ---- 2-3. disagreement features ------------------------------------------------------------------------------------
train = pd.read_csv(DATA / "train.csv")
y = train.label.to_numpy(); ok = y > 0
keys = ["train_" + Path(f).stem for f in train.filename]
with zipfile.ZipFile(find("transcripts.zip")) as z:
    recs = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()}
normalise = EnglishTextNormalizer()


def disagreement(whisper_text, ctc_words_text):
    a, b = normalise(whisper_text).split(), normalise(ctc_words_text).split()
    sub = miss = extra = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        la, lb = i2 - i1, j2 - j1
        if op == "replace": sub += min(la, lb); extra += max(0, lb - la); miss += max(0, la - lb)
        elif op == "insert": extra += lb      # heard by wav2vec2, left out by Whisper (e.g. repeats, fillers)
        elif op == "delete": miss += la       # written by Whisper, not heard by wav2vec2 (Whisper "filled in")
    n = max(len(a), 1)
    return {"dis_sub_rate": sub / n, "dis_extra_rate": extra / n, "dis_missing_rate": miss / n,
            "dis_total_rate": (sub + extra + miss) / n, "ctc_words_ratio": len(b) / n}


dis = pd.DataFrame([disagreement(recs[k]["text"], ctc_text[k]) for k in keys])
for i in range(2):
    k = keys[int(np.argsort(-dis.dis_total_rate.to_numpy() * ok)[i])]
    print(f"\nexample {k} (one of the 2 clips with the MOST disagreement - worst cases, not typical):\n  whisper : {normalise(recs[k]['text'])[:160]}\n  wav2vec2: {normalise(ctc_text[k])[:160]}")

# ---- 4. evaluation (same hand-made features as features.py; same model and folds as v6) -----------------------------
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


print("\nSpearman with score (clips scored 1-5):")
for col in dis.columns:
    print(f"  {col:<18} {spearman(dis[col]):+.3f}")
print(f"\n{'feature set':<30} {'hand-made Ridge':>15} {'audio SVR':>10} {'ENSEMBLE all':>13} {'1-5':>7}")
for name, H in [("v6 baseline (sanity check)", base), ("+ disagreement features", pd.concat([base, dis], axis=1))]:
    a = svr(np.hstack([A, H.to_numpy()])); e = combine(a, deb)
    print(f"{name:<30} {rmse(ridge(H.to_numpy())):>15.4f} {rmse(a):>10.4f} {rmse(e):>13.4f} {rmse(e, ok):>7.4f}", flush=True)
print("expected v6 baseline: hand-made Ridge 0.9657 | audio SVR 0.5359 | ensemble 0.4963")
pd.concat([pd.Series(keys, name="key"), dis], axis=1).to_csv("/kaggle/working/disagreement_features.csv", index=False)
