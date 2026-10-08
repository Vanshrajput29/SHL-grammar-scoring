"""Experiment (Kaggle CPU): can stronger regularization of the audio model (SVR) reduce overfitting and improve CV?

The v6 SVR fits training clips far better than unseen ones (train RMSE 0.109 vs CV 0.536). Nested CV already picked
C=3 over the more regularized C=1, but two knobs were never tuned:
  - gamma: how sharply the RBF kernel bends (default 'scale'); smaller gamma = smoother model
  - input dimensionality: 768 WavLM numbers + 13 features for 769 clips; PCA inside the pipeline (fitted on training
    folds only, so nothing leaks) can compress WavLM before the SVR
Everything is nested: inner 5-fold grid search picks the settings on the outer-train folds; outer folds are scored.
Then the same models inside the v6 ensemble (combined with DeBERTa-large OOF). v6 baseline recomputed first.
"""
import difflib, json, re, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

INP = Path("/kaggle/input")
find = lambda name: next(INP.rglob(name))
DATA = find("train.csv").parent
train = pd.read_csv(DATA / "train.csv")
y = train.label.to_numpy(); ok = y > 0
keys = ["train_" + Path(f).stem for f in train.filename]
with zipfile.ZipFile(find("transcripts.zip")) as z:
    recs = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()}
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "er", "ah", "like"}
norm_words = lambda s: re.findall(r"[a-z0-9']+", s.lower())
gec = json.loads(find("gec.json").read_text())


def base_features(rec, pairs):           # exact copy of features.py
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
X = np.hstack([A, H]); n_audio = A.shape[1]
deb = train.filename.map(pd.read_csv(find("oof_deberta_large.csv")).set_index("filename").iloc[:, -1]).to_numpy()
outer = KFold(5, shuffle=True, random_state=42)
inner = KFold(5, shuffle=True, random_state=1)
rmse = lambda p, m=slice(None): float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
combine = lambda a, t: np.where((a < 1) | np.isnan(t), a, 0.5 * a + 0.5 * np.nan_to_num(t))


def pca_svr():
    """Scale everything, compress only the WavLM block with PCA (fitted inside each training fold), then SVR."""
    pre = ColumnTransformer([("audio", make_pipeline(StandardScaler(), PCA(random_state=0)), list(range(n_audio))),
                             ("hand", StandardScaler(), list(range(n_audio, X.shape[1])))])
    return Pipeline([("pre", pre), ("post", StandardScaler()), ("svr", SVR())])


experiments = [
    ("v6: SVR C=3, eps=0.05, gamma=scale (fixed)", make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05)), None),
    ("tune C, eps, gamma (nested)", Pipeline([("sc", StandardScaler()), ("svr", SVR())]),
     {"svr__C": [1, 3, 10], "svr__epsilon": [0.05, 0.1], "svr__gamma": ["scale", 3e-4, 6e-4, 1.2e-3, 2.5e-3]}),
    ("PCA on WavLM + tune C, gamma (nested)", pca_svr(),
     {"pre__audio__pca__n_components": [32, 64, 128, 256], "svr__C": [1, 3, 10], "svr__epsilon": [0.05],
      "svr__gamma": ["scale", 0.003, 0.01]}),
]
print(f"{'audio model':<44} {'train RMSE':>10} {'CV RMSE':>8} {'ENSEMBLE all':>13} {'1-5':>7}  settings chosen per outer fold")
for name, est, grid in experiments:
    oof, chosen, train_err = np.zeros(len(y)), [], []
    for tr, va in outer.split(X):
        m = (GridSearchCV(est, grid, cv=inner, scoring="neg_root_mean_squared_error", n_jobs=-1) if grid else est).fit(X[tr], y[tr])
        oof[va] = np.clip(m.predict(X[va]), 0, 5)
        train_err.append(float(np.sqrt(np.mean((y[tr] - np.clip(m.predict(X[tr]), 0, 5)) ** 2))))
        chosen.append(m.best_params_ if grid else "-")
    e = combine(oof, deb)
    print(f"{name:<44} {np.mean(train_err):>10.4f} {rmse(oof):>8.4f} {rmse(e):>13.4f} {rmse(e, ok):>7.4f}  "
          f"gate {int((oof[~ok] < 1).sum())}/37", flush=True)
    for c in chosen:
        print("      ", {k.split("__")[-1]: v for k, v in c.items()} if isinstance(c, dict) else c, flush=True)
print("expected v6 baseline: audio CV 0.5359 | ensemble 0.4963")
