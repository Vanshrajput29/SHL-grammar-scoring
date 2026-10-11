"""Kaggle CPU: make the audio model depend less on WHO is speaking, judged by speaker-grouped CV.

Under honest folds the audio model generalises much worse (0.60 vs 0.52 with leaky folds), so it was leaning on voice
identity. Three untested ways to lean on it less:
  A. weight the 13 hand-made grammar/fluency features up: in the RBF kernel they are 13 of 1,549 numbers, the rest WavLM
     (which encodes the speaker), so they barely move the similarity;
  B. take the speaker fingerprint out of the piece rows: drop the whole-clip embedding, or subtract it from each piece;
  C. later WavLM layers (closer to words than to voice): clip-level model on layer subsets (per-layer clip embeddings).
Every variant is the final ensemble (gate + 50/50 with the speaker-fold DeBERTa) on the 1-5 clips, 3 GroupKFold seeds.
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

IN = Path("/kaggle/input")
Z = np.load(next(IN.rglob("eval_inputs.npz"))); train = pd.read_csv(next(IN.rglob("train.csv")))
y, H, A = Z["y"], Z["H"], Z["A"]; n = len(y); pos = y > 0; P = [Z[f"p_{i}"] for i in range(n)]
groups = pd.read_csv(next(IN.rglob("speaker_groups.csv"))).speaker_group.to_numpy()
deb = train.filename.map(pd.read_csv(next(IN.rglob("oof_deberta_speaker.csv"))).set_index("filename").deberta_speaker).to_numpy()
L = np.load(next(IN.rglob("audio_emb.npz"))); AL = np.stack([L[f"train_{f[:-4]}"] for f in train.filename])   # [n, 13, 768]
SEEDS = [42, 1, 2]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
FOLDS = {s: list(GroupKFold(5, shuffle=True, random_state=s).split(y, groups=groups)) for s in SEEDS}


def ens(a):
    e = np.where(a < 1, a, 0.5 * a + 0.5 * deb); return rmse(y[pos], e[pos])


def svr_oof(make_rows, counts, seed, hw=1.0, gs=0.5, nh=13):
    """Generic pieces/clip SVR (v15 settings). make_rows(clips) -> rows whose LAST nh columns are hand-made features,
    which get weight hw after standardising. counts[i] = rows per clip (median over them)."""
    out = np.zeros(n)
    for a, b in FOLDS[seed]:
        X = make_rows(a); yy = np.concatenate([np.repeat(y[i], counts[i]) for i in a])
        sc = StandardScaler().fit(X); w = np.ones(X.shape[1]); w[-nh:] = hw
        Zx = sc.transform(X) * w; g = gs / (Zx.shape[1] * Zx.var())
        svr = SVR(kernel="precomputed", C=1, epsilon=0.05).fit(rbf_kernel(Zx, gamma=g), yy)
        p = rbf_kernel(sc.transform(make_rows(b)) * w, Zx[svr.support_], gamma=g) @ svr.dual_coef_[0] + svr.intercept_[0]
        out[b] = np.clip([np.median(s) for s in np.split(p, np.cumsum([counts[i] for i in b])[:-1])], 0, 5)
    return out


counts = [len(p) for p in P]
pieces = {"v15: [piece, clip, hand]": lambda c: np.vstack([np.concatenate([p, A[i], H[i]]) for i in c for p in P[i]]),
          "[piece, hand] (no clip embedding)": lambda c: np.vstack([np.concatenate([p, H[i]]) for i in c for p in P[i]]),
          "[piece - clip, hand]": lambda c: np.vstack([np.concatenate([p - A[i], H[i]]) for i in c for p in P[i]]),
          "[piece - clip, clip, hand]": lambda c: np.vstack([np.concatenate([p - A[i], A[i], H[i]]) for i in c for p in P[i]])}
LAYERS = {"all 13 (v6 style)": slice(0, 13), "0-4": slice(0, 5), "4-8": slice(4, 9), "6-12": slice(6, 13), "8-12": slice(8, 13), "10-12": slice(10, 13)}


def job(cfg):
    kind, name, hw = cfg
    if kind == "pieces":
        return cfg, [svr_oof(pieces[name], counts, s, hw) for s in SEEDS]
    Ac = AL[:, LAYERS[name]].mean(1)
    return cfg, [svr_oof(lambda c: np.hstack([Ac[c], H[c]]), [1] * n, s, hw) for s in SEEDS]


cfgs = [("pieces", "v15: [piece, clip, hand]", hw) for hw in (1, 2, 4, 8, 16)]
cfgs += [("pieces", k, 1) for k in list(pieces)[1:]] + [("pieces", k, 4) for k in list(pieces)[1:]]
cfgs += [("clip", k, hw) for k in LAYERS for hw in (1, 4)]
with Pool(4) as pool:
    res = dict(pool.map(job, cfgs))
base = [ens(o) for o in res[("pieces", "v15: [piece, clip, hand]", 1)]]
print(f"v15 reference: {np.mean(base):.4f} {np.round(base, 4)}\n")
print(f"{'variant':<58} {'ensemble 1-5':>12}  {'audio all':>9}  per seed               better than v15")
for cfg, o in sorted(res.items(), key=lambda kv: np.mean([ens(x) for x in kv[1]])):
    e = [ens(x) for x in o]
    print(f"{str(cfg):<58} {np.mean(e):12.4f}  {np.mean([rmse(y, x) for x in o]):9.4f}  {np.round(e, 4)}  {sum(a < b for a, b in zip(e, base))}/3")
np.savez("/kaggle/working/speaker_invariant_oof.npz", **{"|".join(map(str, k)): np.stack(v) for k, v in res.items()})
