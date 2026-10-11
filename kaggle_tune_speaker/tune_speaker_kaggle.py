"""Kaggle CPU: re-tune the audio model with SPEAKER-grouped CV (folds never split a speaker's answers).

Every setting so far (SVR C / epsilon / kernel width, pieces vs one row per clip) was chosen with folds split by clip,
where a model could partly score a clip by recognising the speaker's voice from their other answers. Here each candidate
is scored with train.speaker_folds-style GroupKFold on 3 seeds. All out-of-fold predictions are saved, so they can be
blended with the speaker-fold DeBERTa predictions afterwards.
"""
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

Z = np.load(next(Path("/kaggle/input").rglob("eval_inputs.npz")))
y, H, A = Z["y"], Z["H"], Z["A"]; n = len(y); P = [Z[f"p_{i}"] for i in range(n)]
groups = pd.read_csv(next(Path("/kaggle/input").rglob("speaker_groups.csv"))).speaker_group.to_numpy()  # train.csv order
SEEDS = [42, 1, 2]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


def folds(seed):
    return list(GroupKFold(5, shuffle=True, random_state=seed).split(y, groups=groups))


def rows(clips):
    return np.vstack([np.concatenate([p, A[i], H[i]]) for i in clips for p in P[i]])


def pieces_oof(C, eps, gmul, agg, seed):
    oof = np.zeros(n)
    for a, b in folds(seed):
        X = rows(a); yy = np.concatenate([np.repeat(y[i], len(P[i])) for i in a])
        sc = StandardScaler().fit(X); Zx = sc.transform(X); g = gmul / (Zx.shape[1] * Zx.var())
        svr = SVR(kernel="precomputed", C=C, epsilon=eps).fit(rbf_kernel(Zx, gamma=g), yy)
        p = rbf_kernel(sc.transform(rows(b)), Zx[svr.support_], gamma=g) @ svr.dual_coef_[0] + svr.intercept_[0]
        parts = np.split(p, np.cumsum([len(P[i]) for i in b])[:-1])
        oof[b] = np.clip([np.median(s) if agg == "median" else np.mean(s) for s in parts], 0, 5)
    return oof


def clip_oof(kind, seed):
    X = np.hstack([A, H]); oof = np.zeros(n)
    for a, b in folds(seed):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40))) if kind == "ridge" else \
            make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05))
        oof[b] = np.clip(m.fit(X[a], y[a]).predict(X[b]), 0, 5)
    return oof


def job(cfg):
    if cfg[0] == "pieces":
        _, C, eps, gmul, agg = cfg
        return cfg, np.stack([pieces_oof(C, eps, gmul, agg, s) for s in SEEDS])
    return cfg, np.stack([clip_oof(cfg[1], s) for s in SEEDS])


cfgs = [("pieces", C, e, g, "median") for C, e, g in product([0.3, 1, 3], [0.05, 0.2], [0.5, 1, 2])]
cfgs += [("pieces", 1, 0.05, 1, "mean"), ("clip", "ridge"), ("clip", "svr")]
with Pool(4) as pool:
    res = pool.map(job, cfgs)
pos = y > 0
out = {}
print(f"{'model':<34} {'audio CV all':>12} {'1-5':>7}  per seed (all)")
for cfg, o in sorted(res, key=lambda r: np.mean([rmse(y, s) for s in r[1]])):
    name = "_".join(map(str, cfg)); out[name] = o
    print(f"{name:<34} {np.mean([rmse(y, s) for s in o]):12.4f} {np.mean([rmse(y[pos], s[pos]) for s in o]):7.4f}  {np.round([rmse(y, s) for s in o], 4)}")
np.savez("/kaggle/working/tune_speaker_oof.npz", **out)
