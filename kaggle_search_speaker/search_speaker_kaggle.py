"""Kaggle CPU: search with SPEAKER-grouped CV, the validation that finally agreed with the leaderboard (v15).

1. A finer grid around v15's audio settings (SVR C, epsilon, kernel width = gamma_scale x sklearn's "scale").
2. Earlier ideas that lost under clip-split CV, re-checked fairly: w2v-BERT next to WavLM (v12), averaging the two audio
   models (v13), speed-perturbed copies (v11a), pause-matched copies (v14), all with the smoother kernel.
Every variant is scored as the final ensemble (gate + 50/50 with the speaker-fold DeBERTa) on the scored 1-5 clips, on 3
GroupKFold seeds. Out-of-fold audio predictions are saved.
"""
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

IN = Path("/kaggle/input")
Z = np.load(next(IN.rglob("eval_inputs.npz")))
train = pd.read_csv(next(IN.rglob("train.csv")))
y, H, A = Z["y"], Z["H"], Z["A"]; n = len(y); pos = y > 0; P = [Z[f"p_{i}"] for i in range(n)]
groups = pd.read_csv(next(IN.rglob("speaker_groups.csv"))).speaker_group.to_numpy()
deb = train.filename.map(pd.read_csv(next(IN.rglob("oof_deberta_speaker.csv"))).set_index("filename").deberta_speaker).to_numpy()
W = np.load(next(IN.rglob("audio_pieces_w2vbert.npz")))
Pw = [W[f"train_{f[:-4]}"].astype(np.float32).mean(1) for f in train.filename]
Aw = np.stack([W[f"clip:train_{f[:-4]}"].astype(np.float32).mean(0) for f in train.filename])
SPEED = [([Z[f"p{sp}_{i}"] for i in range(n)], Z[f"A{sp}"], H) for sp in (0.9, 1.1)]
Q = np.load(next(IN.rglob("audio_pieces_pauses.npz")))
Hq = H.copy(); Hq[:, 1] = H[:, 0] / Q["duration"]; Hq[:, 10] = Q["rms"]          # words_per_sec, rms of the shortened audio
PAUSE = ([Q[f"train_{f[:-4]}"] for f in train.filename], np.stack([Q[f"clip:train_{f[:-4]}"] for f in train.filename]), Hq)
ORIG, W2V = (P, A, H), ([np.hstack([a, b]) for a, b in zip(P, Pw)], np.hstack([A, Aw]), H)
SEEDS = [42, 1, 2]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
folds = {s: list(GroupKFold(5, shuffle=True, random_state=s).split(y, groups=groups)) for s in SEEDS}


def rows(Pp, Aa, Hh, clips):
    return np.vstack([np.concatenate([p, Aa[i], Hh[i]]) for i in clips for p in Pp[i]])


def oof(train_sets, val_set, C, eps, gs, seed):
    """Pieces SVR (same as train.ScaledRbfSVR with gamma_scale) trained on several versions of the training clips."""
    out = np.zeros(n)
    for a, b in folds[seed]:
        X = np.vstack([rows(*s, a) for s in train_sets])
        yy = np.concatenate([np.repeat(y[i], len(s[0][i])) for s in train_sets for i in a])
        sc = StandardScaler().fit(X); Zx = sc.transform(X); g = gs / (Zx.shape[1] * Zx.var())
        svr = SVR(kernel="precomputed", C=C, epsilon=eps).fit(rbf_kernel(Zx, gamma=g), yy)
        p = rbf_kernel(sc.transform(rows(*val_set, b)), Zx[svr.support_], gamma=g) @ svr.dual_coef_[0] + svr.intercept_[0]
        out[b] = np.clip([np.median(s) for s in np.split(p, np.cumsum([len(val_set[0][i]) for i in b])[:-1])], 0, 5)
    return out


def ens(a):
    e = np.where(a < 1, a, 0.5 * a + 0.5 * deb)
    return rmse(y[pos], e[pos])


SETS = {"wavlm": ([ORIG], ORIG), "w2v": ([W2V], W2V), "speed": ([ORIG] + SPEED, ORIG), "pause": ([ORIG, PAUSE], ORIG)}


def job(cfg):
    name, C, eps, gs = cfg
    tr, va = SETS[name]
    return cfg, np.stack([oof(tr, va, C, eps, gs, s) for s in SEEDS])


grid = [("wavlm", C, e, g) for C, e, g in product([0.5, 1, 2], [0.05, 0.1, 0.2], [0.2, 0.3, 0.4, 0.5])] + [("wavlm", 1, 0.05, 1.0)]
ideas = [("w2v", 1, 0.05, 0.5), ("w2v", 1, 0.05, 0.3), ("pause", 1, 0.05, 0.5)]
with Pool(4) as pool:
    res = dict(pool.map(job, grid + ideas))
with Pool(2) as pool:                                     # 3x the rows: fewer workers so the kernel matrices fit in RAM
    res.update(dict(pool.map(job, [("speed", 1, 0.05, 0.5)])))
res[("v13-style avg", "wavlm+w2v", "", 0.5)] = (res[("wavlm", 1, 0.05, 0.5)] + res[("w2v", 1, 0.05, 0.5)]) / 2

v10 = [ens(s) for s in res[("wavlm", 1, 0.05, 1.0)]]; v15 = [ens(s) for s in res[("wavlm", 1, 0.05, 0.5)]]
print(f"reference  v10 {np.mean(v10):.4f} {np.round(v10, 4)} | v15 {np.mean(v15):.4f} {np.round(v15, 4)}\n")
print(f"{'variant (set, C, eps, gamma_scale)':<38} {'ensemble 1-5':>12}  per seed               better than v15 on   score-0 caught")
for cfg, o in sorted(res.items(), key=lambda kv: np.mean([ens(s) for s in kv[1]])):
    e = [ens(s) for s in o]
    print(f"{str(cfg):<38} {np.mean(e):12.4f}  {np.round(e, 4)}  {sum(a < b for a, b in zip(e, v15))}/3   {np.mean([((s < 1) & ~pos).sum() for s in o]):.0f}/37")
np.savez("/kaggle/working/search_speaker_oof.npz", **{"|".join(map(str, k)): v for k, v in res.items()})
