"""Kaggle CPU: is weighting the hand-made features up (speaker_invariant run: x4 best, 0.5318 vs 0.5339) a real gain?

1. Nested: the weight is chosen from {1, 2, 4, 8} inside each training fold (inner speaker folds), then scored on the outer
   fold, so the reported number isn't flattered by picking the best of a grid on the same folds. Compared with v15 (x1)
   on identical outer folds.
2. The v16 idea with the weight: average of the WavLM pieces model and the WavLM + w2v-BERT pieces model, both x4.
All: final ensemble (gate + 50/50 with the speaker-fold DeBERTa), 1-5 clips, 3 speaker-grouped seeds.
Speed: the weight only scales the 13 hand-made columns, so squared distances split into (WavLM part) + w^2 (hand part).
Both parts are computed once per (train, validation) split and every weight reuses them (same predictions, ~4x fewer
distance computations).
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

IN = Path("/kaggle/input")
Z = np.load(next(IN.rglob("eval_inputs.npz"))); train = pd.read_csv(next(IN.rglob("train.csv")))
y, H, A = Z["y"], Z["H"], Z["A"]; n = len(y); pos = y > 0; P = [Z[f"p_{i}"] for i in range(n)]
groups = pd.read_csv(next(IN.rglob("speaker_groups.csv"))).speaker_group.to_numpy()
deb = train.filename.map(pd.read_csv(next(IN.rglob("oof_deberta_speaker.csv"))).set_index("filename").deberta_speaker).to_numpy()
W = np.load(next(IN.rglob("audio_pieces_w2vbert.npz")))
Pw = [np.hstack([p, W[f"train_{f[:-4]}"].astype(np.float32).mean(1)]) for p, f in zip(P, train.filename)]
Aw = np.hstack([A, np.stack([W[f"clip:train_{f[:-4]}"].astype(np.float32).mean(0) for f in train.filename])])
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
WEIGHTS = [1, 2, 4, 8]


def split_distances(Pp, Aa, tr, va):
    """Standardise on the training rows, then squared distances of the embedding columns and of the 13 hand-made columns,
    train-train and validation-train. Plus the per-column variances, which set gamma for any weight."""
    rows = lambda c: np.vstack([np.concatenate([p, Aa[i], H[i]]) for i in c for p in Pp[i]])
    sc = StandardScaler().fit(rows(tr)); Zt, Zv = sc.transform(rows(tr)), sc.transform(rows(va))
    d2 = lambda a, b: euclidean_distances(a, b, squared=True)
    return dict(Dt=(d2(Zt[:, :-13], Zt[:, :-13]), d2(Zt[:, -13:], Zt[:, -13:])), Dv=(d2(Zv[:, :-13], Zt[:, :-13]), d2(Zv[:, -13:], Zt[:, -13:])),
                var=(Zt[:, :-13].var(0).sum(), Zt[:, -13:].var(0).sum()), d=Zt.shape[1],
                yy=np.concatenate([np.repeat(y[i], len(Pp[i])) for i in tr]), counts=[len(Pp[i]) for i in va])


def predict_w(D, hw):
    """Same model as before (gamma = 0.5 / (n_columns * variance of the weighted matrix)), for weight hw."""
    g = 0.5 / (D["d"] * (D["var"][0] + hw ** 2 * D["var"][1]) / D["d"])
    svr = SVR(kernel="precomputed", C=1, epsilon=0.05).fit(np.exp(-g * (D["Dt"][0] + hw ** 2 * D["Dt"][1])), D["yy"])
    Kv = np.exp(-g * (D["Dv"][0][:, svr.support_] + hw ** 2 * D["Dv"][1][:, svr.support_]))
    p = Kv @ svr.dual_coef_[0] + svr.intercept_[0]
    return np.clip([np.median(s) for s in np.split(p, np.cumsum(D["counts"])[:-1])], 0, 5)



def ens(a, idx):
    e = np.where(a < 1, a, 0.5 * a + 0.5 * deb[idx]); m = pos[idx]; return e[m], y[idx][m]


def job(seed):
    outer = list(GroupKFold(5, shuffle=True, random_state=seed).split(y, groups=groups))
    pred = {k: np.zeros(n) for k in ("nested", "v15 x1", "x4", "v16-style x1", "v16-style x4")}; chosen = []
    for a, b in outer:
        inner = list(GroupKFold(4, shuffle=True, random_state=seed + 7).split(a, groups=groups[a]))
        Ds = [(split_distances(P, A, a[i], a[j]), a[j]) for i, j in inner]       # once per inner split, reused for every weight
        sc = {}
        for hw in WEIGHTS:
            errs = [ens(predict_w(D, hw), idx) for D, idx in Ds]
            sc[hw] = rmse(np.concatenate([e for e, _ in errs]), np.concatenate([t for _, t in errs]))
        best = min(sc, key=sc.get); chosen.append(best)
        D, Dw = split_distances(P, A, a, b), split_distances(Pw, Aw, a, b)
        pred["nested"][b] = predict_w(D, best)
        pred["v15 x1"][b] = predict_w(D, 1)
        pred["x4"][b] = predict_w(D, 4)
        pred["v16-style x1"][b] = (pred["v15 x1"][b] + predict_w(Dw, 1)) / 2
        pred["v16-style x4"][b] = (pred["x4"][b] + predict_w(Dw, 4)) / 2
    return seed, chosen, {k: rmse(*ens(v, np.arange(n))) for k, v in pred.items()}


with Pool(3) as pool:
    res = pool.map(job, [42, 1, 2])
for seed, chosen, r in res:
    print(f"seed {seed}: weight chosen per outer fold {chosen} | " + " | ".join(f"{k} {v:.4f}" for k, v in r.items()))
print("\nmean over seeds:")
for k in res[0][2]:
    v = [r[k] for _, _, r in res]
    print(f"   {k:<14} {np.mean(v):.4f}  per seed {np.round(v, 4)}  better than v15 on {sum(a < b for a, b in zip(v, [r['v15 x1'] for _, _, r in res]))}/3")
