"""Kaggle CPU: honesty checks for the final model (v15), all with speaker-grouped folds.

1. Label-shuffle test: scramble the scores (whole speakers move together) and run the full audio pipeline. If nothing
   leaks, it can't beat predicting the mean; if it does, information is getting from validation into training.
2. Nested selection: v15's kernel width was the best of 21 settings on the same folds it's scored on, so its CV is
   slightly flattering. Here the width is chosen inside each training fold only (inner speaker folds), giving an
   unbiased estimate of the v15 procedure vs v10's fixed settings on identical outer folds.
3. Per-fold gains of v15 over v10, to see whether one fold drives the improvement.
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
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
GAMMAS = [0.3, 0.5, 0.7, 1.0]


def rows(clips):
    return np.vstack([np.concatenate([p, A[i], H[i]]) for i in clips for p in P[i]])


def fit_predict(tr, va, yy, gs):
    X = rows(tr); yr = np.concatenate([np.repeat(yy[i], len(P[i])) for i in tr])
    sc = StandardScaler().fit(X); Zx = sc.transform(X); g = gs / (Zx.shape[1] * Zx.var())
    svr = SVR(kernel="precomputed", C=1, epsilon=0.05).fit(rbf_kernel(Zx, gamma=g), yr)
    p = rbf_kernel(sc.transform(rows(va)), Zx[svr.support_], gamma=g) @ svr.dual_coef_[0] + svr.intercept_[0]
    return np.clip([np.median(s) for s in np.split(p, np.cumsum([len(P[i]) for i in va])[:-1])], 0, 5)


def ens(a, idx):
    e = np.where(a < 1, a, 0.5 * a + 0.5 * deb[idx])
    m = pos[idx]; return rmse(y[idx][m], e[m])


def shuffle_job(seed):
    rng = np.random.default_rng(seed); ug = np.unique(groups)
    # move whole speakers: each group takes the scores of another group of the same size where possible
    perm = dict(zip(ug, rng.permutation(ug))); ys = y.copy()
    for g0, g1 in perm.items():
        a, b = np.where(groups == g0)[0], np.where(groups == g1)[0]
        ys[a] = y[b[np.arange(len(a)) % len(b)]]
    oof = np.zeros(n)
    for a, b in GroupKFold(5, shuffle=True, random_state=seed).split(y, groups=groups):
        oof[b] = fit_predict(a, b, ys, 0.5)
    return seed, rmse(ys, oof), rmse(ys, np.full(n, ys.mean()))


def nested_job(seed):
    outer = list(GroupKFold(5, shuffle=True, random_state=seed).split(y, groups=groups))
    res = []
    for k, (a, b) in enumerate(outer):
        inner = list(GroupKFold(4, shuffle=True, random_state=seed + 7).split(a, groups=groups[a]))
        score = {gs: np.mean([ens(fit_predict(a[i], a[j], y, gs), a[j]) for i, j in inner]) for gs in GAMMAS}
        best = min(score, key=score.get)
        res.append((k, best, ens(fit_predict(a, b, y, best), b), ens(fit_predict(a, b, y, 0.5), b), ens(fit_predict(a, b, y, 1.0), b), len(b)))
    return seed, res


with Pool(4) as pool:
    shuf = pool.map(shuffle_job, [11, 12, 13])
    nest = pool.map(nested_job, [42, 1, 2])
print("1. label-shuffle test (scores moved between whole speakers; audio model, speaker folds):")
for s, r, base in shuf:
    print(f"   seed {s}: model RMSE {r:.4f} vs predicting the mean {base:.4f}  -> {'OK, no leak' if r >= base - 0.02 else 'LEAK?'}")
print("\n2./3. nested choice of kernel width (ensemble RMSE on the 1-5 clips of each outer fold):")
tot = {"nested": [], "v15 (0.5)": [], "v10 (1.0)": []}
for s, res in nest:
    w = np.array([r[5] for r in res])
    for name, col in [("nested", 2), ("v15 (0.5)", 3), ("v10 (1.0)", 4)]:
        tot[name].append(np.sqrt(np.sum(w * np.array([r[col] for r in res]) ** 2) / w.sum()))
    print(f"   seed {s}: widths chosen per fold {[r[1] for r in res]} | v15 better than v10 in {sum(r[3] < r[4] for r in res)}/5 folds, "
          f"fold gains {np.round([r[4] - r[3] for r in res], 4)}")
for k, v in tot.items():
    print(f"   {k:<10} {np.mean(v):.4f}  per seed {np.round(v, 4)}")
