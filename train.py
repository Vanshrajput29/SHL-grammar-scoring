"""Audio model: SVR trained on overlapping ~10 s pieces of each clip (one every 5 s). Each piece's row is [piece WavLM
embedding, whole-clip WavLM embedding, hand-crafted + grammar-correction features of the clip]; it inherits the clip's
score, and a clip's prediction is the MEDIAN over its pieces. ~7,700 training rows instead of 769 (v10/v15; v9 used
non-overlapping pieces and the mean, v6 one row per clip). v15 = v10 with a smoother kernel (see audio_model).
CV always keeps all pieces of a clip, and all clips of a speaker, on the same side.
5-fold CV comparison against simpler baselines; saves models/audio_model.joblib; writes test_audio.csv.
CV is grouped by SPEAKER (speaker_groups): many training speakers answered several questions, usually with the same score,
and the test speakers are all new, so splitting by clip let models score a validation clip by recognising the voice.
(The frozen sentence-embedding baselines are compared in the notebook only: computing them here took ~95% of the runtime.)

Predictions cover the full 0-5 range. Label-0 clips (unintelligible / off-task speech) stay in training: the audio
model scores (almost) all of them below 1 and every real speaker above 1, so the final ensemble uses
"audio model < 1" as an intelligibility gate (see shl_grammar_scoring.ipynb).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import features as F


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def model():
    """Linear baseline (alpha picked by internal CV) - used for the comparisons and feature-weight plot."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))


def clip_svr():
    """v6's audio model: one row per clip (kept for comparison). Nested CV picked C=3, epsilon=0.05 in every fold."""
    return make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05))


class ScaledRbfSVR:
    """The same model as make_pipeline(StandardScaler(), SVR(C, epsilon)) with the default RBF kernel and gamma="scale"
    (predictions agree to ~1e-13), but ~6x faster to fit and ~25x faster to predict on thousands of rows: the kernel is
    computed as one matrix product instead of libsvm's pair-by-pair loop. Only the support vectors are kept."""

    def __init__(self, C, epsilon, gamma_scale=1.0):
        self.C, self.epsilon, self.gamma_scale = C, epsilon, gamma_scale

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.gamma = self.gamma_scale / (Z.shape[1] * Z.var())  # gamma_scale=1 is SVR's gamma="scale"
        svr = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon).fit(rbf_kernel(Z, gamma=self.gamma), y)
        self.support_vectors, self.dual_coef, self.intercept = Z[svr.support_], svr.dual_coef_[0], svr.intercept_[0]
        return self

    def predict(self, X):
        return rbf_kernel(self.scaler.transform(X), self.support_vectors, gamma=self.gamma) @ self.dual_coef + self.intercept


def audio_model():
    """Final audio model (v15), trained on pieces. C=1, epsilon=0.05 as in v10, with a smoother kernel (half of
    sklearn's default gamma): re-tuned with speaker-grouped CV, which v10's settings were not, a wider kernel generalises
    better to new speakers (ensemble CV on the 1-5 clips 0.5339 vs 0.5362, better on 3/3 splits; public 0.3428 vs 0.3466)."""
    return ScaledRbfSVR(C=1, epsilon=0.05, gamma_scale=0.5)


def piece_rows(pieces, A, H, clips):
    """Rows for the given clip indices: [piece embedding, clip embedding, hand-made features], one per piece."""
    return np.vstack([np.concatenate([p, A[i], H[i]]) for i in clips for p in pieces[i]])


def fit_pieces(pieces, A, H, y, clips):
    ytr = np.concatenate([[y[i]] * len(pieces[i]) for i in clips])
    return audio_model().fit(piece_rows(pieces, A, H, clips), ytr)


def predict_pieces(m, pieces, A, H, clips):
    """Clip score = median of its pieces' predictions (less sensitive to one odd piece than the mean), clipped to 0-5."""
    p = m.predict(piece_rows(pieces, A, H, clips))
    return np.clip([np.median(s) for s in np.split(p, np.cumsum([len(pieces[i]) for i in clips])[:-1])], 0, 5)


def speaker_groups(df, A, y):
    """Likely-same-speaker groups among the scored clips. Two clips are linked if their WavLM clip embeddings are
    near-identical (cosine >= 0.98, ~0.1% of random pairs), or their file numbers are within 3 and cosine >= 0.93 (a
    speaker's answers are usually numbered side by side; only ~1.6% of random pairs are that similar). Most groups are
    one speaker's 3 answers. Score-0 clips stay on their own: they sound alike because they're mostly noise."""
    num = df.filename.str.extract(r"(\d+)")[0].astype(int).to_numpy()
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    S, pos = An @ An.T, np.asarray(y) > 0
    link = (S >= 0.98) | ((np.abs(num[:, None] - num[None, :]) <= 3) & (S >= 0.93))
    link &= pos[:, None] & pos[None, :]
    np.fill_diagonal(link, False)
    return connected_components(csr_matrix(link), directed=False)[1]


def speaker_folds(groups, seed=42, k=5):
    """k folds that never split a speaker group: list of (train_idx, val_idx), usable as cv= anywhere."""
    return list(GroupKFold(k, shuffle=True, random_state=seed).split(groups, groups=groups))


def pieces_cv(pieces, A, H, y, folds):
    """Out-of-fold clip predictions for the given folds (a list of (train_idx, val_idx))."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = predict_pieces(fit_pieces(pieces, A, H, y, tr), pieces, A, H, va)
    return oof


UNINTELLIGIBLE_BELOW = 1.0  # audio-model scores under the rubric's minimum (1) mean unintelligible / off-task speech


def combine_scores(audio, text, w=0.5):
    """The final scoring rule, shared by the notebook and predict.py.
    Audio score below UNINTELLIGIBLE_BELOW -> keep the audio score (~0); otherwise w*audio + (1-w)*text.
    `text` may be NaN where the text model was not run (it is never needed for unintelligible clips)."""
    audio, text = np.asarray(audio, dtype=float), np.asarray(text, dtype=float)
    return np.where((audio < UNINTELLIGIBLE_BELOW) | np.isnan(text), audio, w * audio + (1 - w) * np.nan_to_num(text))


def evaluate(name, X, y, cv, make=model):
    oof = np.clip(cross_val_predict(make(), X, y, cv=cv), 0, 5)
    print(f"{name:<24} OOF rmse={rmse(y, oof):.4f}  pearson={pearsonr(y, oof)[0]:.4f}")
    return oof


def main():
    train, test = F.load("train"), F.load("test")
    y = train.label.to_numpy()
    H_tr, H_te = F.handcrafted(train, "train"), F.handcrafted(test, "test")
    A_tr, A_te = F.audio(train, "train"), F.audio(test, "test")
    P_tr, P_te = F.audio_pieces(train, "train"), F.audio_pieces(test, "test")
    H_trn, H_ten = H_tr.to_numpy(), H_te.to_numpy()
    base = [c for c in H_tr.columns if not c.startswith("gec_")]

    cv = speaker_folds(speaker_groups(train, A_tr, y))
    evaluate("handcrafted (no gec)", H_tr[base].to_numpy(), y, cv)
    evaluate("handcrafted + gec", H_tr.to_numpy(), y, cv)
    evaluate("audio (WavLM)", A_tr, y, cv)
    X_tr = np.hstack([A_tr, H_tr.to_numpy()])
    evaluate("audio + handcrafted Ridge", X_tr, y, cv)
    evaluate("audio + handcrafted SVR (v6)", X_tr, y, cv, make=clip_svr)
    oof = pieces_cv(P_tr, A_tr, H_trn, y, cv)  # final (v15)
    print(f"{'pieces SVR (v15, final)':<24} OOF rmse={rmse(y, oof):.4f}  pearson={pearsonr(y, oof)[0]:.4f}")
    leaky = pieces_cv(P_tr, A_tr, H_trn, y, list(KFold(5, shuffle=True, random_state=42).split(y)))
    print(f"  (same model with folds split by clip, as before: {rmse(y, leaky):.4f} - optimistic, speakers leak across folds)")
    pd.DataFrame({"filename": train.filename, "label": y, "audio": oof}).to_csv("oof_audio.csv", index=False)

    every = np.arange(len(y))
    m = fit_pieces(P_tr, A_tr, H_trn, y, every)
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": m, "columns": list(H_tr.columns), "kind": "pieces"}, "models/audio_model.joblib")  # predict.py
    print(f"train rmse={rmse(y, predict_pieces(m, P_tr, A_tr, H_trn, every)):.4f}")
    pred = predict_pieces(m, P_te, A_te, H_ten, np.arange(len(test)))
    print("test clips flagged unintelligible (audio model < 1):", int((pred < 1).sum()))
    # The final submission (audio + DeBERTa ensemble) is written by the notebook.
    pd.DataFrame({"filename": test.filename, "audio": pred}).to_csv("test_audio.csv", index=False)
    print("wrote test_audio.csv", len(test))


if __name__ == "__main__":
    import train  # run via the module so the saved model pickles as train.ScaledRbfSVR, which predict.py can load
    train.main()
