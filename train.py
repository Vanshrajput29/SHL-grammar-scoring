"""Audio model: SVR trained on overlapping ~10 s pieces of each clip (one every 5 s). Each piece's row is [piece WavLM
embedding, whole-clip WavLM embedding, hand-crafted + grammar-correction features of the clip]; it inherits the clip's
score, and a clip's prediction is the MEDIAN over its pieces. ~7,700 training rows instead of 769 (v10; v9 used
non-overlapping pieces and the mean, v6 one row per clip).
CV always splits by CLIP, so pieces of one clip never sit in both train and validation.
5-fold CV comparison against simpler baselines; saves models/audio_model.joblib; writes test_audio.csv.
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
from sklearn.model_selection import KFold, cross_val_predict
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

    def __init__(self, C, epsilon):
        self.C, self.epsilon = C, epsilon

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.gamma = 1 / (Z.shape[1] * Z.var())  # what SVR's gamma="scale" uses
        svr = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon).fit(rbf_kernel(Z, gamma=self.gamma), y)
        self.support_vectors, self.dual_coef, self.intercept = Z[svr.support_], svr.dual_coef_[0], svr.intercept_[0]
        return self

    def predict(self, X):
        return rbf_kernel(self.scaler.transform(X), self.support_vectors, gamma=self.gamma) @ self.dual_coef + self.intercept


def audio_model():
    """Final audio model, trained on pieces. Nested CV with clip-grouped inner and outer folds (on 10 s pieces) picked
    C=1, epsilon=0.05 in 3 of 5 outer folds (C=3 in the other 2): with many more rows, a more regularised SVR."""
    return ScaledRbfSVR(C=1, epsilon=0.05)


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


def pieces_cv(pieces, A, H, y, cv):
    """Out-of-fold clip predictions, folds split by clip."""
    oof = np.zeros(len(y))
    for tr, va in cv.split(y):
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

    cv = KFold(5, shuffle=True, random_state=42)
    evaluate("handcrafted (no gec)", H_tr[base].to_numpy(), y, cv)
    evaluate("handcrafted + gec", H_tr.to_numpy(), y, cv)
    evaluate("audio (WavLM)", A_tr, y, cv)
    X_tr = np.hstack([A_tr, H_tr.to_numpy()])
    evaluate("audio + handcrafted Ridge", X_tr, y, cv)
    evaluate("audio + handcrafted SVR (v6)", X_tr, y, cv, make=clip_svr)
    oof = pieces_cv(P_tr, A_tr, H_trn, y, cv)  # final (v10)
    print(f"{'pieces SVR (v10, final)':<24} OOF rmse={rmse(y, oof):.4f}  pearson={pearsonr(y, oof)[0]:.4f}")
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
