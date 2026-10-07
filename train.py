"""Audio model: SVR on WavLM audio embedding + hand-crafted + grammar-correction features.
5-fold CV comparison against simpler Ridge baselines; saves models/audio_model.joblib; writes test_audio.csv.

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


def audio_model():
    """Final audio model. C and epsilon were chosen by nested CV (every outer fold picked C=3, epsilon=0.05)."""
    return make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05))


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
    E_tr = F.embed(train.text)
    A_tr, A_te = F.audio(train, "train"), F.audio(test, "test")
    base = [c for c in H_tr.columns if not c.startswith("gec_")]

    cv = KFold(5, shuffle=True, random_state=42)
    evaluate("handcrafted (no gec)", H_tr[base].to_numpy(), y, cv)
    evaluate("handcrafted + gec", H_tr.to_numpy(), y, cv)
    evaluate("text embedding", E_tr, y, cv)
    evaluate("handcrafted + text emb", np.hstack([H_tr.to_numpy(), E_tr]), y, cv)
    evaluate("audio (WavLM)", A_tr, y, cv)
    X_tr, X_te = np.hstack([A_tr, H_tr.to_numpy()]), np.hstack([A_te, H_te.to_numpy()])
    evaluate("audio + handcrafted Ridge", X_tr, y, cv)
    oof = evaluate("audio + handcrafted SVR", X_tr, y, cv, make=audio_model)  # final
    pd.DataFrame({"filename": train.filename, "label": y, "audio": oof}).to_csv("oof_audio.csv", index=False)

    m = audio_model().fit(X_tr, y)
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": m, "columns": list(H_tr.columns)}, "models/audio_model.joblib")  # used by predict.py
    print(f"train rmse={rmse(y, np.clip(m.predict(X_tr), 0, 5)):.4f}")
    pred = np.clip(m.predict(X_te), 0, 5)
    print("test clips flagged unintelligible (audio model < 1):", int((pred < 1).sum()))
    # The final submission (audio + DeBERTa ensemble) is written by the notebook.
    pd.DataFrame({"filename": test.filename, "audio": pred}).to_csv("test_audio.csv", index=False)
    print("wrote test_audio.csv", len(test))


if __name__ == "__main__":
    main()
