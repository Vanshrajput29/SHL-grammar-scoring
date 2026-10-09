"""Experiment (Kaggle GPU): fine-tune DeBERTa-v3-large on ~15 s WINDOWS of each transcript instead of whole transcripts.

The audio model improved when it learned from ~10 s pieces (v9/v10) instead of one row per clip. This applies the same
idea to the text model: Whisper's segment timestamps cut each transcript into overlapping ~15 s windows (one every 7.5 s);
each window inherits its clip's score, and a clip's text score is the MEDIAN over its windows.
Same setup as kaggle_deberta_large/ otherwise: clips scored 1-5 only, KFold(5, shuffle, 42) over those 732 clips (so its
out-of-fold predictions line up with the existing DeBERTa runs), fixed epochs, no epoch picking on the validation fold.
All windows of a clip are always in the same fold.
"""
import json, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL, MAX_LEN, EPOCHS, LR, BS, SEED = "microsoft/deberta-v3-large", 128, 4, 1e-5, 16, 42
WIN_S, HOP_S = 15.0, 7.5
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
ZIP = next(Path("/kaggle/input").rglob("transcripts.zip"))
torch.manual_seed(SEED); np.random.seed(SEED)

with zipfile.ZipFile(ZIP) as z:
    recs = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()}


def windows(rec):
    """Overlapping ~15 s text windows from Whisper's segments (a segment belongs to a window if its midpoint does)."""
    # Whisper sometimes stamps a segment past the end of the clip (often a hallucinated "Thank you."); clamp it to the
    # end so every transcript word lands in some window, as it does in the whole-transcript model.
    end = rec["duration"] - 1e-3
    segs = [(min(0.5 * (s["start"] + s["end"]), end), s["text"].strip()) for s in rec["segments"] if s["text"].strip()]
    if not segs or rec["duration"] <= WIN_S:
        return [rec["text"].strip() or " "]
    out, t = [], 0.0
    while t < rec["duration"] - HOP_S or not out:
        txt = " ".join(x for mid, x in segs if t <= mid < t + WIN_S)
        if txt:
            out.append(txt)
        t += HOP_S
    return out or [rec["text"].strip() or " "]


train = pd.read_csv(DATA / "train.csv")
train = train[train.label > 0].reset_index(drop=True)
test = pd.read_csv(DATA / "test.csv")
tr_win = [windows(recs["train_" + f[:-4]]) for f in train.filename]
te_win = [windows(recs["test_" + f[:-4]]) for f in test.filename]
print("windows per training clip: median", int(np.median([len(w) for w in tr_win])),
      "| total", sum(map(len, tr_win)), "| example:", tr_win[0][0][:120], flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
y_clip = train.label.to_numpy()
scale, unscale = (lambda v: (v - 1) / 4), (lambda p: p * 4 + 1)


def predict_clips(model, clip_windows):
    """Median of the window predictions for each clip."""
    model.eval(); out = []
    with torch.no_grad():
        for wins in clip_windows:
            enc = tok(wins, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                p = model(**enc).logits.float().squeeze(-1).cpu().numpy()
            out.append(float(np.median(unscale(np.atleast_1d(p)))))
    return np.clip(out, 1, 5)


oof, test_pred = np.zeros(len(train)), np.zeros(len(test))
for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(train)):
    texts = [w for i in tr for w in tr_win[i]]
    ys = torch.tensor(np.concatenate([[scale(y_clip[i])] * len(tr_win[i]) for i in tr]), dtype=torch.float32)
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, torch_dtype=torch.float32).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n = len(texts); steps = EPOCHS * (n // BS + 1)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(EPOCHS):
        model.train()
        for idx in np.array_split(np.random.permutation(n), n // BS + 1):
            b = {k: v[idx].cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(**b).logits.squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred.float(), ys[idx].cuda())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        p = predict_clips(model, [tr_win[i] for i in va])
        print(f"fold {fold} epoch {ep} val rmse {np.sqrt(np.mean((p - y_clip[va]) ** 2)):.4f}", flush=True)  # monitoring only
    oof[va] = p
    test_pred += predict_clips(model, te_win) / 5
    del model; torch.cuda.empty_cache()

print("OOF rmse", np.sqrt(np.mean((oof - y_clip) ** 2)), "pearson", np.corrcoef(oof, y_clip)[0, 1])
pd.DataFrame({"filename": train.filename, "label": y_clip, "deberta_windows": oof}).to_csv("/kaggle/working/oof_deberta_windows.csv", index=False)
pd.DataFrame({"filename": test.filename, "deberta_windows": test_pred}).to_csv("/kaggle/working/test_deberta_windows.csv", index=False)
