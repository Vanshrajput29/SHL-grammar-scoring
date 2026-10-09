"""Experiment (Kaggle GPU): DeBERTa-v3-large trained on BOTH transcripts of every training clip (Whisper turbo and large-v3).

Same as kaggle_deberta_large/ except the training rows: each clip appears twice, once per transcript, both with the clip's
real score. That doubles the text data without the label noise that sank the 15 s windows. Validation and test are scored
on the turbo transcripts (what the final model uses), and also as the average over both transcripts.
EPOCHS is halved (6 -> 3) so the number of optimisation steps matches the original run."""
import json, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL, MAX_LEN, EPOCHS, LR, BS = "microsoft/deberta-v3-large", 256, 3, 1e-5, 8
SEEDS = [42]  # several seeds -> average out fine-tuning noise
TAG = "deberta_both_asr"  # output file / column name
FOLD_SEED = 42  # folds fixed so OOF lines up with the Ridge model's folds
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
ZIP, ZIP3 = next(Path("/kaggle/input").rglob("transcripts.zip")), next(Path("/kaggle/input").rglob("transcripts_v3.zip"))

def read(zp):
    with zipfile.ZipFile(zp) as z:
        return {Path(n).stem: json.loads(z.read(n))["text"] for n in z.namelist()}
text, text3 = read(ZIP), read(ZIP3)
train = pd.read_csv(DATA / "train.csv")
train = train[train.label > 0].reset_index(drop=True)
test = pd.read_csv(DATA / "test.csv")
train["text"] = [text["train_" + f[:-4]] for f in train.filename]
test["text"] = [text["test_" + f[:-4]] for f in test.filename]
train["text3"] = [text3["train_" + f[:-4]] for f in train.filename]
test["text3"] = [text3["test_" + f[:-4]] for f in test.filename]

tok = AutoTokenizer.from_pretrained(MODEL)
enc = lambda texts: tok(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
# Scale targets to [0,1] so the regression head starts in a sensible range.
scale = lambda y: (y - 1) / 4
unscale = lambda p: p * 4 + 1


def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X["input_ids"]), 32):
            b = {k: v[i:i + 32].cuda() for k, v in X.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(model(**b).logits.float().squeeze(-1).cpu())
    return unscale(torch.cat(out).numpy()).clip(1, 5)


# rows 0..n-1: turbo transcripts, rows n..2n-1: large-v3 transcripts of the same clips
n = len(train)
X_all, X_te, X_te3 = enc(list(train.text) + list(train.text3)), enc(test.text), enc(test.text3)
y = torch.tensor(scale(np.concatenate([train.label.to_numpy()] * 2)), dtype=torch.float32)
oof, test_pred = np.zeros(n), np.zeros(len(test))
oof_both, test_both = np.zeros(n), np.zeros(len(test))
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=FOLD_SEED).split(train)):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, torch_dtype=torch.float32).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        tr2 = np.concatenate([tr, tr + n])  # both transcripts of each training clip; validation clips never appear
        steps = EPOCHS * (len(tr2) // BS + 1)
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.cuda.amp.GradScaler()
        Xva, Xva3 = {k: v[va] for k, v in X_all.items()}, {k: v[va + n] for k, v in X_all.items()}
        for ep in range(EPOCHS):
            model.train()
            for idx in np.array_split(np.random.permutation(tr2), len(tr2) // BS + 1):
                b = {k: v[idx].cuda() for k, v in X_all.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(**b).logits.squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred.float(), y[idx].cuda())
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
            p = predict(model, Xva)
            r = float(np.sqrt(np.mean((p - train.label.to_numpy()[va]) ** 2)))
            print(f"seed {seed} fold {fold} epoch {ep} val rmse {r:.4f}", flush=True)  # monitoring only: no epoch picking on val
        p3, pt = predict(model, Xva3), predict(model, X_te)
        oof[va] += p / len(SEEDS); test_pred += pt / (5 * len(SEEDS))
        oof_both[va] += (p + p3) / 2 / len(SEEDS); test_both += (pt + predict(model, X_te3)) / 2 / (5 * len(SEEDS))
        del model; torch.cuda.empty_cache()

y_true = train.label.to_numpy()
for name, o in [("turbo", oof), ("avg of both transcripts", oof_both)]:
    print(f"OOF ({name}) rmse", np.sqrt(np.mean((o - y_true) ** 2)), "pearson", np.corrcoef(o, y_true)[0, 1])
pd.DataFrame({"filename": train.filename, "label": y_true, TAG: oof, TAG + "_tta": oof_both}).to_csv(f"/kaggle/working/oof_{TAG}.csv", index=False)
pd.DataFrame({"filename": test.filename, TAG: test_pred, TAG + "_tta": test_both}).to_csv(f"/kaggle/working/test_{TAG}.csv", index=False)
