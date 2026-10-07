"""Fine-tune DeBERTa-v3 as a grammar-score regressor on Whisper transcripts (Kaggle GPU).
5-fold CV on label>0 clips; writes out-of-fold + fold-averaged test predictions."""
import json, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL, MAX_LEN, EPOCHS, LR, BS = "microsoft/deberta-v3-base", 256, 6, 2e-5, 8
SEEDS = [42]  # several seeds -> average out fine-tuning noise
TAG = "deberta"  # output file / column name
FOLD_SEED = 42  # folds fixed so OOF lines up with the Ridge model's folds
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
ZIP = next(Path("/kaggle/input").rglob("transcripts.zip"))

with zipfile.ZipFile(ZIP) as z:
    text = {Path(n).stem: json.loads(z.read(n))["text"] for n in z.namelist()}
train = pd.read_csv(DATA / "train.csv")
train = train[train.label > 0].reset_index(drop=True)
test = pd.read_csv(DATA / "test.csv")
train["text"] = [text["train_" + f[:-4]] for f in train.filename]
test["text"] = [text["test_" + f[:-4]] for f in test.filename]

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


X_all, X_te = enc(train.text), enc(test.text)
y = torch.tensor(scale(train.label.to_numpy()), dtype=torch.float32)
oof, test_pred = np.zeros(len(train)), np.zeros(len(test))
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=FOLD_SEED).split(train)):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, torch_dtype=torch.float32).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * (len(tr) // BS + 1)
        sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.cuda.amp.GradScaler()
        Xva = {k: v[va] for k, v in X_all.items()}
        for ep in range(EPOCHS):
            model.train()
            for idx in np.array_split(np.random.permutation(tr), len(tr) // BS + 1):
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
        oof[va] += p / len(SEEDS); test_pred += predict(model, X_te) / (5 * len(SEEDS))
        del model; torch.cuda.empty_cache()

y_true = train.label.to_numpy()
print("OOF rmse", np.sqrt(np.mean((oof - y_true) ** 2)), "pearson", np.corrcoef(oof, y_true)[0, 1])
pd.DataFrame({"filename": train.filename, "label": y_true, TAG: oof}).to_csv(f"/kaggle/working/oof_{TAG}.csv", index=False)
pd.DataFrame({"filename": test.filename, TAG: test_pred}).to_csv(f"/kaggle/working/test_{TAG}.csv", index=False)
