"""Train the shipped DeBERTa-v3-large on ALL label>0 training clips (same settings as the CV runs, no validation split)
and save it for predict.py. Output: /kaggle/working/deberta_final/ (weights + tokenizer)."""
import json, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL, MAX_LEN, EPOCHS, LR, BS, SEED = "microsoft/deberta-v3-large", 256, 6, 1e-5, 8, 42
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
ZIP = next(Path("/kaggle/input").rglob("transcripts.zip"))
OUT = Path("/kaggle/working/deberta_final")
torch.manual_seed(SEED); np.random.seed(SEED)

with zipfile.ZipFile(ZIP) as z:
    text = {Path(n).stem: json.loads(z.read(n))["text"] for n in z.namelist()}
train = pd.read_csv(DATA / "train.csv")
train = train[train.label > 0].reset_index(drop=True)  # score-0 clips are handled by the audio model's gate
tok = AutoTokenizer.from_pretrained(MODEL)
enc = tok([text["train_" + f[:-4]] for f in train.filename], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
y = torch.tensor((train.label.to_numpy() - 1) / 4, dtype=torch.float32)  # scaled to [0, 1]; predict.py unscales

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, torch_dtype=torch.float32).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
n = len(train); steps = EPOCHS * (n // BS + 1)
sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
for ep in range(EPOCHS):
    model.train(); total = 0.0
    for idx in np.array_split(np.random.permutation(n), n // BS + 1):
        b = {k: v[idx].cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(**b).logits.squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred.float(), y[idx].cuda())
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); total += loss.item() * len(idx)
    print(f"epoch {ep} train mse {total / n:.4f}", flush=True)
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print("saved", OUT)
