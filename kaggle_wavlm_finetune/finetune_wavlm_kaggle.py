"""Fine-tune WavLM-base-plus end to end as a grammar-score regressor on raw audio (Kaggle GPU).

5-fold CV over all 769 training clips with the SAME folds as the SVR audio model (KFold(5, shuffle, 42)), so the
out-of-fold predictions can be compared and combined fairly. Writes out-of-fold + fold-averaged test predictions.
- learnable softmax-weighted mix of all hidden layers, mean-pooled over time, linear head
- CNN feature encoder frozen (standard; limits overfitting on 769 clips); LayerDrop off so every layer is mixed
- training on random 15 s crops (augmentation); prediction = mean over all 15 s chunks of a clip
- fixed number of epochs: no epoch picking on the validation fold
"""
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from transformers import AutoFeatureExtractor, WavLMModel, get_cosine_schedule_with_warmup

MODEL, SR = "microsoft/wavlm-base-plus", 16000
CROP, EPOCHS, BS, LR, HEAD_LR, SEED, FOLD_SEED = 15 * SR, 10, 8, 3e-5, 1e-3, 42, 42
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return x.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)  # int16 keeps all clips in RAM


train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
tr_audio = [load_wav(DATA / "train" / f) for f in train.filename]
te_audio = [load_wav(DATA / "test" / f) for f in test.filename]
y = train.label.to_numpy(np.float32)
# Training batches are random crops of exactly CROP samples, so they never need padding - this relies on every
# training clip being at least CROP long (the shortest is 20 s). Fail loudly if the data ever breaks that.
if min(len(x) for x in tr_audio) < CROP:
    raise ValueError("a training clip is shorter than the 15 s crop; batches would need padding + attention masks")
fe = AutoFeatureExtractor.from_pretrained(MODEL)
print(len(tr_audio), "train clips,", len(te_audio), "test clips", flush=True)


class WavLMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # layerdrop=0: WavLM randomly skips layers in training by default (5%), which changes how many hidden
        # states come back per step; the learned layer mix needs all of them every time.
        self.wavlm = WavLMModel.from_pretrained(MODEL, torch_dtype=torch.float32, layerdrop=0.0)
        self.wavlm.feature_extractor._freeze_parameters()
        self.layer_w = nn.Parameter(torch.zeros(self.wavlm.config.num_hidden_layers + 1))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.wavlm.config.hidden_size, 1))

    def forward(self, x):
        hs = torch.stack(self.wavlm(x, output_hidden_states=True).hidden_states)       # [layers, B, T, H]
        mixed = (torch.softmax(self.layer_w, 0)[:, None, None, None] * hs).sum(0)       # [B, T, H]
        return self.head(mixed.mean(1)).squeeze(-1)                                      # score / 5


def to_input(chunks):
    return fe([c.astype(np.float32) / 32768 for c in chunks], sampling_rate=SR, return_tensors="pt",
              padding=True).input_values.cuda()


def random_crop(x):
    if len(x) <= CROP:
        return x
    s = rng.integers(0, len(x) - CROP)
    return x[s:s + CROP]


def predict(model, clips):
    """Mean prediction over consecutive 15 s chunks (a final chunk shorter than 5 s is dropped unless it is the only one).
    Each chunk is scored on its own: batching a shorter last chunk with full ones would zero-pad it, and the model
    would then attend to and average in the padding (the first version did this; see notebook section 10)."""
    model.eval(); out = []
    with torch.no_grad():
        for x in clips:
            chunks = [x[s:s + CROP] for s in range(0, len(x), CROP)]
            chunks = [c for c in chunks if len(c) >= 5 * SR] or chunks[:1]
            with torch.autocast("cuda", dtype=torch.float16):
                scores = [float(model(to_input([c])).float()) for c in chunks]
            out.append(np.mean(scores) * 5)
    return np.clip(out, 0, 5)


oof, test_pred = np.zeros(len(train)), np.zeros(len(test))
for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=FOLD_SEED).split(train)):
    model = WavLMRegressor().cuda()
    head_params = [model.layer_w, *model.head.parameters()]
    body_params = [p for n, p in model.named_parameters() if n.startswith("wavlm.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body_params, "lr": LR}, {"params": head_params, "lr": HEAD_LR}], weight_decay=0.01)
    steps = EPOCHS * (len(tr) // BS + 1)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(EPOCHS):
        model.train()
        for idx in np.array_split(rng.permutation(tr), len(tr) // BS + 1):
            x = to_input([random_crop(tr_audio[i]) for i in idx])
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x)
            loss = nn.functional.mse_loss(pred.float(), torch.tensor(y[idx] / 5, device="cuda"))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        if ep in (0, 4, EPOCHS - 1):  # monitoring only - no epoch picking
            p = predict(model, [tr_audio[i] for i in va])
            print(f"fold {fold} epoch {ep} val rmse {np.sqrt(np.mean((p - y[va]) ** 2)):.4f}", flush=True)
    oof[va] = predict(model, [tr_audio[i] for i in va])
    test_pred += predict(model, te_audio) / 5
    print(f"fold {fold} done: val rmse {np.sqrt(np.mean((oof[va] - y[va]) ** 2)):.4f} | layer weights "
          f"{np.round(torch.softmax(model.layer_w, 0).detach().cpu().numpy(), 2).tolist()}", flush=True)
    del model; torch.cuda.empty_cache()

print("OOF rmse", np.sqrt(np.mean((oof - y) ** 2)), "pearson", np.corrcoef(oof, y)[0, 1])
pd.DataFrame({"filename": train.filename, "label": y, "wavlm_ft": oof}).to_csv("/kaggle/working/oof_wavlm_ft.csv", index=False)
pd.DataFrame({"filename": test.filename, "wavlm_ft": test_pred}).to_csv("/kaggle/working/test_wavlm_ft.csv", index=False)
