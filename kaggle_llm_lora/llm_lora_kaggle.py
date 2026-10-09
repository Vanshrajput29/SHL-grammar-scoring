"""Experiment (Kaggle GPU): fine-tune an LLM (Qwen2.5-3B) as the text model, instead of DeBERTa-v3-large.

The zero-shot LLM judge (kaggle_llm_judge/) was the strongest single text feature I found but never learned from the scored
transcripts. Here the LLM is trained on them: a regression head on the last token, with LoRA adapters (small trainable
matrices added to the frozen model) on a 4-bit copy of the weights (QLoRA), so a 3B model fits on one T4.
Same setup as kaggle_deberta_large/ otherwise: clips scored 1-5, KFold(5, shuffle, 42) over those 732 clips (so OOF
predictions line up with DeBERTa's), targets scaled to 0-1, MSE loss, fixed epochs, no epoch picking on the validation fold.
Run 1 (3 epochs, LR 1e-4) scored 0.718 OOF vs DeBERTa's 0.599: validation error was still falling at the last epoch and
jumped around mid-training, so run 2 trains longer with a smaller learning rate (5 epochs, 4e-5).
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "peft"], check=True)

import json, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

MODEL, MAX_LEN, EPOCHS, LR, MICRO, ACCUM, SEED = "Qwen/Qwen2.5-3B", 320, 5, 4e-5, 4, 2, 42
TAG = "qwen3b_lora"
DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
ZIP = next(Path("/kaggle/input").rglob("transcripts.zip"))
torch.manual_seed(SEED); np.random.seed(SEED)

with zipfile.ZipFile(ZIP) as z:
    text = {Path(n).stem: json.loads(z.read(n))["text"] for n in z.namelist()}
train = pd.read_csv(DATA / "train.csv")
train = train[train.label > 0].reset_index(drop=True)
test = pd.read_csv(DATA / "test.csv")
prompt = "Transcript of a spoken English answer. Rate its grammar from 1 to 5.\n\nTranscript: {}\n\nGrammar score:"
tr_text = [prompt.format(text["train_" + f[:-4]].strip()) for f in train.filename]
te_text = [prompt.format(text["test_" + f[:-4]].strip()) for f in test.filename]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"           # the head reads the last token; left padding keeps it at the end
enc = lambda t: tok(t, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
X_all, X_te = enc(tr_text), enc(te_text)
print("max tokens", int(X_all["attention_mask"].sum(1).max()), "of", MAX_LEN, flush=True)
y_clip = train.label.to_numpy()
y = torch.tensor((y_clip - 1) / 4, dtype=torch.float32)
unscale = lambda p: p * 4 + 1
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True)


def new_model():
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, quantization_config=bnb,
                                                           torch_dtype=torch.float16, device_map={"": 0})
    m.config.pad_token_id = tok.pad_token_id
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(m, cfg)
    for p in m.parameters():        # trainable LoRA + head weights in fp32 (the GradScaler needs fp32 gradients)
        if p.requires_grad:
            p.data = p.data.float()
    return m


def predict(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X["input_ids"]), 16):
            b = {k: v[i:i + 16].cuda() for k, v in X.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(model(**b).logits.float().squeeze(-1).cpu())
    return np.clip(unscale(torch.cat(out).numpy()), 1, 5)


oof, test_pred = np.zeros(len(train)), np.zeros(len(test))
for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(train)):
    t0 = time.time()
    model = new_model()
    if fold == 0:
        model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    n_steps = EPOCHS * (len(tr) // (MICRO * ACCUM) + 1)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * n_steps), n_steps)
    scaler = torch.amp.GradScaler("cuda")
    Xva = {k: v[va] for k, v in X_all.items()}
    for ep in range(EPOCHS):
        model.train()
        batches = np.array_split(np.random.permutation(tr), len(tr) // MICRO + 1)
        for j, idx in enumerate(batches):
            b = {k: v[idx].cuda() for k, v in X_all.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(**b).logits.squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred.float(), y[idx].cuda()) / ACCUM
            scaler.scale(loss).backward()
            if (j + 1) % ACCUM == 0 or j == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        p = predict(model, Xva)
        print(f"fold {fold} epoch {ep} val rmse {np.sqrt(np.mean((p - y_clip[va]) ** 2)):.4f}  ({time.time() - t0:.0f}s)",
              flush=True)  # monitoring only
    oof[va] = p
    test_pred += predict(model, X_te) / 5
    del model, opt; torch.cuda.empty_cache()

print("OOF rmse", np.sqrt(np.mean((oof - y_clip) ** 2)), "pearson", np.corrcoef(oof, y_clip)[0, 1])
pd.DataFrame({"filename": train.filename, "label": y_clip, TAG: oof}).to_csv(f"/kaggle/working/oof_{TAG}.csv", index=False)
pd.DataFrame({"filename": test.filename, TAG: test_pred}).to_csv(f"/kaggle/working/test_{TAG}.csv", index=False)
