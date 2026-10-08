"""Experiment (Kaggle GPU): an LLM grammar judge that reads each transcript with SHL's rubric.

Our features count errors; the rubric also rewards "complex grammar used accurately". An instruction-tuned LLM
(Qwen2.5-7B-Instruct, Apache-2.0, 4-bit) is asked for a 1-5 grammar rating per the rubric. Zero-shot on purpose:
putting scored training clips in the prompt would leak their labels into cross-validation. Instead of generating text,
we read the probabilities of the answers "1".."5" and use the expected rating (and the probabilities) as features -
one deterministic forward pass per transcript.
Evaluated exactly like v6 (SVR(C=3, eps=0.05) on [WavLM + hand-made], KFold(5, shuffle, 42), combined with DeBERTa-large
OOF); the v6 baseline is recomputed first. Features are computed for train and test and saved.
"""
import difflib, json, re, subprocess, sys, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LLM = "Qwen/Qwen2.5-7B-Instruct"
INP = Path("/kaggle/input")
find = lambda name: next(INP.rglob(name))
DATA = find("train.csv").parent
train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
y = train.label.to_numpy(); ok = y > 0
keys = ["train_" + Path(f).stem for f in train.filename]
test_keys = ["test_" + Path(f).stem for f in test.filename]
with zipfile.ZipFile(find("transcripts.zip")) as z:
    recs = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()}

# ---- 1. LLM ratings ---------------------------------------------------------------------------------------------------
RUBRIC = """1 - The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns.
2 - The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete.
3 - The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure.
4 - The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them.
5 - Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary."""


def prompt(text):
    return [{"role": "system", "content": "You are an expert examiner of spoken English grammar."},
            {"role": "user", "content": (
                "Below is an automatic transcript of someone speaking English for about a minute. It comes from speech "
                "recognition, so ignore punctuation, capitalisation and spelling; fillers (uh, um) and repetitions are "
                "part of how the person spoke. Rate the speaker's GRAMMAR on this rubric:\n\n" + RUBRIC +
                "\n\nTranscript:\n\"\"\"" + text.strip() + "\"\"\"\n\nAnswer with a single digit from 1 to 5.")}]


tok = AutoTokenizer.from_pretrained(LLM)
tok.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained(LLM, device_map="cuda", quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")).eval()
digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]
assert len(set(digit_ids)) == 5, digit_ids

all_keys = keys + test_keys
chats = [tok.apply_chat_template(prompt(recs[k]["text"] or "(no speech)"), tokenize=False, add_generation_prompt=True)
         for k in all_keys]
probs = []
for i in range(0, len(chats), 8):
    enc = tok(chats[i:i + 8], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        logits = llm(**enc).logits[:, -1, :].float()           # next-token logits after the assistant prompt
    p = torch.softmax(logits[:, digit_ids], dim=-1)           # renormalised over the 5 valid answers
    probs.append(p.cpu().numpy())
    if i % 200 == 0:
        print(i, len(chats), "digit mass:", round(float(torch.softmax(logits, -1)[:, digit_ids].sum(-1).mean()), 3), flush=True)
P = np.concatenate(probs)
llm_df = pd.DataFrame(P, columns=[f"llm_p{d}" for d in range(1, 6)])
llm_df["llm_expected"] = P @ np.arange(1, 6)
llm_df.insert(0, "key", all_keys)
llm_df.to_csv("/kaggle/working/llm_judge.csv", index=False)
del llm; torch.cuda.empty_cache()
L = llm_df.set_index("key").loc[keys].reset_index(drop=True)

# ---- 2. evaluation (same hand-made features as features.py; same model and folds as v6) -------------------------------
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "er", "ah", "like"}
norm_words = lambda s: re.findall(r"[a-z0-9']+", s.lower())
gec = json.loads(find("gec.json").read_text())


def base_features(rec, pairs):
    text, seg = rec["text"], rec["segments"]
    words = re.findall(r"[a-z']+", text.lower()); n = max(len(words), 1)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    f = {"n_words": len(words), "words_per_sec": len(words) / rec["duration"],
         "mean_sentence_len": len(words) / max(len(sentences), 1), "type_token_ratio": len(set(words)) / n,
         "filler_rate": sum(w in FILLERS for w in words) / n,
         "repeat_rate": sum(a == b for a, b in zip(words, words[1:])) / n,
         "mean_word_len": np.mean([len(w) for w in words]) if words else 0.0,
         "non_ascii_rate": sum(ord(c) > 127 for c in text) / max(len(text), 1),
         "avg_logprob": np.mean([s["avg_logprob"] for s in seg]) if seg else -2.0,
         "no_speech_prob": np.mean([s["no_speech_prob"] for s in seg]) if seg else 1.0, "rms": rec["rms"]}
    edits = nw = changed = 0
    for orig, corr in pairs:
        a, b = norm_words(orig), norm_words(corr)
        e = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
                difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal")
        edits, nw, changed = edits + e, nw + len(a), changed + (e > 0)
    f.update(gec_edit_rate=edits / max(nw, 1), gec_changed_frac=changed / max(len(pairs), 1))
    return f


base = pd.DataFrame([base_features(recs[k], gec[k]) for k in keys])
A_npz = np.load(find("audio_emb.npz")); A = np.stack([A_npz[k].mean(0) for k in keys])
deb = train.filename.map(pd.read_csv(find("oof_deberta_large.csv")).set_index("filename").iloc[:, -1]).to_numpy()
cv = KFold(5, shuffle=True, random_state=42)
rmse = lambda p, m=slice(None): float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
svr = lambda X: np.clip(cross_val_predict(make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05)), X, y, cv=cv), 0, 5)
ridge = lambda X: np.clip(cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))), X, y, cv=cv), 0, 5)
combine = lambda a, t: np.where((a < 1) | np.isnan(t), a, 0.5 * a + 0.5 * np.nan_to_num(t))


def spearman(values):
    """Spearman with the score on clips scored 1-5, paired by position."""
    return pd.Series(np.asarray(values, dtype=float)[ok]).corr(pd.Series(y[ok]), method="spearman")


print("\nLLM expected rating vs true score (clips 1-5): Spearman", round(spearman(L.llm_expected), 3),
      "| Pearson", round(float(np.corrcoef(L.llm_expected[ok], y[ok])[0, 1]), 3),
      "| for comparison, grammar-correction edit rate:", round(spearman(base.gec_edit_rate), 3))
print("mean LLM rating by true score:", L.assign(score=y).groupby("score").llm_expected.mean().round(2).to_dict())

print(f"\n{'feature set':<32} {'hand-made Ridge':>15} {'audio SVR':>10} {'ENSEMBLE all':>13} {'1-5':>7}")
res = {}
for name, H in [("v6 baseline (sanity check)", base), ("+ LLM expected rating", pd.concat([base, L[["llm_expected"]]], axis=1)),
                ("+ LLM rating + probabilities", pd.concat([base, L.drop(columns=[])], axis=1))]:
    a = svr(np.hstack([A, H.to_numpy()])); e = combine(a, deb); res[name] = (a, e)
    print(f"{name:<32} {rmse(ridge(H.to_numpy())):>15.4f} {rmse(a):>10.4f} {rmse(e):>13.4f} {rmse(e, ok):>7.4f}", flush=True)
print("expected v6 baseline: hand-made Ridge 0.9657 | audio SVR 0.5359 | ensemble 0.4963")

# LLM rating as a third model in the blend, its weight chosen by nested CV (outer folds never see their own weight)
a6, e6 = res["v6 baseline (sanity check)"]
llm_score = np.clip(L.llm_expected.to_numpy(), 1, 5)
out, ws = np.zeros(len(y)), []
for tr, va in KFold(5, shuffle=True, random_state=7).split(y):
    grid = np.linspace(0, 0.5, 11)
    blend = lambda w, idx: np.where(a6[idx] < 1, a6[idx], (1 - w) * e6[idx] + w * llm_score[idx])
    sub_rmse = lambda pred, idx: float(np.sqrt(np.mean((y[idx] - pred) ** 2)))   # pred is already the idx subset
    w = grid[np.argmin([sub_rmse(blend(w, tr), tr) for w in grid])]; ws.append(round(float(w), 2))
    out[va] = blend(w, va)
print(f"\nv6 ensemble + LLM rating as a third model (weight by nested CV, chosen per fold {ws}): all {rmse(out):.4f} | 1-5 {rmse(out, ok):.4f}")
