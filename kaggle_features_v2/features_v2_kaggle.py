"""Experiment (Kaggle, CPU): do grammatical-complexity and error-type features improve the v6 model?

The rubric rewards complex grammar used accurately, but the existing grammar feature only counts errors. Adds:
  - complexity (spaCy parse): subordinate-clause rate, parse-tree depth, verb-form variety, passive and
    coordination rates, mean clause length
  - error types (from the existing CoEdIT corrections): missing-word, extra-word and wrong-word rates
Evaluated exactly like v6: SVR(C=3, eps=0.05) on [WavLM + hand-made features], KFold(5, shuffle, 42), combined with
the DeBERTa-large out-of-fold predictions by the same rule. The v6 baseline is recomputed first as a sanity check.
Inputs come from earlier Kaggle runs (kernel_sources), so nothing is uploaded.
"""
import difflib, json, re, subprocess, sys, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spacy"], check=True)
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
import numpy as np
import pandas as pd
import spacy
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

INP = Path("/kaggle/input")
find = lambda name: next(INP.rglob(name))
DATA = find("train.csv").parent
train = pd.read_csv(DATA / "train.csv")
y = train.label.to_numpy(); ok = y > 0
with zipfile.ZipFile(find("transcripts.zip")) as z:
    recs = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()}
gec = json.loads(find("gec.json").read_text())
A_npz = np.load(find("audio_emb.npz"))
deb = train.filename.map(pd.read_csv(find("oof_deberta_large.csv")).set_index("filename").iloc[:, -1]).to_numpy()
keys = ["train_" + Path(f).stem for f in train.filename]

# ---- existing hand-made features: an exact copy of features.py (asr_fields, text_features, gec_features) ----------
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "er", "ah", "like"}
norm_words = lambda s: re.findall(r"[a-z0-9']+", s.lower())


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


# ---- new features ---------------------------------------------------------------------------------------------------
def error_types(pairs):
    """Word-level edits CoEdIT made, split by type, per original word."""
    ins = dele = rep = nw = 0
    for orig, corr in pairs:
        a, b = norm_words(orig), norm_words(corr); nw += len(a)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "insert": ins += j2 - j1           # correction added words  -> speaker left words out
            elif op == "delete": dele += i2 - i1        # correction removed words -> extra / repeated words
            elif op == "replace":                       # wrong form: "buyed" -> "bought"; an unequal-length block
                la, lb = i2 - i1, j2 - j1                # also hides missing/extra words ("go school" -> "goes to school")
                rep += min(la, lb); ins += max(0, lb - la); dele += max(0, la - lb)
    nw = max(nw, 1)
    return {"err_missing_rate": ins / nw, "err_extra_rate": dele / nw, "err_wrong_rate": rep / nw}


SUBORD = {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj"}
VERB_TAGS = {"VBD", "VBZ", "VBP", "VBN", "VBG", "MD", "VB"}
nlp = spacy.load("en_core_web_sm", disable=["ner"])


def depth(tok):
    # spaCy builds a new Token object on every .head access, so `tok.head is tok` is never true at the root
    # (the first version looped forever on it); ancestors walks up to the root safely.
    return sum(1 for _ in tok.ancestors)


def complexity(doc):
    sents = [s for s in doc.sents if sum(t.is_alpha for t in s) >= 2] or [doc[:]]
    toks = [t for t in doc if t.is_alpha]; n = max(len(toks), 1); ns = len(sents)
    clauses = sum(t.dep_ in SUBORD for t in doc) + ns
    return {"subord_per_sent": sum(t.dep_ in SUBORD for t in doc) / ns,
            "max_depth_mean": np.mean([max((depth(t) for t in s), default=0) for s in sents]),
            "verb_form_variety": len({t.tag_ for t in doc if t.tag_ in VERB_TAGS}) / len(VERB_TAGS),
            "passive_rate": sum(t.dep_ in ("nsubjpass", "auxpass") for t in doc) / n,
            "coord_rate": sum(t.dep_ == "conj" for t in doc) / n,
            "words_per_clause": n / clauses}


base = pd.DataFrame([base_features(recs[k], gec[k]) for k in keys])
errs = pd.DataFrame([error_types(gec[k]) for k in keys])
comp = pd.DataFrame([complexity(d) for d in nlp.pipe([recs[k]["text"] for k in keys], batch_size=64)])
A = np.stack([A_npz[k].mean(0) for k in keys])
print("features:", base.shape[1], "base +", errs.shape[1], "error-type +", comp.shape[1], "complexity", flush=True)

# ---- evaluation ---------------------------------------------------------------------------------------------------
cv = KFold(5, shuffle=True, random_state=42)
rmse = lambda p, m=slice(None): float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
svr = lambda X: np.clip(cross_val_predict(make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05)), X, y, cv=cv), 0, 5)
ridge = lambda X: np.clip(cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))), X, y, cv=cv), 0, 5)
combine = lambda a, t: np.where((a < 1) | np.isnan(t), a, 0.5 * a + 0.5 * np.nan_to_num(t))


def spearman(values):
    """Spearman with the score on clips scored 1-5, paired by position. (The first version used
    v[ok].corr(pd.Series(y[ok])): pandas aligned the two by index labels, which differ after filtering,
    so it paired the wrong rows and printed near-zero correlations.)"""
    return pd.Series(np.asarray(values, dtype=float)[ok]).corr(pd.Series(y[ok]), method="spearman")


print("\nSpearman with score (clips scored 1-5):")
for col in list(errs.columns) + list(comp.columns):
    v = pd.concat([errs, comp], axis=1)[col]
    print(f"  {col:<20} {spearman(v):+.3f}")

print(f"\n{'feature set':<34} {'hand-made Ridge':>15} {'audio SVR':>10} {'ENSEMBLE all':>13} {'1-5':>7}")
for name, H in [("v6 baseline (sanity check)", base), ("+ error types", pd.concat([base, errs], axis=1)),
                ("+ complexity", pd.concat([base, comp], axis=1)), ("+ both", pd.concat([base, errs, comp], axis=1))]:
    a = svr(np.hstack([A, H.to_numpy()])); e = combine(a, deb)
    print(f"{name:<34} {rmse(ridge(H.to_numpy())):>15.4f} {rmse(a):>10.4f} {rmse(e):>13.4f} {rmse(e, ok):>7.4f}", flush=True)
print("expected v6 baseline: hand-made Ridge 0.9657 | audio SVR 0.5359 | ensemble 0.4963")
pd.concat([pd.Series(keys, name="key"), errs, comp], axis=1).to_csv("/kaggle/working/features_v2.csv", index=False)
