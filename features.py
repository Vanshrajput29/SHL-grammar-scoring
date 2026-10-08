"""Turn Whisper transcripts into model features: interpretable fluency/grammar proxies + a sentence embedding."""
import json, re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/Dataset_Final")
TRANSCRIPTS = Path("transcripts")
FILLERS = {"uh", "um", "umm", "uhm", "hmm", "er", "ah", "like"}
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


def asr_fields(rec):
    """Transcript record (as written by transcribe.py) -> text + ASR metadata used as features."""
    seg = rec["segments"]
    return {"text": rec["text"], "duration": rec["duration"], "rms": rec["rms"],
            "avg_logprob": np.mean([s["avg_logprob"] for s in seg]) if seg else -2.0,
            "no_speech_prob": np.mean([s["no_speech_prob"] for s in seg]) if seg else 1.0}


def load(split):
    """DataFrame of one split: filename, label (train only), transcript and ASR metadata."""
    df = pd.read_csv(DATA / f"{split}.csv")
    recs = [asr_fields(json.loads((TRANSCRIPTS / f"{split}_{Path(f).stem}.json").read_text())) for f in df.filename]
    return pd.concat([df, pd.DataFrame(recs, index=df.index)], axis=1)


def text_features(text, duration):
    words = re.findall(r"[a-z']+", text.lower())
    n = max(len(words), 1)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "n_words": len(words),
        "words_per_sec": len(words) / duration,
        "mean_sentence_len": len(words) / max(len(sentences), 1),
        "type_token_ratio": len(set(words)) / n,
        "filler_rate": sum(w in FILLERS for w in words) / n,
        "repeat_rate": sum(a == b for a, b in zip(words, words[1:])) / n,  # "I I", "the the"
        "mean_word_len": np.mean([len(w) for w in words]) if words else 0.0,
        # Whisper emits stray non-English script on unintelligible speech (common in label-0 clips)
        "non_ascii_rate": sum(ord(c) > 127 for c in text) / max(len(text), 1),
    }


def gec_sentences(text):
    """Sentences sent to the grammar-correction model (same rule as kaggle_gec/). Drops Whisper's ". . ." noise."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 2]


def _norm_words(s):
    return re.findall(r"[a-z0-9']+", s.lower())  # ignore case/punctuation changes: only real word edits count


def gec_features(pairs):
    """pairs = [(original_sentence, corrected_sentence)] from CoEdIT. More edits needed -> worse grammar."""
    import difflib
    edits = words = changed = 0
    for orig, corr in pairs:
        a, b = _norm_words(orig), _norm_words(corr)
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        e = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        edits, words, changed = edits + e, words + len(a), changed + (e > 0)
    return {"gec_edit_rate": edits / max(words, 1), "gec_changed_frac": changed / max(len(pairs), 1)}


def handcrafted(df, split=None, gec_pairs=None):
    """gec_pairs: one list of (original, corrected) sentence pairs per row; by default read from gec.json for `split`."""
    f = pd.DataFrame([text_features(t, d) for t, d in zip(df.text, df.duration)], index=df.index)
    parts = [f, df[["avg_logprob", "no_speech_prob", "rms"]]]
    if gec_pairs is None and split and Path("gec.json").exists():
        gec = json.loads(Path("gec.json").read_text())
        gec_pairs = [gec[f"{split}_{Path(n).stem}"] for n in df.filename]
    if gec_pairs is not None:
        parts.append(pd.DataFrame([gec_features(p) for p in gec_pairs], index=df.index))
    return pd.concat(parts, axis=1)


def audio(df, split, path="kaggle_out/audio/audio_emb.npz"):
    """WavLM-base-plus hidden states, mean over time then over all 13 layers (made on Kaggle GPU)."""
    A = np.load(path)
    v = [A[f"{split}_{Path(f).stem}"] for f in df.filename]
    return np.stack([x.mean(0) if x.ndim == 2 else x for x in v])  # large file is stored pre-averaged over layers


def audio_pieces(df, split, path="kaggle_out/audio_pieces_variants/audio_pieces_p10hop5.npz"):
    """WavLM-base-plus embedding of every overlapping ~10 s piece of each clip (transcribe.split_pieces; Kaggle GPU):
    one array [n_pieces, 768] per clip, mean over time then over all 13 layers."""
    P = np.load(path)
    return [P[f"{split}_{Path(f).stem}"] for f in df.filename]


def embed(texts):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL).encode([t or " " for t in texts], batch_size=16,
                                                   normalize_embeddings=True, show_progress_bar=False)


if __name__ == "__main__":
    f = text_features("I I go to the, uh, market. I buyed some vegetables.", 5.0)
    assert f["n_words"] == 11 and f["repeat_rate"] == 1 / 11 and f["filler_rate"] == 1 / 11, f
    assert f["mean_sentence_len"] == 11 / 2
    g = gec_features([("I buyed some vegetables.", "I bought some vegetables."), ("It is good.", "It's good.")])
    assert g["gec_changed_frac"] == 1.0 and g["gec_edit_rate"] == 3 / 7, g  # 1 substitution + "it is"->"it's" (2)
    print("ok", f, g)
