"""Score a spoken-English audio file for grammar, 0-5.

    python predict.py clip.wav [more.wav ...]

Same pipeline as the leaderboard model, end to end on one file:
  1. Whisper large-v3-turbo transcript (disfluent prompt)            -- transcribe.py settings
  2. CoEdIT grammar correction of each sentence -> edit-rate features
  3. WavLM-base-plus embeddings (mean over time and layers): of the whole clip, and of each overlapping ~10 s piece
  4. SVR audio model scores every piece on [piece emb, clip emb, hand-made features]; the clip's audio score is the
     median over its pieces                                         -- models/audio_model.joblib (train.py)
  5. If the audio model says < 1 the clip is unintelligible -> return that score (~0) (train.combine_scores).
     Otherwise average it with the fine-tuned DeBERTa on the transcript -- models/deberta_final/
Needs Apple Silicon for Whisper (mlx-whisper). Downloads ~3 GB of pretrained models on first run.
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import features as F
import train
import transcribe as T

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SR, CHUNK = T.SR, T.SR * 20
MODELS = Path(__file__).parent / "models"


# Each model is loaded once per process and reused, so scoring many files doesn't reload ~5 GB of weights each time.
@lru_cache(maxsize=None)
def _coedit():
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    return (AutoTokenizer.from_pretrained("grammarly/coedit-large"),
            T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large").to(DEVICE).eval())


@lru_cache(maxsize=None)
def _wavlm():
    from transformers import AutoFeatureExtractor, WavLMModel
    return (AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus"),
            WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval())


@lru_cache(maxsize=None)
def _deberta():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = MODELS / "deberta_final"
    return (AutoTokenizer.from_pretrained(path),
            AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval())


@lru_cache(maxsize=None)
def _audio_model():
    return joblib.load(MODELS / "audio_model.joblib")  # made locally by train.py (pickle: only load trusted files)


def grammar_pairs(text):
    sents = F.gec_sentences(text)
    if not sents:
        return []
    tok, m = _coedit()
    enc = tok(["Fix grammatical errors in this sentence: " + s for s in sents], return_tensors="pt",
              padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = tok.batch_decode(m.generate(**enc, max_new_tokens=256, num_beams=1), skip_special_tokens=True)
    return list(zip(sents, out))


def wavlm_embedding(audio):
    """Same as kaggle_audio/: mean over time (20 s chunks) of every hidden layer, then mean over layers."""
    fe, m = _wavlm()
    sums, n = None, 0
    for s in range(0, len(audio), CHUNK):
        seg = audio[s:s + CHUNK]
        if len(seg) < SR:
            continue
        with torch.no_grad():
            hs = torch.stack(m(fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE),
                               output_hidden_states=True).hidden_states)
        layer_sum = hs[:, 0].sum(1).cpu().numpy()
        sums, n = (layer_sum if sums is None else sums + layer_sum), n + hs.shape[2]
    return (sums / n).mean(0)


def wavlm_piece_embeddings(audio):
    """Same as kaggle_audio_pieces_variants/ (p10hop5): each overlapping ~10 s piece (transcribe.split_pieces) on its own,
    mean over time, then layers."""
    fe, m = _wavlm()
    out = []
    for seg in T.split_pieces(audio):
        with torch.no_grad():
            hs = torch.stack(m(fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE),
                               output_hidden_states=True).hidden_states)
        out.append(hs[:, 0].mean(1).mean(0).cpu().numpy())
    return np.stack(out)


def deberta_score(text):
    tok, m = _deberta()
    with torch.no_grad():
        p = m(**tok(text, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)).logits.item()
    return float(np.clip(p * 4 + 1, 1, 5))  # trained on (score - 1) / 4


def score(path, verbose=False):
    audio = T.load_audio(path)
    if len(audio) < SR:  # training clips were 20-61 s (mostly 45-60 s); WavLM needs at least 1 s
        raise ValueError(f"{path}: clip is {len(audio) / SR:.2f} s long; need at least 1 s of audio")
    rec = T.transcribe_clip(audio)
    H = F.handcrafted(pd.DataFrame([F.asr_fields(rec)]), gec_pairs=[grammar_pairs(rec["text"])])
    audio_model = _audio_model()
    if list(H.columns) != audio_model["columns"]:
        raise RuntimeError("feature columns differ from the ones the audio model was trained on - re-run train.py")
    clip_emb, h = wavlm_embedding(audio), H.to_numpy()[0]
    rows = np.vstack([np.concatenate([p, clip_emb, h]) for p in wavlm_piece_embeddings(audio)])
    r = float(np.clip(np.median(audio_model["model"].predict(rows)), 0, 5))   # clip score = median over pieces
    # The text model is only needed for intelligible speech (see train.combine_scores).
    d = deberta_score(rec["text"]) if r >= train.UNINTELLIGIBLE_BELOW else np.nan
    final = float(train.combine_scores(r, d))
    if verbose:
        text_score = "not used (unintelligible)" if np.isnan(d) else f"{d:.2f}"
        print(f"  transcript: {rec['text'][:150]}...\n  audio model {r:.2f} | text model {text_score}")
    return final


if __name__ == "__main__":
    if not sys.argv[1:]:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        print(f"{p}: {score(p, verbose=True):.2f}")
