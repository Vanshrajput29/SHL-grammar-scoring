"""Score a spoken-English audio file for grammar, 0-5.

    python predict.py clip.wav [more.wav ...]

Same pipeline as the leaderboard model, end to end on one file:
  1. Whisper large-v3-turbo transcript (disfluent prompt)            -- transcribe.py settings
  2. CoEdIT grammar correction of each sentence -> edit-rate features
  3. WavLM-base-plus embedding (mean over time and layers)
  4. SVR audio model on [audio + hand-crafted features]             -- models/audio_model.joblib (train.py)
  5. If the audio model says < 1 the clip is unintelligible -> return that score (~0).
     Otherwise average it with the fine-tuned DeBERTa on the transcript -- models/deberta_final/
Needs Apple Silicon for Whisper (mlx-whisper). Downloads ~3 GB of pretrained models on first run.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import features as F
import transcribe as T

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SR, CHUNK = T.SR, T.SR * 20
MODELS = Path(__file__).parent / "models"


def grammar_pairs(text):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    sents = F.gec_sentences(text)
    if not sents:
        return []
    tok = AutoTokenizer.from_pretrained("grammarly/coedit-large")
    m = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large").to(DEVICE).eval()
    enc = tok(["Fix grammatical errors in this sentence: " + s for s in sents], return_tensors="pt",
              padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = tok.batch_decode(m.generate(**enc, max_new_tokens=256, num_beams=1), skip_special_tokens=True)
    return list(zip(sents, out))


def wavlm_embedding(audio):
    """Same as kaggle_audio/: mean over time (20 s chunks) of every hidden layer, then mean over layers."""
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval()
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


def deberta_score(text):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = MODELS / "deberta_final"
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()
    with torch.no_grad():
        p = m(**tok(text, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)).logits.item()
    return float(np.clip(p * 4 + 1, 1, 5))  # trained on (score - 1) / 4


def score(path, verbose=False):
    audio = T.load_audio(path)
    rec = T.transcribe_clip(audio)
    row = pd.DataFrame([F.asr_fields(rec)])
    H = F.handcrafted(row, gec_pairs=[grammar_pairs(rec["text"])])
    audio_model = joblib.load(MODELS / "audio_model.joblib")
    assert list(H.columns) == audio_model["columns"], "feature columns differ from training"
    x = np.hstack([wavlm_embedding(audio)[None], H.to_numpy()])
    r = float(np.clip(audio_model["model"].predict(x)[0], 0, 5))
    if r < 1:  # below the rubric's minimum: unintelligible / off-task speech
        final, d = r, None
    else:
        d = deberta_score(rec["text"])
        final = 0.5 * r + 0.5 * d
    if verbose:
        print(f"  transcript: {rec['text'][:150]}...\n  audio model {r:.2f} | text model {d if d is None else round(d, 2)}")
    return final


if __name__ == "__main__":
    for p in sys.argv[1:] or sys.exit(__doc__):
        print(f"{p}: {score(p, verbose=True):.2f}")
