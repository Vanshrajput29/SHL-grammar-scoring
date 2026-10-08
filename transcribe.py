"""Transcribe every train/test clip with Whisper (MLX, Apple Silicon). Resumable: one JSON per clip."""
import json, sys, wave
from pathlib import Path

import mlx_whisper
import numpy as np
from scipy.signal import resample_poly

DATA = Path("data/Dataset_Final")
OUT = Path("transcripts")
SR = 16000
MODEL = "mlx-community/whisper-large-v3-turbo"  # override: python transcribe.py <hf-model>
# Whisper tends to "clean up" speech; a disfluent prompt nudges it to keep fillers and errors verbatim.
PROMPT = "Umm, so I was, uh, I go to the market and, like, I buyed some... some vegetables."


def load_audio(path):
    """16 kHz mono float32 from a 16-bit PCM WAV file (resampled if needed). Shared with predict.py."""
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:  # a real check, not an assert: asserts vanish under `python -O`
            raise ValueError(f"expects 16-bit PCM WAV, got {8 * w.getsampwidth()}-bit: {path}")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        x, sr = x.reshape(-1, w.getnchannels()).mean(axis=1), w.getframerate()
    return resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x


PIECE_S, HOP_S = 10, 5   # the audio model scores overlapping 10 s pieces, one starting every 5 s (see split_pieces)


def split_pieces(x):
    """Cut audio into overlapping ~10 s pieces for the audio model, one starting every 5 s. If what's left after the
    last full piece is at least half a piece, one more piece ending at the end of the clip is added; otherwise the last
    piece is stretched to the end. Audio of 10 s or less is a single piece. Same rule as kaggle_audio_pieces_variants/
    ("p10hop5", where the training embeddings were made), so training and predict.py cut clips identically."""
    piece, hop = PIECE_S * SR, HOP_S * SR
    if len(x) <= piece:
        return [x]
    segs = [(s, s + piece) for s in range(0, len(x) - piece + 1, hop)]
    if len(x) - segs[-1][1] >= piece // 2:
        segs.append((len(x) - piece, len(x)))
    else:
        segs[-1] = (segs[-1][0], len(x))
    return [x[a:b] for a, b in segs]


def transcribe_clip(audio, model=MODEL):
    """Whisper transcript + ASR metadata for one clip. The single place the Whisper settings live,
    so training transcripts and predict.py can never drift apart."""
    r = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language="en",
                               initial_prompt=PROMPT, condition_on_previous_text=False)
    segs = [{k: s[k] for k in ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")}
            for s in r["segments"]]
    return {"duration": len(audio) / SR, "rms": float(np.sqrt((audio ** 2).mean())),
            "text": r["text"].strip(), "segments": segs}


def main(model=MODEL):
    OUT.mkdir(exist_ok=True)
    files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
    todo = [f for f in files if not (OUT / f"{f.parent.name}_{f.stem}.json").exists()]
    print(f"{len(todo)}/{len(files)} to transcribe with {model}", flush=True)
    for i, f in enumerate(todo):
        rec = {"file": f.name, "split": f.parent.name, **transcribe_clip(load_audio(f), model)}
        (OUT / f"{f.parent.name}_{f.stem}.json").write_text(json.dumps(rec))
        if i % 25 == 0:
            print(i, f.name, rec["text"][:80], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
