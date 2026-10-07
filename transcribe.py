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
        assert w.getsampwidth() == 2, f"expects 16-bit PCM WAV: {path}"
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        x, sr = x.reshape(-1, w.getnchannels()).mean(axis=1), w.getframerate()
    return resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x


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
