"""Transcribe every train/test clip with Whisper (MLX, Apple Silicon). Resumable: one JSON per clip."""
import json, sys, wave
from pathlib import Path

import mlx_whisper
import numpy as np

DATA = Path("data/Dataset_Final")
OUT = Path("transcripts")
MODEL = "mlx-community/whisper-large-v3-turbo"  # override: python transcribe.py <hf-model>
# Whisper tends to "clean up" speech; a disfluent prompt nudges it to keep fillers and errors verbatim.
PROMPT = "Umm, so I was, uh, I go to the market and, like, I buyed some... some vegetables."


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


def main(model=MODEL):
    OUT.mkdir(exist_ok=True)
    files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
    todo = [f for f in files if not (OUT / f"{f.parent.name}_{f.stem}.json").exists()]
    print(f"{len(todo)}/{len(files)} to transcribe with {model}", flush=True)
    for i, f in enumerate(todo):
        audio = load_wav(f)
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language="en",
                                   initial_prompt=PROMPT, condition_on_previous_text=False)
        segs = [{k: s[k] for k in ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")}
                for s in r["segments"]]
        rec = {"file": f.name, "split": f.parent.name, "duration": len(audio) / 16000,
               "rms": float(np.sqrt((audio ** 2).mean())), "text": r["text"].strip(), "segments": segs}
        (OUT / f"{f.parent.name}_{f.stem}.json").write_text(json.dumps(rec))
        if i % 25 == 0:
            print(i, f.name, rec["text"][:80], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
