"""Kaggle GPU version of transcribe.py: same model (Whisper large-v3-turbo), prompt and output format."""
import json, subprocess, sys, wave, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-whisper"], check=True)
import numpy as np
import torch
import whisper

DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
OUT = Path("/kaggle/working/transcripts")
PROMPT = "Umm, so I was, uh, I go to the market and, like, I buyed some... some vegetables."


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


OUT.mkdir(parents=True, exist_ok=True)
print("data:", DATA, "cuda:", torch.cuda.is_available(), flush=True)
model = whisper.load_model("turbo", device="cuda")
files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
for i, f in enumerate(files):
    audio = load_wav(f)
    r = model.transcribe(audio, language="en", initial_prompt=PROMPT, condition_on_previous_text=False, fp16=True)
    segs = [{k: s[k] for k in ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")}
            for s in r["segments"]]
    rec = {"file": f.name, "split": f.parent.name, "duration": len(audio) / 16000,
           "rms": float(np.sqrt((audio ** 2).mean())), "text": r["text"].strip(), "segments": segs}
    (OUT / f"{f.parent.name}_{f.stem}.json").write_text(json.dumps(rec))
    if i % 50 == 0:
        print(i, len(files), f.name, rec["text"][:80], flush=True)

with zipfile.ZipFile("/kaggle/working/transcripts.zip", "w") as z:
    for p in OUT.glob("*.json"):
        z.write(p, p.name)
print("done", len(list(OUT.glob("*.json"))))
