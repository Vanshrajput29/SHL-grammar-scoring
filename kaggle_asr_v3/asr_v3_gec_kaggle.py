"""Experiment: re-transcribe every clip with Whisper large-v3 (full model, not turbo), then grammar-correct the
new transcripts with CoEdIT - one Kaggle GPU run. Everything else is identical to kaggle_asr/ + kaggle_gec/
(same prompt, decoding settings, record format, sentence splitting and correction settings), so the transcription
model is the only thing that changes. Outputs: transcripts_v3.zip, gec_v3.json."""
import gc, json, re, subprocess, sys, wave, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openai-whisper"], check=True)
import numpy as np
import torch
import whisper
from transformers import AutoTokenizer, T5ForConditionalGeneration

DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
OUT = Path("/kaggle/working/transcripts_v3")
ASR_MODEL = "large-v3"
PROMPT = "Umm, so I was, uh, I go to the market and, like, I buyed some... some vegetables."
GEC_MODEL, INSTRUCTION = "grammarly/coedit-large", "Fix grammatical errors in this sentence: "


def load_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels()).mean(axis=1)


def sentences(text):
    # Same rule as kaggle_gec/ and features.gec_sentences: drop Whisper's ". . ." noise.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 2]


# --- 1. transcription -------------------------------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
asr = whisper.load_model(ASR_MODEL, device="cuda")
files = sorted((DATA / "train").glob("*.wav")) + sorted((DATA / "test").glob("*.wav"))
texts = {}
for i, f in enumerate(files):
    audio = load_wav(f)
    r = asr.transcribe(audio, language="en", initial_prompt=PROMPT, condition_on_previous_text=False, fp16=True)
    segs = [{k: s[k] for k in ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")}
            for s in r["segments"]]
    rec = {"file": f.name, "split": f.parent.name, "duration": len(audio) / 16000,
           "rms": float(np.sqrt((audio ** 2).mean())), "text": r["text"].strip(), "segments": segs}
    key = f"{f.parent.name}_{f.stem}"
    (OUT / f"{key}.json").write_text(json.dumps(rec)); texts[key] = rec["text"]
    if i % 50 == 0:
        print(i, len(files), f.name, rec["text"][:80], flush=True)
with zipfile.ZipFile("/kaggle/working/transcripts_v3.zip", "w") as z:
    for p in OUT.glob("*.json"):
        z.write(p, p.name)
print("transcribed", len(texts), flush=True)
del asr; gc.collect(); torch.cuda.empty_cache()

# --- 2. grammar correction ------------------------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(GEC_MODEL)
gec = T5ForConditionalGeneration.from_pretrained(GEC_MODEL, torch_dtype=torch.float16).cuda().eval()
jobs = [(k, s) for k, t in texts.items() for s in sentences(t)]
out = {k: [] for k in texts}
for i in range(0, len(jobs), 64):
    batch = jobs[i:i + 64]
    enc = tok([INSTRUCTION + s for _, s in batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        gen = gec.generate(**enc, max_new_tokens=256, num_beams=1)
    for (k, s), c in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
        out[k].append([s, c])
Path("/kaggle/working/gec_v3.json").write_text(json.dumps(out))
print("done:", len(out), "transcripts,", len(jobs), "sentences corrected")
