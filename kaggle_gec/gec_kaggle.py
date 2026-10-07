"""Grammar-correct every transcript sentence with CoEdIT (Kaggle GPU). Output: {key: [[original, corrected], ...]}."""
import json, re
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

SRC = next(Path("/kaggle/input").rglob("transcripts.zip")).parent
MODEL = "grammarly/coedit-large"
INSTRUCTION = "Fix grammatical errors in this sentence: "

import zipfile
recs = {}
with zipfile.ZipFile(SRC / "transcripts.zip") as z:
    for n in z.namelist():
        recs[Path(n).stem] = json.loads(z.read(n))["text"]


def sentences(text):
    # Whisper sometimes emits ". . . ." on silence; keep only real sentences.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 2]


tok = AutoTokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

jobs = [(k, s) for k, t in recs.items() for s in sentences(t)]
print(len(recs), "transcripts", len(jobs), "sentences", flush=True)
out = {k: [] for k in recs}
B = 64
for i in range(0, len(jobs), B):
    batch = jobs[i:i + B]
    enc = tok([INSTRUCTION + s for _, s in batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=256, num_beams=1)
    for (k, s), c in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
        out[k].append([s, c])
    if i % (B * 20) == 0:
        print(i, batch[0][1][:80], "->", tok.decode(gen[0], skip_special_tokens=True)[:80], flush=True)

Path("/kaggle/working/gec.json").write_text(json.dumps(out))
print("done")
