# Grammar Scoring from Spoken Audio

This is my solution for the SHL Research Engineer Kaggle challenge (`shl-hiring-assessment-2026`).
The task is to take an audio file of someone speaking English for 45–60 seconds and output a grammar score from 0 to 5.

The full write-up, with plots and the training RMSE, is in **[`shl_grammar_scoring.ipynb`](shl_grammar_scoring.ipynb)**.
This README is the short version.

```bash
python predict.py clip.wav      # audio file in -> score (0-5) out
```

## How I approached it

My first thought was that grammar is about *words*, so I should turn the audio into text and work from there.
That worked okay, but the biggest jump actually came later, when I also used the audio itself.

**1. Speech to text (Whisper).**
I transcribed every clip with Whisper large-v3-turbo. One problem I noticed early: Whisper likes to "clean up" speech.
It drops the "uh"s and "um"s and sometimes smooths over mistakes, which is exactly what I needed to keep.
I found that if you give Whisper a starting prompt written in messy, broken English, it copies that style and transcribes
more faithfully. I tested this on 10 clips: with the prompt it kept 9 fillers, without it kept 0.
It still isn't perfect, but it helped.

**2. Simple features I could explain.**
From each transcript I computed things like speaking rate, number of words, repeated words ("I I", "the the")
and how confident Whisper was. The most useful one was a grammar-error signal: I ran each sentence through a
grammar-correction model (Grammarly's CoEdIT) and measured how many words it had to change.
For example, "if a market crowded" became "if a market **is** crowded".
More corrections usually meant a lower score.

**3. Using the audio itself (WavLM).**
This was the surprise. When I added WavLM speech embeddings, the error dropped a lot. Looking back it makes sense:
the raters were *listening*, so fluency, pauses and pronunciation affect the score, and a transcript loses all of that.
I feed the WavLM embedding plus my hand-made features into an **SVR** (support vector regression with an RBF kernel).
I call this the **audio model**. I started with Ridge regression, but it can only learn straight-line relationships; the SVR can
learn curved ones and did clearly better. I chose its settings with nested cross-validation (every fold picked the same ones),
so they weren't tuned on my own validation scores. I also tried gradient boosting and random forests, which did worse.

**4. Fine-tuning DeBERTa on the transcripts.**
I fine-tuned DeBERTa-v3-large to predict the score directly from the transcript. This is the **text model**.
I trained for a fixed number of epochs, rather than picking the best epoch on the validation fold,
so my cross-validation numbers wouldn't be over-optimistic. The large model beat the base one (and beat averaging 3 base runs).

**5. Combining them, and handling score 0.**
37 training clips have a score of **0**. They aren't silent: they're speech that Whisper can barely understand, so I think
they're unintelligible or off-topic recordings. At first I dropped them, but the task asks for a 0–5 score, and it turned out
the audio model recognises them: in cross-validation it predicts **below 1 for 36 of the 37** and **above 1 for all 732**
real speakers. So the final rule is:

- if the audio model predicts below 1 → the speech is unintelligible, use the audio model's score (close to 0);
- otherwise → a simple 50/50 average of the audio model and the text model. I didn't tune the weight, to keep it simple.

## Results

All numbers are 5-fold cross-validation on the training data, so every clip is predicted by a model that never saw it.

| what I tried | CV RMSE (all clips) | CV RMSE (score 1–5) | CV Pearson |
|---|---|---|---|
| hand-made features only (Ridge) | 0.966 | 0.854 | 0.627 |
| WavLM audio + hand-made features (Ridge) | 0.563 | 0.576 | 0.891 |
| WavLM audio + hand-made features (**SVR**, audio model) | 0.536 | 0.543 | 0.903 |
| fine-tuned DeBERTa-v3-large (text model) | – | 0.599 | 0.815 |
| **final: gate + 50/50 average** | **0.496** | **0.502** | **0.918** |

- **Training RMSE** (audio model fitted on all training data): **0.109**. It's much lower than the CV number because an SVR can fit
  the points it was trained on very closely, which is exactly why I judge everything by cross-validation instead.
- **Public leaderboard RMSE: 0.3510** (version 6, rank 25 when submitted). The competition evaluates with Pearson and RMSE;
  the leaderboard ranks by RMSE, so I used RMSE to make decisions and checked that Pearson improved too.
  The step-by-step history is in the notebook.

Things I tried that didn't make the final model: averaging several DeBERTa-base runs (tiny gain, the large model was better),
and WavLM-large (better on its own, but no gain inside the final ensemble, so I kept the smaller, faster base model).

## What's in this repo

| file / folder | what it is |
|---|---|
| `shl_grammar_scoring.ipynb` | the main notebook: explanation, plots, evaluation, final predictions, `predict.py` demo |
| `predict.py` | **audio file in → score out**, running the whole pipeline end to end |
| `features.py` | builds the features from transcripts (shared by training and `predict.py`) |
| `train.py` | the audio model (SVR) and its cross-validation; saves `models/audio_model.joblib` |
| `transcribe.py` | runs Whisper locally on a Mac (Apple Silicon) |
| `kaggle_*/` | scripts I ran on Kaggle's free GPU (Whisper, grammar correction, WavLM, DeBERTa) |
| `submission.csv` | my final predictions for the test set |

**What's deliberately not here:** the competition rules don't allow sharing the data or anything derived from it, so the
audio, transcripts, grammar corrections, embeddings and per-clip predictions are not in the repo. The trained models aren't
either: the SVR stores feature vectors of training clips inside it, and the fine-tuned DeBERTa-large is 1.7 GB.
The steps below re-create all of them.

I ran the heavy parts on Kaggle because my laptop (an M1 MacBook Air) was overheating while transcribing.

## How to reproduce

1. **Set up and get the data** (you need to join the competition on Kaggle first):
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   kaggle competitions download -c shl-hiring-assessment-2026 -p data && unzip -q data/*.zip -d data
   ```
2. **Run the GPU steps on Kaggle.** In each `kaggle_*/kernel-metadata.json`, change `vanshrajput29` to your Kaggle username.
   Then push them in this order (later ones read earlier outputs), waiting for each to finish:
   ```bash
   kaggle kernels push -p kaggle_asr            # Whisper transcripts -> transcripts.zip
   kaggle kernels push -p kaggle_gec            # grammar correction -> gec.json
   kaggle kernels push -p kaggle_audio          # WavLM embeddings   -> audio_emb.npz
   kaggle kernels push -p kaggle_deberta_large  # DeBERTa-large 5-fold predictions
   kaggle kernels push -p kaggle_deberta_final  # DeBERTa-large trained on all clips (for predict.py)
   ```
   (`kaggle_deberta` and `kaggle_deberta_seeds` are the base-model comparison runs shown in the notebook;
   `kaggle_audio_large` is the WavLM-large comparison.)
3. **Download the outputs** with `kaggle kernels output <your-username>/<notebook-name> -p <folder>` into:
   `transcripts/` (unzip `transcripts.zip` there), `gec.json` (project root), `kaggle_out/audio/`, `kaggle_out/deberta/`,
   `kaggle_out/deberta_seeds/`, `kaggle_out/deberta_large/`, `kaggle_out/audio_large/`, and `models/deberta_final/`.
4. **Train the audio model and run the notebook:**
   ```bash
   .venv/bin/python train.py
   .venv/bin/jupyter nbconvert --to notebook --execute shl_grammar_scoring.ipynb
   ```
5. **Score any audio file:** `.venv/bin/python predict.py clip.wav` (needs Apple Silicon for local Whisper;
   it downloads about 3 GB of pretrained models the first time).

## What I'd improve with more time

- **Whisper still cleans up some speech.** I can't measure exactly how much without human-written transcripts.
- **Very few clips score 1–2**, so the model is least accurate there and tends to predict towards the middle.
- **The audio model is frozen.** Fine-tuning WavLM itself, or training one model on audio and text together, is what I'd try next.
- **The score-0 rule** catches 36 of 37 here, but it's based on very few examples.
- **For production** I'd start from the audio model alone (CV RMSE 0.54 without Whisper or DeBERTa) and use lighter models;
  the notebook's last section has the details.

## Note on tools

SHL said AI tools were allowed, and I used an AI assistant to help write code and debug.
I made sure I understand every step, and I'm happy to walk through any of it.

Licensed under the MIT License (see `LICENSE`).
