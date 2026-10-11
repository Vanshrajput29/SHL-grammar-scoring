# Grammar Scoring from Spoken Audio

This is my solution for the SHL Research Engineer Kaggle challenge (`shl-hiring-assessment-2026`).
The task is to take an audio file of someone speaking English (mostly 45–60 seconds) and output a grammar score from 0 to 5.

The full write-up, with plots and the training RMSE, is in **[`shl_grammar_scoring.ipynb`](shl_grammar_scoring.ipynb)**.
This README is the short version.

```bash
python predict.py clip.wav      # audio file in -> score (0-5) out
```

## How I approached it

![Pipeline: Whisper transcript feeds a fine-tuned DeBERTa (text model) and hand-made features; WavLM speech embeddings of the clip and of each overlapping 10 s piece, plus those features, feed an SVR (audio model) that scores each piece and takes the median; the two scores are combined](pipeline.png)

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

Later I found the biggest remaining limit: the audio model learned from just **769 rows**, one per clip. So I cut each clip
into **overlapping 10-second pieces** (one every 5 s), each inheriting its clip's score: about **7,700 rows** from the same audio.
Each piece is scored on [its own WavLM embedding, the whole clip's embedding, the clip's hand-made features], and the clip's
audio score is the **median** of its pieces. Cross-validation keeps all pieces of a clip, and all clips of a speaker (see
"The real problem" below), in the same fold. Later, re-tuned with speaker-grouped folds, a **smoother SVR kernel** (half of
scikit-learn's default width) generalised better to new speakers: that's the final model, **v15**.

**4. Fine-tuning DeBERTa on the transcripts.**
I fine-tuned DeBERTa-v3-large to predict the score directly from the transcript. This is the **text model**.
I trained for a fixed number of epochs, rather than picking the best epoch on the validation fold,
so my cross-validation numbers wouldn't be over-optimistic. The large model beat the base one (and beat averaging 3 base runs).

**5. Combining them, and handling score 0.**
37 training clips have a score of **0**. They aren't silent: they're speech that Whisper can barely understand, so I think
they're unintelligible or off-topic recordings. At first I dropped them, but the task asks for a 0–5 score, and it turned out
the audio model recognises them: in cross-validation it predicts **below 1 for all 37** and **above 1 for all 732**
real speakers. So the final rule is:

- if the audio model predicts below 1 → the speech is unintelligible, use the audio model's score (close to 0);
- otherwise → a simple 50/50 average of the audio model and the text model. I didn't tune the weight, to keep it simple.

## Results

All numbers are 5-fold cross-validation on the training data, with folds **grouped by speaker**: no speaker's answers are
ever split between training and validation, because the test speakers are all new. The last column is the clip-split CV I
used during development, which was too optimistic (it partly rewarded recognising a speaker's voice).

| what I tried | CV RMSE (all clips) | CV RMSE (score 1–5) | CV Pearson | clip-split CV (all clips) |
|---|---|---|---|---|
| hand-made features only (Ridge) | 0.969 | 0.858 | 0.623 | 0.966 |
| WavLM audio + hand-made features (Ridge) | 0.610 | 0.624 | 0.870 | 0.563 |
| WavLM audio + hand-made features (**SVR**, one row per clip; v6) | 0.612 | 0.620 | 0.870 | 0.536 |
| same, trained on **overlapping 10 s pieces**, smoother kernel (audio model, v15) | 0.595 | 0.609 | 0.877 | 0.518 |
| fine-tuned DeBERTa-v3-large (text model) | – | 0.596 | 0.819 | – (0.599 on 1–5) |
| **final: gate + 50/50 average (v15)** | **0.521** | **0.533** | **0.909** | 0.484 |

- **Training RMSE** (audio model fitted on all training data): **0.158**. It's much lower than the CV number because an SVR can
  fit the points it was trained on closely (and each training clip is seen as several overlapping pieces), which is exactly
  why I judge everything by cross-validation instead.
- **Public leaderboard RMSE: 0.3428** (version 15). The path there: v6 0.3510 → v9 (pieces) 0.3477 → v10 (overlapping pieces)
  0.3466 → v15 (smoother kernel, chosen with speaker-grouped CV) 0.3428.
  The competition evaluates with Pearson and RMSE; the leaderboard ranks by RMSE, so I used RMSE to make decisions and checked
  that Pearson improved too. The step-by-step history is in the notebook.

Things I tried that didn't make the final model: averaging several DeBERTa-base runs (tiny gain, the large model was better),
and WavLM-large (no better on its own and worse inside the final ensemble, so I kept the smaller, faster base model).

After v6 I also tried several more ideas on Kaggle. Only scoring pieces of each clip (above), and later the smoother kernel, improved both CV and the
public score; these didn't:
- **Fine-tuning WavLM** itself (`kaggle_wavlm_finetune/`): it overfit on 769 clips, reaching 0.666 on its own versus
  0.536 for the frozen version, and nested CV gave it zero weight in the ensemble.
- **Averaging 3 DeBERTa-large runs** (`kaggle_deberta_large_seeds/`): ensemble CV 0.4963 → 0.4947, too small to matter.
  I submitted it as v7 to check: public 0.3522 vs v6's 0.3510, so v6 stayed.
- **Better transcripts** from Whisper large-v3 instead of turbo (`kaggle_asr_v3/`, `kaggle_deberta_large_v3/`): the grammar
  features got stronger on their own, but the full model didn't improve (best combination 0.4968), because WavLM already
  covers that signal and DeBERTa did worse on the noisier text.
- **"What Whisper corrected":** disagreement between Whisper and models without a language model (wav2vec2, NVIDIA Parakeet;
  `kaggle_ctc_disagreement/`, `kaggle_parakeet/`), plus **grammatical-complexity** and **error-type** features
  (`kaggle_features_v2/`). Several of these track the score strongly (Parakeet disagreement: Spearman −0.48, close to the
  grammar-correction rate's −0.52), but none improved the model (best 0.4956): they measure the same thing it already knows.
- **An LLM grammar judge** (Qwen2.5-7B-Instruct rating each transcript against SHL's rubric, zero-shot;
  `kaggle_llm_judge/`): the strongest single text feature I found (Spearman +0.60), but no gain in the ensemble, and nested
  CV gave it zero weight as a third model, since it overlaps with what DeBERTa already learned.
- **Whisper's encoder** as a second audio embedding (`kaggle_whisper_encoder/`): worse (ensemble 0.5155).
- **Stronger regularization** of the audio model (`kaggle_regularization/`): tuning the SVR's `gamma` and compressing WavLM
  with PCA shrink its train/CV gap, but CV error doesn't change (0.536), so the gap isn't costing accuracy.
- **A learned blend** instead of 50/50 (0.667 × audio + 0.420 × text − 0.332, fitted with nested CV): a real CV gain
  (0.4963 → 0.4884), but submitted as v8 it scored 0.3566 publicly against v6's 0.3510. It stretches predictions toward the
  training spread, and the test set seems more tightly bunched.
- **Scores as ordered categories** (one logistic model per score step): no real gain; its small improvement came only from
  the score-0 clips, and the test set has none.
- **Other piece settings** (`kaggle_audio_pieces_variants/`): 5 s pieces and the mean were close; 15 s pieces were worse.
- **The same trick for the text model** (`kaggle_deberta_windows/`): DeBERTa on ~15 s transcript windows was worse
  (0.616 vs 0.599 on its own). A 15 s window has only ~27 words, so many contain no mistake but still get the clip's score.
- **Speed augmentation** (`kaggle_audio_augment/`): training the audio model also on 0.9× / 1.1× speed copies of each clip
  improved CV on all 5 splits, but only by 0.002 (0.4822 → 0.4800), and as v11a it scored 0.3471 publicly vs v10's 0.3466.
- **Re-checking the 50/50 blend** after v10: nested CV picked 0.65 for audio (CV 0.4822 → 0.4765), but it scored 0.3604
  publicly. In CV each clip's text score comes from one DeBERTa model, while on the test set it's the average of five,
  which is more accurate, so CV underrates the text model. That's likely why v8's learned blend failed too.
- **DeBERTa trained on both transcripts** (Whisper turbo and large-v3; `kaggle_deberta_both_asr/`): text error 0.599 → 0.590,
  but most of the ensemble gain came from averaging two models, and on the test set the two agree at r = 0.993.
- **Fine-tuning an LLM** (Qwen2.5-3B with LoRA; `kaggle_llm_lora/`): worse than DeBERTa (0.682 on its own vs 0.599, even after a second run with longer, steadier training).
- **A second speech encoder**, w2v-BERT 2.0, next to WavLM in every piece (`kaggle_audio_w2vbert/`): ensemble CV 0.4822 →
  0.4762, better on all 5 splits, but as v12 it scored 0.3569 publicly. **Pseudo-labelling** the test clips: +0.001, too small.
- **Averaging the v10 and v12 audio models** (v13): CV 0.4822 → 0.4774, better on all 5 splits; public 0.3498.
- **Test-style training audio** (v14; `kaggle_audio_pauses/`): test speakers pause about half as often, so I re-cut every
  training clip with the test's pause pattern and trained on both versions. On pause-shortened validation clips it beat v10
  on all 5 splits (0.4885 vs 0.4959), but it scored 0.3563 publicly.

**Why CV stopped predicting the leaderboard.** A simple classifier tells training clips from test clips with AUC ≈ 0.83
(0.5 would mean no difference), so the test audio differs in ways cross-validation on training clips can't see. That's also
why every version scores ~0.35 publicly against ~0.49 in CV. After v10, all three changes that improved CV lost on the public
score, so I stopped there instead of tuning against a ~100-clip public set. One concrete difference is clip length: about
half the test clips are 45–55 s long, where v10 over-predicts by ~0.16 in CV (shorter training clips come from lower-scoring
speakers). Giving the model the clip's duration, or weighting training clips to the test's length mix, barely changed that
(best: 0.5153 → 0.5136 on length-weighted CV), so I left it documented rather than guessing a correction.

**The winner's curse.** v10 was picked *because* it had the best public score among about ten submissions, and with ~130
public clips the top scorer is usually a bit lucky. So v10's 0.3466 is probably optimistic, and every challenger was measured
against an inflated bar: five ideas that beat v10 on all 5 CV splits each lost publicly by a similar small margin. The
private 40% is a fresh sample, and it decides.

**The real problem: the CV was leaking speakers.** Many training speakers recorded several answers (usually 3, to different
questions, filed next to each other), mostly with the same score; no test speaker matches any training speaker. Clip-split
folds put a speaker's other answers in training, so a model could partly score a clip by recognising the voice. Grouping the
folds by speaker (`train.speaker_groups`: near-identical voices, or similar voices with neighbouring file numbers) raised every
audio model's CV error by ~0.08, left DeBERTa unchanged (0.599 → 0.596), and finally ranked models the way the leaderboard did:
the 50/50 blend is right (best weight 0.45–0.5) and a smoother kernel is better. That smoother model, **v15**, improved
speaker-grouped CV (0.5362 → 0.5339) *and* the public score (0.3466 → 0.3428). Honesty checks on Kaggle
(`kaggle_honesty_audit/`): with scores shuffled between speakers the pipeline does worse than predicting the mean (no leak),
and choosing the kernel width inside each training fold puts v15's real gain at ~0.001, so part of its public gain is luck.

**Final picks.** The public leaderboard uses 60% of the test set (~130 clips); the final ranking uses the other 40% (~86
clips), and Kaggle keeps the better of two chosen submissions. I picked **v15** (best public score, and the model `train.py`
and `predict.py` run) and **v16** (v15's audio model averaged with a smoother WavLM + w2v-BERT one: the best speaker-grouped CV,
0.5316, and 0.3443 publicly).

Details are in sections 10, 10b and 10c of the notebook.

## What's in this repo

| file / folder | what it is |
|---|---|
| `shl_grammar_scoring.ipynb` | the main notebook: explanation, plots, evaluation, final predictions, `predict.py` demo |
| `predict.py` | **audio file in → score out**, running the whole pipeline end to end |
| `features.py` | builds the features from transcripts (shared by training and `predict.py`) |
| `train.py` | the audio model (SVR on overlapping 10 s pieces), speaker-grouped cross-validation, and the final scoring rule (`combine_scores`) |
| `test_pipeline.py` | fast checks of the scoring rule, audio loading and input validation (no data needed): `python test_pipeline.py` |
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
   kaggle kernels push -p kaggle_audio_pieces_variants   # WavLM per 10 s piece -> audio_pieces_p10hop5.npz
   kaggle kernels push -p kaggle_deberta_large  # DeBERTa-large 5-fold predictions
   kaggle kernels push -p kaggle_deberta_final  # DeBERTa-large trained on all clips (for predict.py)
   ```
   Also push `kaggle_deberta` and `kaggle_deberta_seeds` (the DeBERTa-base runs): the notebook compares them with the large
   model in section 5. The other `kaggle_*` folders (`kaggle_audio_pieces` (v9), `kaggle_audio_large`, `kaggle_wavlm_finetune`, `kaggle_deberta_large_seeds`, `kaggle_asr_v3`,
   `kaggle_deberta_large_v3`, `kaggle_ctc_disagreement`, `kaggle_parakeet`, `kaggle_features_v2`, `kaggle_llm_judge`, `kaggle_whisper_encoder`, `kaggle_regularization`, `kaggle_deberta_windows`, `kaggle_audio_augment`, `kaggle_deberta_both_asr`, `kaggle_llm_lora`, `kaggle_audio_w2vbert`, `kaggle_audio_pauses`, `kaggle_deberta_speaker`, `kaggle_tune_speaker`, `kaggle_search_speaker`, `kaggle_honesty_audit`) are experiments that
   didn't make the final model (notebook section 10); they're optional.
3. **Download the outputs** with `kaggle kernels output <your-username>/<notebook-name> -p <folder>` into:
   `transcripts/` (unzip `transcripts.zip` there), `gec.json` (project root), `kaggle_out/audio/`, `kaggle_out/audio_pieces_variants/`, `kaggle_out/deberta/`,
   `kaggle_out/deberta_seeds/`, `kaggle_out/deberta_large/`, `kaggle_out/audio_large/`, and `models/deberta_final/`.
4. **Train the audio model and run the notebook:**
   ```bash
   .venv/bin/python test_pipeline.py
   .venv/bin/python train.py
   .venv/bin/jupyter nbconvert --to notebook --execute shl_grammar_scoring.ipynb
   ```
5. **Score any audio file:** `.venv/bin/python predict.py clip.wav` (needs Apple Silicon for local Whisper;
   it downloads about 3 GB of pretrained models the first time). To score many files, pass them all in one call
   (`predict.py a.wav b.wav ...`): each model loads once, scores every file, and is freed before the next one loads, so
   the first file takes about a minute (mostly loading) and each extra file about 6 s on my 8 GB M1.

## What I'd improve with more time

- **Whisper still cleans up some speech.** I can't measure exactly how much without human-written transcripts.
- **Very few clips score 1–2**, so the model is least accurate there and tends to predict towards the middle.
- **More data before bigger audio models.** Fine-tuning WavLM overfit here; with more labelled audio it would be worth revisiting.
- **The score-0 rule** catches all 37 here, but it's based on very few examples.
- **Training and test audio differ.** With more audio like the test set, even unscored, I'd first find out what differs
  (microphones, prompts, speakers) and build validation around it, before trying more models.
- **For production** I'd start from the audio model alone (CV RMSE 0.52 without DeBERTa) and use lighter models;
  the notebook's last section has the details.

## Note on tools

SHL said AI tools were allowed, and I used an AI assistant to help write code and debug.
I made sure I understand every step, and I'm happy to walk through any of it.

Licensed under the MIT License (see `LICENSE`).
