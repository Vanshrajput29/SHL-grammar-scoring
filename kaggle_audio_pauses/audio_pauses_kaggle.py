"""Experiment (Kaggle GPU): make the training audio sound like the test audio, then retrain the audio model.

Test speakers pause about half as often as training speakers (silence 16% vs 26% of a clip, at every pause length), speak
~10% faster, and are recorded ~2 dB quieter, with the same amount of actual speech (notebook section 10b). Pauses barely
relate to the grammar score in training, but WavLM hears them, so a fluent-sounding test speaker may look better than they are.
Here every training clip gets the test's pause pattern: each pause of at least 0.2 s keeps 45% of its length (cut from the
middle), leading silence is cut to 0.1 s, trailing silence removed, and the volume lowered by 20% (keep=0.45 matched the
test's silence share, pause count and duration on 150 sampled clips). Speech is never touched.
Then the same WavLM pieces + clip embedding as v10, and a 5-split comparison on test-style validation clips:
  v10 (trained on original audio) vs pause-matched training vs both, always validated on pause-matched clips.
Outputs audio_pieces_pauses.npz ('train_<stem>' pieces, 'clip:train_<stem>', plus 'duration'/'rms' arrays) and the results.
"""
import wave
from pathlib import Path

import numpy as np

SR, FRAME, KEEP, GAIN = 16000, 320, 0.45, 0.8


def silent_frames(x):
    fr = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    return db < np.percentile(db, 95) - 35          # 20 ms frames 35 dB below the loud parts


def shorten_pauses(x, keep=KEEP, lead_s=0.1, min_pause_s=0.2):
    sil = silent_frames(x)
    speech = np.where(~sil)[0]
    if len(speech) < 2:
        return x
    first, last = speech[0], speech[-1]
    keep_mask = np.ones(len(sil), bool)
    keep_mask[: max(first - int(lead_s / .02), 0)] = False
    keep_mask[last + 1:] = False
    i = first
    while i <= last:
        if not sil[i]:
            i += 1
            continue
        j = i
        while j <= last and sil[j]:
            j += 1
        n = j - i
        if n * .02 >= min_pause_s:
            drop = n - max(int(round(n * keep)), 1)
            keep_mask[i + (n - drop) // 2: i + (n - drop) // 2 + drop] = False
        i = j
    return x[: len(sil) * FRAME].reshape(-1, FRAME)[keep_mask].ravel()


def split_pieces(x, piece=10 * SR, hop=5 * SR):     # identical rule to transcribe.split_pieces
    if len(x) <= piece:
        return [x]
    segs = [(s, s + piece) for s in range(0, len(x) - piece + 1, hop)]
    if len(x) - segs[-1][1] >= piece // 2:
        segs.append((len(x) - piece, len(x)))
    else:
        segs[-1] = (segs[-1][0], len(x))
    return [x[a:b] for a, b in segs]


if __name__ == "__main__":
    import pandas as pd
    import torch
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from transformers import AutoFeatureExtractor, WavLMModel

    DATA = next(Path("/kaggle/input").rglob("train.csv")).parent
    Z = np.load(next(Path("/kaggle/input").rglob("eval_inputs.npz")))   # v10's training inputs, rows in train.csv order

    def load_wav(path):
        with wave.open(str(path)) as w:
            assert w.getframerate() == SR and w.getsampwidth() == 2, path
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            return x.reshape(-1, w.getnchannels()).mean(axis=1)

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda().eval()

    def hidden(seg):
        with torch.no_grad():
            return torch.stack(model(fe(seg, sampling_rate=SR, return_tensors="pt").input_values.cuda(),
                                     output_hidden_states=True).hidden_states)[:, 0]          # [13, T, 768]

    def clip_embedding(x):                           # same as kaggle_audio/: 20 s chunks, time mean, then layer mean
        sums, n = None, 0
        for s in range(0, len(x), 20 * SR):
            seg = x[s:s + 20 * SR]
            if len(seg) < SR:
                continue
            h = hidden(seg); ls = h.sum(1).cpu().numpy()
            sums, n = (ls if sums is None else sums + ls), n + h.shape[1]
        return (sums / n).mean(0)

    train = pd.read_csv(DATA / "train.csv")
    out, dur, rms, before, after = {}, [], [], [], []
    for i, f in enumerate(train.filename):
        x = load_wav(DATA / "train" / f)
        xs = shorten_pauses(x) * GAIN
        before.append(silent_frames(x).mean()); after.append(silent_frames(xs).mean())
        key = f"train_{f[:-4]}"
        out[key] = np.stack([hidden(p).mean(1).mean(0).cpu().numpy() for p in split_pieces(xs)]).astype(np.float32)
        out["clip:" + key] = clip_embedding(xs).astype(np.float32)
        dur.append(len(xs) / SR); rms.append(float(np.sqrt((xs ** 2).mean())))
        if i % 100 == 0:
            print(i, f, f"{len(x) / SR:.1f}s -> {len(xs) / SR:.1f}s", flush=True)
    out["duration"], out["rms"] = np.array(dur), np.array(rms)
    np.savez_compressed("/kaggle/working/audio_pieces_pauses.npz", **out)
    y = Z["y"]; pos = y > 0
    print(f"silence share {np.mean(before):.3f} -> {np.mean(after):.3f} (test: ~0.16); mean duration "
          f"{np.mean([len(Z[f'p_{i}']) for i in range(len(y))]):.1f} pieces -> {np.mean([len(out[f'train_{f[:-4]}']) for f in train.filename]):.1f} pieces", flush=True)

    # ---- evaluation: hand-made features of the pause-matched clips (words/sec and rms follow the new audio) ----
    H = Z["H"]; Hs = H.copy()
    Hs[:, 1] = H[:, 0] / out["duration"]               # words_per_sec
    Hs[:, 10] = out["rms"]                              # rms
    P, A = [Z[f"p_{i}"] for i in range(len(y))], Z["A"]
    Ps = [out[f"train_{f[:-4]}"] for f in train.filename]
    As = np.stack([out[f"clip:train_{f[:-4]}"] for f in train.filename])
    deb = Z["t_deberta_large"]
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

    def rows(Pp, Aa, Hh, clips):
        return np.vstack([np.concatenate([p, Aa[i], Hh[i]]) for i in clips for p in Pp[i]])

    def fit(sets, clips):                               # same model as train.ScaledRbfSVR(C=1, epsilon=0.05)
        X = np.vstack([rows(Pp, Aa, Hh, clips) for Pp, Aa, Hh in sets])
        yy = np.concatenate([np.repeat(y[i], len(Pp[i])) for Pp, _, _ in sets for i in clips])
        sc = StandardScaler().fit(X); Zx = sc.transform(X); g = 1 / (Zx.shape[1] * Zx.var())
        svr = SVR(kernel="precomputed", C=1, epsilon=0.05).fit(rbf_kernel(Zx, gamma=g), yy)
        return sc, g, Zx[svr.support_], svr.dual_coef_[0], svr.intercept_[0]

    def predict(m, Pp, Aa, Hh, clips):
        sc, g, sv, dc, b = m
        p = rbf_kernel(sc.transform(rows(Pp, Aa, Hh, clips)), sv, gamma=g) @ dc + b
        return np.clip([np.median(s) for s in np.split(p, np.cumsum([len(Pp[i]) for i in clips])[:-1])], 0, 5)

    def combine(a, t):                                  # same as train.combine_scores
        return np.where((a < 1) | np.isnan(t), a, 0.5 * a + 0.5 * np.nan_to_num(t))

    orig, paus = (P, A, H), (Ps, As, Hs)
    variants = {"v10: train original, validate original": ([orig], orig),
                "v10: train original, validate pause-matched": ([orig], paus),
                "train pause-matched, validate pause-matched": ([paus], paus),
                "train both, validate pause-matched": ([orig, paus], paus)}
    res = {k: [] for k in variants}
    for seed in [42, 1, 2, 3, 4]:
        for k, (sets, val) in variants.items():
            oof = np.zeros(len(y))
            for a, b in KFold(5, shuffle=True, random_state=seed).split(y):
                oof[b] = predict(fit(sets, a), *val, b)
            e = combine(oof, deb)
            res[k].append([rmse(y, e), rmse(y[pos], e[pos]), float(np.mean((e - y)[pos])), int((oof[~pos] < 1).sum())])
        print("seed", seed, "done", flush=True)
    print(f"\n{'variant':<46} {'CV all':>7} {'CV 1-5':>7} {'bias 1-5':>9} {'score-0 caught':>15}  per-split CV 1-5")
    for k, v in res.items():
        v = np.array(v); m = v.mean(0)
        print(f"{k:<46} {m[0]:7.4f} {m[1]:7.4f} {m[2]:+9.3f} {m[3]:15.0f}  {np.round(v[:, 1], 4)}")
