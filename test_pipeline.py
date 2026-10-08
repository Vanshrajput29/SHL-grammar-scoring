"""Fast checks for the parts of the pipeline that make decisions. No competition data or large models needed.

    python test_pipeline.py        (also works with pytest)
"""
import tempfile
import wave
from pathlib import Path

import numpy as np

import features as F
import train
import transcribe as T


def write_wav(path, x, sr, channels=1, sampwidth=2):
    dtype = {2: np.int16, 4: np.int32}[sampwidth]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels); w.setsampwidth(sampwidth); w.setframerate(sr)
        w.writeframes((x * (2 ** (8 * sampwidth - 1) - 1)).astype(dtype).tobytes())


def test_combine_scores_gate_and_blend():
    # Below the cutoff the clip is unintelligible: keep the audio score, ignore the text model.
    assert train.combine_scores(0.4, 4.0) == 0.4
    assert train.combine_scores(0.4, np.nan) == 0.4
    # At or above the cutoff: 50/50 average of the two models.
    assert train.combine_scores(3.0, 4.0) == 3.5
    assert train.combine_scores(train.UNINTELLIGIBLE_BELOW, 2.0) == 1.5
    # No text prediction available -> fall back to the audio score.
    assert train.combine_scores(3.0, np.nan) == 3.0
    # Works element-wise on arrays (how the notebook uses it) and honours the weight.
    out = train.combine_scores(np.array([0.2, 3.0, 4.0]), np.array([np.nan, 5.0, 2.0]), w=0.75)
    assert np.allclose(out, [0.2, 3.5, 3.5]), out


def test_load_audio_resamples_and_downmixes():
    with tempfile.TemporaryDirectory() as d:
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(44100 * 2) / 44100)
        write_wav(Path(d) / "a.wav", np.repeat(tone, 2), 44100, channels=2)   # 2 s, 44.1 kHz stereo
        x = T.load_audio(Path(d) / "a.wav")
    assert x.dtype == np.float32 and len(x) == 2 * T.SR, (x.dtype, len(x))
    assert abs(np.abs(x).max() - 0.5) < 0.01


def test_load_audio_rejects_non_16bit():
    with tempfile.TemporaryDirectory() as d:
        write_wav(Path(d) / "b.wav", np.zeros(1000), 16000, sampwidth=4)
        try:
            T.load_audio(Path(d) / "b.wav")
        except ValueError as e:
            assert "16-bit" in str(e)
        else:
            raise AssertionError("32-bit WAV was accepted")


def test_predict_rejects_clips_under_one_second():
    import predict  # imports only; the check runs before any model is loaded
    with tempfile.TemporaryDirectory() as d:
        write_wav(Path(d) / "c.wav", np.zeros(T.SR // 2), T.SR)
        try:
            predict.score(Path(d) / "c.wav")
        except ValueError as e:
            assert "at least 1 s" in str(e)
        else:
            raise AssertionError("0.5 s clip was accepted")


def test_split_pieces_overlap_and_cover_all_audio():
    cut = lambda sec: T.split_pieces(np.arange(int(sec * T.SR)))
    lengths = lambda sec: [len(p) / T.SR for p in cut(sec)]
    assert lengths(6.5) == [6.5]                     # 10 s or less: one piece
    assert lengths(23) == [10, 10, 13]               # starts at 0, 5, 10 s; the 3 s left over stretches the last piece
    assert lengths(45) == [10] * 8                   # starts every 5 s: 0, 5, ..., 35 s
    assert len(cut(61)) == 11 and lengths(61)[-1] == 11
    for sec in (6.5, 23, 45, 52.3, 61):              # every sample is covered, nothing is dropped
        assert np.unique(np.concatenate(cut(sec))).size == int(sec * T.SR)


def test_gec_features_count_word_edits_only():
    g = F.gec_features([("I buyed some vegetables.", "I bought some vegetables."), ("It is good.", "It is good!")])
    assert g == {"gec_edit_rate": 1 / 7, "gec_changed_frac": 0.5}, g


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
