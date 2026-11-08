import numpy as np
import pytest

from vvc.pitch import hz_to_note


def test_note_names():
    assert hz_to_note(440.0) == ("A4", 0)
    assert hz_to_note(261.63)[0] == "C4"
    assert hz_to_note(110.0) == ("A2", 0)
    assert hz_to_note(float("nan")) == (None, None)


def test_live_estimate():
    """Live estimator: 330 Hz tone -> E4, silence -> NaN."""
    pytest.importorskip("librosa")
    from live_pitch import estimate_pitch, WINDOW, SR
    t = np.arange(WINDOW) / SR
    hz = estimate_pitch(0.5 * np.sin(2 * np.pi * 330.0 * t).astype(np.float32))
    assert hz_to_note(hz)[0] == "E4"
    assert np.isnan(estimate_pitch(np.zeros(WINDOW, dtype=np.float32)))


def test_pyin_on_synthetic_tone():
    """A pure 220 Hz sine should come back as A3 within a few cents."""
    pytest.importorskip("librosa")
    from vvc.pitch import track_pitch, summarise
    sr = 22050
    t = np.arange(int(sr * 1.5)) / sr
    y = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    s = summarise(track_pitch(y, sr))
    assert s["note"] == "A3"
    assert abs(s["median_hz"] - 220.0) < 2.0
    assert s["voiced_fraction"] > 0.8
