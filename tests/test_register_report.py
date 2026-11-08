import numpy as np

from vvc.features import register_features
from vvc.register import classify, fit_model
from vvc.report import build_report, held_notes, vibrato

SR = 22050


def tone(f0, amps, noise=0.0, n=2048, seed=0):
    t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * f0 * (k + 1) * t) for k, a in enumerate(amps))
    return y + noise * np.random.default_rng(seed).standard_normal(n)


def test_features_order_rich_vs_light():
    """Rich harmonics (chest-like) vs near-sine + noise (falsetto-like)."""
    rich = register_features(tone(220, [1 / k for k in range(1, 20)]), SR, 220)
    light = register_features(tone(220, [1, 0.08, 0.02], noise=0.05), SR, 220)
    assert light[0] > rich[0] + 10   # H1-H2 higher when light
    assert light[1] > rich[1]        # more energy below 1 kHz
    assert light[2] < rich[2]        # breathier -> lower HNR


def test_unvoiced_has_no_features():
    assert register_features(np.zeros(2048), SR, float("nan")) is None


def test_fit_and_classify():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal([0, 6, 20], 1, (30, 3)), rng.normal([14, 20, 10], 1, (30, 3))])
    y = ["chest"] * 30 + ["falsetto"] * 30
    m = fit_model(X, y)
    assert classify([1, 7, 19], m)[0] == "chest"
    assert classify([13, 19, 11], m)[0] == "falsetto"


def test_vibrato_detected():
    t = np.arange(0, 2, 0.05)
    rate, extent = vibrato(t, 40 * np.sin(2 * np.pi * 5.5 * t))
    assert abs(rate - 5.5) < 0.6 and 25 < extent < 50
    assert vibrato(t, np.zeros_like(t) + np.linspace(0, 3, len(t))) is None


def test_report_on_synthetic_session():
    t = np.arange(0, 12, 0.05)
    midi = np.concatenate([np.full(80, 57.0), np.linspace(57, 72, 80), np.full(80, 72.0)])
    midi += 0.3 * np.sin(2 * np.pi * 5.5 * t)       # vibrato +/-30 cents
    hz = 440 * 2 ** ((midi - 69) / 12)
    labels = ["chest"] * 130 + ["head"] * 110
    text, fig = build_report(t, hz, labels, np.full(len(t), 18.0), calibrated=True)
    assert "A3 to C5" in text
    assert "held notes" in text and "Vibrato" in text
    assert fig is not None


def test_held_notes_splits_on_note_change():
    t = np.arange(0, 2, 0.05)
    m = np.where(t < 1, 60.0, 62.0)
    assert len(held_notes(t, m)) == 2
