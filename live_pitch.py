"""Live VVC: pitch graph, current note + register, and where it's felt.

Usage: python live_pitch.py
Sing into the mic. Close the window to stop; a report is saved in reports/.

Pitch uses librosa.yin on the latest ~93 ms each update (the fast sibling of
the pyin tracker in pitch.py, so numbers can differ slightly).
"""
import collections
import os
import time
from datetime import datetime

import numpy as np

from vvc.features import register_features
from vvc.pitch import FMAX_HZ, FMIN_HZ, hz_to_midi
from vvc.register import classify, load_model

SR = 22050
WINDOW = 2048          # samples analysed per update (~93 ms)
UPDATE_MS = 50
HISTORY_S = 10
SILENCE_RMS = 0.01
SMOOTH = 5             # majority vote over the last N register labels


def estimate_pitch(block, sr=SR):
    """Return pitch in Hz for one block of audio, or NaN if too quiet."""
    import librosa
    if np.sqrt(np.mean(block ** 2)) < SILENCE_RMS:
        return float("nan")
    f0 = librosa.yin(block, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr,
                     frame_length=len(block), center=False)
    return float(np.median(f0))


def save_report(session, calibrated):
    from vvc.report import build_report
    if len(session["t"]) < 10:
        return
    text, fig = build_report(session["t"], session["hz"], session["label"],
                             session["hnr"], calibrated, title="VVC live session")
    out = os.path.join("reports", datetime.now().strftime("live_%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(text + "\n")
    if fig:
        fig.savefig(os.path.join(out, "report.png"), dpi=130)
    np.savez(os.path.join(out, "session.npz"),
             **{k: np.asarray(v, dtype=object if k == "label" else float)
                for k, v in session.items()})
    print("\n" + text)
    print(f"\nSaved to {out}/")


def main():
    import sounddevice as sd
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from vvc.ui import LiveView

    model = load_model()
    audio = np.zeros(WINDOW, dtype=np.float32)
    n_points = int(HISTORY_S * 1000 / UPDATE_MS)
    view_t = collections.deque(maxlen=n_points)
    view_m = collections.deque(maxlen=n_points)
    view_l = collections.deque(maxlen=n_points)
    recent = collections.deque(maxlen=SMOOTH)
    proc = collections.deque(maxlen=100)
    session = {"t": [], "hz": [], "label": [], "conf": [], "hnr": []}
    t0 = time.perf_counter()

    def on_audio(indata, frames, time_info, status):
        nonlocal audio
        audio = np.concatenate([audio, indata[:, 0]])[-WINDOW:]

    fig = plt.figure(figsize=(13, 7))
    view = LiveView(fig, model.get("calibrated", False))

    def update(_):
        start = time.perf_counter()
        block = audio.copy()
        hz = estimate_pitch(block)
        feat = register_features(block, SR, hz)
        raw, conf = classify(feat, model)
        proc.append((time.perf_counter() - start) * 1000)

        recent.append(raw)
        votes = [r for r in recent if r]
        label = max(set(votes), key=votes.count) if raw and votes else None

        now = time.perf_counter() - t0
        view_t.append(now)
        view_m.append(float(hz_to_midi(hz)) if np.isfinite(hz) else np.nan)
        view_l.append(label)
        for k, v in zip(session, (now, hz, raw, conf,
                                  feat[2] if feat is not None else np.nan)):
            session[k].append(v)
        view.update(list(view_t), list(view_m), list(view_l), hz, label, conf,
                    float(np.mean(proc)), HISTORY_S)

    with sd.InputStream(samplerate=SR, channels=1, blocksize=512, callback=on_audio):
        _anim = FuncAnimation(fig, update, interval=UPDATE_MS,
                              blit=False, cache_frame_data=False)
        plt.show()
    save_report(session, model.get("calibrated", False))


if __name__ == "__main__":
    main()
