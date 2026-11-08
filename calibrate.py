"""Teach the register classifier your voice, and measure how well it does.

Name clips by register first: chest_A3_1.wav, head_A4_2.wav, falsetto_A4_1.wav,
mix_E4_1.wav ... (at least 2 clips per register for an honest test).

Usage: python calibrate.py clips/*.wav

Accuracy is leave-one-clip-out: each clip is classified by a model trained on
all the OTHER clips, so it never grades itself.
"""
import os
import sys
from collections import Counter

import numpy as np

from vvc.analysis import frame_features
from vvc.pitch import load_audio, track_pitch
from vvc.register import REGISTERS, classify, fit_model, save_model, MODEL_PATH


def label_from_name(path):
    prefix = os.path.basename(path).split("_")[0].lower()
    return prefix if prefix in REGISTERS else None


def main(paths):
    clips = []
    for p in paths:
        lab = label_from_name(p)
        if lab is None:
            print(f"skip {p}: name must start with one of {REGISTERS}")
            continue
        y, sr = load_audio(p)
        f = frame_features(y, sr, track_pitch(y, sr))
        f = f[~np.isnan(f).any(axis=1)]
        if len(f) < 5:
            print(f"skip {p}: almost no voiced frames")
            continue
        clips.append((os.path.basename(p), lab, f))
        print(f"{os.path.basename(p):<28} {lab:<9} {len(f)} voiced frames")

    if len({c[1] for c in clips}) < 2:
        sys.exit("Need clips from at least 2 registers.")

    print("\nLeave-one-clip-out test")
    correct = 0
    frame_hits = frame_total = 0
    for i, (name, lab, f) in enumerate(clips):
        others = [c for j, c in enumerate(clips) if j != i]
        if lab not in {c[1] for c in others}:
            print(f"  {name:<28} skipped (only clip of {lab})")
            continue
        model = fit_model(np.vstack([c[2] for c in others]),
                          np.concatenate([[c[1]] * len(c[2]) for c in others]))
        preds = [classify(row, model)[0] for row in f]
        vote = Counter(preds).most_common(1)[0][0]
        correct += vote == lab
        frame_hits += sum(p == lab for p in preds)
        frame_total += len(preds)
        print(f"  {name:<28} true {lab:<9} predicted {vote:<9}"
              f"{'OK' if vote == lab else 'WRONG'}")

    tested = sum(1 for i, c in enumerate(clips)
                 if c[1] in {d[1] for j, d in enumerate(clips) if j != i})
    if tested:
        print(f"\nClip accuracy:  {correct}/{tested} ({100 * correct / tested:.0f}%)")
        print(f"Frame accuracy: {frame_hits}/{frame_total} "
              f"({100 * frame_hits / frame_total:.0f}%)")

    final = fit_model(np.vstack([c[2] for c in clips]),
                      np.concatenate([[c[1]] * len(c[2]) for c in clips]))
    final["n_clips"] = len(clips)
    save_model(final)
    print(f"\nSaved {MODEL_PATH}. The live view and reports now use it.")
    for lab, c in zip(final["labels"], final["centroids"]):
        print(f"  {lab:<9} H1-H2 {c[0]:6.1f} dB   low/high {c[1]:6.1f} dB   HNR {c[2]:6.1f} dB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
