"""Offline analysis of a whole recording: pitch, features and register per frame."""
import numpy as np

from vvc.features import register_features
from vvc.pitch import track_pitch
from vvc.register import classify

FRAME = 2048


def frame_features(y, sr, track, hop=512):
    """Spectral features for every voiced pyin frame (NaN rows when unvoiced)."""
    padded = np.pad(y, FRAME // 2)
    feats = np.full((len(track.f0), 3), np.nan)
    for i, hz in enumerate(track.f0):
        block = padded[i * hop: i * hop + FRAME]
        if len(block) == FRAME:
            f = register_features(block, sr, hz)
            if f is not None:
                feats[i] = f
    return feats


def analyse_audio(y, sr, model):
    """Return dict of per-frame arrays: times, hz, labels, conf, hnr."""
    track = track_pitch(y, sr)
    feats = frame_features(y, sr, track)
    labels, conf = [], []
    for f in feats:
        lab, c = classify(None if np.isnan(f).any() else f, model)
        labels.append(lab)
        conf.append(c)
    return {"times": track.times, "hz": track.f0, "labels": labels,
            "conf": np.array(conf), "hnr": feats[:, 2], "feats": feats}
