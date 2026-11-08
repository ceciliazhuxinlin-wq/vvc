"""Experimental register classifier: nearest centroid on 3 spectral features.

Without calibration it uses rough default centroids (guesses, not measured).
Run calibrate.py on your own labelled clips to replace them.
"""
import json
import os

import numpy as np

REGISTERS = ["chest", "mix", "head", "falsetto"]
COLORS = {"chest": "#e8743b", "mix": "#c04abc", "head": "#3b7fe8",
          "falsetto": "#5cc8c8", None: "#9a9a9a"}
MODEL_PATH = "register_model.json"

DEFAULT_MODEL = {
    "labels": REGISTERS,
    # [h1h2_db, alpha_db, hnr_db] -- rough starting guesses only
    "centroids": [[0, 6, 20], [4, 10, 18], [9, 15, 15], [14, 20, 10]],
    "scale": [4, 5, 5],
    "calibrated": False,
}


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return DEFAULT_MODEL


def save_model(model, path=MODEL_PATH):
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def classify(feat, model):
    """Return (label, confidence 0-1). confidence is a softmax over distances."""
    if feat is None:
        return None, 0.0
    c = np.asarray(model["centroids"], dtype=float)
    s = np.asarray(model["scale"], dtype=float)
    d2 = (((np.asarray(feat) - c) / s) ** 2).sum(axis=1)
    p = np.exp(-0.5 * (d2 - d2.min()))
    p /= p.sum()
    i = int(np.argmin(d2))
    return model["labels"][i], float(p[i])


def fit_model(X, y):
    """X: (n, 3) features, y: labels. Returns a calibrated model dict."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    labels = [r for r in REGISTERS if r in set(y)]
    centroids = [X[y == lab].mean(axis=0).tolist() for lab in labels]
    # pooled within-register spread, so one loud register doesn't dominate
    resid = np.concatenate([X[y == lab] - X[y == lab].mean(axis=0) for lab in labels])
    scale = np.maximum(resid.std(axis=0), 1e-3).tolist()
    return {"labels": labels, "centroids": centroids, "scale": scale,
            "calibrated": True}
