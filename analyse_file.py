"""Full report for a recording.

Usage: python analyse_file.py recording.wav
Saves reports/<name>/report.md and report.png.
"""
import os
import sys

from vvc.analysis import analyse_audio
from vvc.pitch import load_audio
from vvc.register import load_model
from vvc.report import build_report


def main(path):
    model = load_model()
    y, sr = load_audio(path)
    r = analyse_audio(y, sr, model)
    name = os.path.splitext(os.path.basename(path))[0]
    text, fig = build_report(r["times"], r["hz"], r["labels"], r["hnr"],
                             model.get("calibrated", False),
                             title=f"VVC report: {name}")
    out = os.path.join("reports", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(text + "\n")
    if fig:
        fig.savefig(os.path.join(out, "report.png"), dpi=130)
    print(text)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
