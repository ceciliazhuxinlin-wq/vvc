"""Usage: python track_pitch.py recording.wav [--csv out.csv]"""
import argparse
import csv

from vvc.pitch import load_audio, track_pitch, summarise, hz_to_note


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--csv", help="save frame-by-frame pitch to this file")
    args = p.parse_args()

    y, sr = load_audio(args.audio)
    track = track_pitch(y, sr)
    for k, v in summarise(track).items():
        print(f"{k:>16}: {v}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "f0_hz", "note", "voiced_prob"])
            for t, hz, vp in zip(track.times, track.f0, track.voiced_prob):
                w.writerow([f"{t:.3f}", "" if hz != hz else f"{hz:.2f}",
                            hz_to_note(hz)[0] or "", f"{vp:.3f}"])
        print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
