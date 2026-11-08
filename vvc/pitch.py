"""Pitch tracking for VVC, built on librosa.pyin."""
import time
from dataclasses import dataclass

import numpy as np

# Covers low chest voice up to high falsetto.
FMIN_HZ = 65.0    # C2
FMAX_HZ = 1047.0  # C6
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class PitchTrack:
    times: np.ndarray        # seconds, one per frame
    f0: np.ndarray           # Hz, NaN where unvoiced
    voiced_prob: np.ndarray  # pyin's voicing probability per frame
    sr: int
    duration_s: float
    elapsed_s: float         # wall-clock time spent in pyin


def load_audio(path, sr=22050):
    """Load .wav or .mp3 as mono float audio. mp3 needs ffmpeg installed."""
    import librosa
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def track_pitch(y, sr, frame_length=2048, hop_length=512):
    """Run pyin and time it."""
    import librosa
    start = time.perf_counter()
    f0, _, voiced_prob = librosa.pyin(
        y, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    elapsed = time.perf_counter() - start
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return PitchTrack(times, f0, voiced_prob, sr, len(y) / sr, elapsed)


def hz_to_midi(hz):
    return 69 + 12 * np.log2(np.asarray(hz, dtype=float) / 440.0)


def hz_to_note(hz):
    """440 -> 'A4'. Rounds to nearest semitone, also returns cents offset."""
    if hz is None or not np.isfinite(hz) or hz <= 0:
        return None, None
    midi = float(hz_to_midi(hz))
    nearest = int(round(midi))
    cents = round((midi - nearest) * 100)
    name = f"{NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}"
    return name, cents


def summarise(track):
    voiced = track.f0[np.isfinite(track.f0)]
    if voiced.size == 0:
        return {"voiced_fraction": 0.0, "median_hz": None, "note": None}
    median = float(np.median(voiced))
    note, cents = hz_to_note(median)
    return {
        "duration_s": round(track.duration_s, 3),
        "frames": int(track.f0.size),
        "voiced_fraction": round(voiced.size / track.f0.size, 3),
        "median_hz": round(median, 2),
        "note": note,
        "cents_off": cents,
        "pyin_time_s": round(track.elapsed_s, 3),
    }
