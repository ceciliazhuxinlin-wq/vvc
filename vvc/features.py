"""Spectral features that differ between vocal registers.

All three are computed from one short block of audio plus its pitch (f0):

  h1h2_db   Strength of the 1st harmonic minus the 2nd. Thin, light vocal
            fold contact (head, falsetto) puts more energy in H1, so this
            goes up. Firm contact (chest) brings it down.
  alpha_db  Energy below 1 kHz minus energy from 1 to 5 kHz. Chest voice has
            stronger upper harmonics, so this is lower in chest and higher
            in falsetto (steeper spectral tilt).
  hnr_db    Harmonic-to-noise estimate: energy at the harmonics vs between
            them. Breathy sound (often falsetto) lowers it.

These depend on vowel, loudness and microphone too, which is why the
classifier is calibrated on each singer's own labelled clips.
"""
import numpy as np

FEATURE_NAMES = ["h1h2_db", "alpha_db", "hnr_db"]
NFFT = 8192


def _db(power):
    return 10 * np.log10(np.maximum(power, 1e-12))


def register_features(block, sr, f0):
    """Return [h1h2_db, alpha_db, hnr_db] for one block, or None if unvoiced."""
    if f0 is None or not np.isfinite(f0) or f0 <= 0:
        return None
    x = np.asarray(block, dtype=float)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), NFFT)) ** 2
    freqs = np.fft.rfftfreq(NFFT, 1 / sr)

    def peak(f):
        band = np.abs(freqs - f) <= 0.2 * f0
        return spec[band].max() if band.any() else 1e-12

    h1h2 = _db(peak(f0)) - _db(peak(2 * f0))

    low = spec[(freqs >= 50) & (freqs < 1000)].sum()
    high = spec[(freqs >= 1000) & (freqs < 5000)].sum()
    alpha = _db(low) - _db(high)

    band = (freqs >= 50) & (freqs < 5000)
    nearest_harmonic = np.maximum(np.round(freqs / f0), 1) * f0
    half_width = max(12.0, min(0.25 * f0, 30.0))
    harmonic = band & (np.abs(freqs - nearest_harmonic) <= half_width)
    hnr = _db(spec[harmonic].sum()) - _db(spec[band & ~harmonic].sum())

    return np.array([h1h2, alpha, hnr])
