# VVC: Visual Voice Canvas

A Python tool that listens to singing and shows, live, **which note** you are on
and **how** you are singing it (chest, mix, head or falsetto), then writes a
report on range, voice type, pitch accuracy, vibrato and clarity.

> Work in progress. Results will be added here once they are measured.

## The question

Pitch tracking tells you *which* note someone sings. But the same A4 can be sung
in chest voice or falsetto. Can a plain microphone tell *how* a note is sung?

## How it works

1. **Pitch**: `librosa.pyin` (recordings) or `librosa.yin` (live) finds the
   fundamental frequency, converted to a note name and cents.
2. **Register features**: from each ~93 ms slice:
   - **H1-H2**: 1st harmonic minus 2nd (higher with light fold contact)
   - **Low/high energy balance**: below vs above 1 kHz (spectral tilt)
   - **Harmonic-to-noise estimate**: clean vs breathy
3. **Register label (experimental)**: nearest-centroid match to each register's
   average features. Defaults are rough guesses; `calibrate.py` learns them from
   the singer's own labelled clips and reports leave-one-clip-out accuracy.
4. **Report**: range (2nd-98th percentile), closest voice type using range
   *without falsetto* plus tessitura, held-note pitch accuracy in cents,
   vibrato rate/extent, harmonic-to-noise.

The on-screen "where it's felt" figure is singer imagery, not acoustics: all
sound is produced at the vocal folds.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q

python live_pitch.py                 # live window; report saved on close
python analyse_file.py clip.wav      # report for a recording
python calibrate.py clips/*.wav      # clips named chest_A3_1.wav, head_A4_1.wav ...
```

`.mp3` input needs ffmpeg.

## Limitations

- Register features also depend on vowel, loudness, microphone and room, so the
  classifier is only meaningful after calibration, and only for that singer.
- Mix voice is a blend, not a clean category, and is the least reliable label.
- Voice type from pitch alone is a rough match (e.g. mezzo vs tenor overlap).
- Pitch accuracy is measured against A440 equal temperament, not the intended note.
