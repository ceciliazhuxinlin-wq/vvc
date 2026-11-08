"""Session report: range, voice type, register use, pitch accuracy, vibrato, clarity."""
import numpy as np
from matplotlib.figure import Figure

from vvc.pitch import NOTE_NAMES, hz_to_midi
from vvc.register import COLORS, REGISTERS

# Commonly cited approximate classical/choral ranges (MIDI note numbers).
# Real classification also uses timbre, comfort and where the voice shifts,
# so this is a rough match only.
VOICE_TYPES = {
    "Soprano": (60, 84),        # C4-C6
    "Mezzo-soprano": (57, 81),  # A3-A5
    "Contralto": (53, 77),      # F3-F5
    "Tenor": (48, 72),          # C3-C5
    "Baritone": (45, 69),       # A2-A4
    "Bass": (40, 64),           # E2-E4
}

HELD_MIN_S = 0.25     # shorter than this is a slide, not a held note
VIBRATO_MIN_S = 0.8


def midi_name(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def held_notes(times, midi):
    """Split into runs that stay on one semitone; keep runs >= HELD_MIN_S."""
    segs, start = [], None
    for i in range(len(midi)):
        ok = np.isfinite(midi[i])
        same = (start is not None and ok
                and round(midi[i]) == round(midi[start])
                and times[i] - times[i - 1] <= 0.15)
        if not same:
            if start is not None and times[i - 1] - times[start] >= HELD_MIN_S:
                segs.append((start, i))
            start = i if ok else None
    if start is not None and times[-1] - times[start] >= HELD_MIN_S:
        segs.append((start, len(midi)))
    return segs


def vibrato(times, cents):
    """Return (rate_hz, extent_cents) if a 4-8 Hz wobble dominates, else None."""
    dt = np.median(np.diff(times))
    grid = np.arange(times[0], times[-1], dt)
    c = np.interp(grid, times, cents)
    c = c - np.polyval(np.polyfit(grid, c, 1), grid)
    power = np.abs(np.fft.rfft(c * np.hanning(len(c)))) ** 2
    freqs = np.fft.rfftfreq(len(c), dt)
    band = (freqs >= 4) & (freqs <= 8)
    rest = freqs >= 0.5
    if not band.any() or power[rest].sum() == 0:
        return None
    if power[band].sum() / power[rest].sum() < 0.4:
        return None
    return float(freqs[band][np.argmax(power[band])]), float(np.sqrt(2) * c.std())


def voice_type_scores(lo, hi, centre):
    scores = {}
    for name, (a, b) in VOICE_TYPES.items():
        overlap = max(0.0, min(hi, b) - max(lo, a))
        coverage = overlap / (hi - lo)           # how much of your range fits
        fill = overlap / (b - a)                 # how much of the type you reach
        closeness = max(0.0, 1 - abs(centre - (a + b) / 2) / 12)
        scores[name] = 0.4 * coverage + 0.3 * fill + 0.3 * closeness
    return sorted(scores.items(), key=lambda kv: -kv[1])


def build_report(times, hz, labels, hnr, calibrated, title="VVC report"):
    times = np.asarray(times, dtype=float)
    hz = np.asarray(hz, dtype=float)
    labels = np.array([lab if lab else "none" for lab in labels])
    hnr = np.asarray(hnr, dtype=float)
    midi = hz_to_midi(np.where(hz > 0, hz, np.nan))
    voiced = np.isfinite(midi)
    L = [f"# {title}", ""]

    if voiced.sum() < 10:
        L.append("Not enough singing detected to analyse.")
        return "\n".join(L), None

    dt = np.median(np.diff(times)) if len(times) > 1 else 0
    L.append(f"Session {times[-1] - times[0]:.1f} s, voiced {voiced.sum() * dt:.1f} s. "
             f"Register model: {'calibrated on your clips' if calibrated else 'UNCALIBRATED defaults (run calibrate.py)'}.")
    L.append("")

    # --- Range ---
    v = midi[voiced]
    lo, hi = np.percentile(v, [2, 98])
    modal = voiced & (labels != "falsetto")
    L.append("## Range")
    L.append(f"- Full range: **{midi_name(lo)} to {midi_name(hi)}** "
             f"({hi - lo:.0f} semitones; 2nd-98th percentile, so stray glitches are ignored)")
    if modal.sum() >= 10:
        mlo, mhi = np.percentile(midi[modal], [2, 98])
        L.append(f"- Without falsetto: {midi_name(mlo)} to {midi_name(mhi)}")
    else:
        mlo, mhi = lo, hi
    fals = voiced & (labels == "falsetto")
    if fals.sum() >= 10:
        flo, fhi = np.percentile(midi[fals], [2, 98])
        L.append(f"- Falsetto: {midi_name(flo)} to {midi_name(fhi)}")
    L.append("")

    # --- Registers ---
    L.append("## Register use")
    for r in REGISTERS:
        n = (voiced & (labels == r)).sum()
        if n:
            L.append(f"- {r}: {100 * n / voiced.sum():.0f}% of sung time")
    chest_notes = np.round(midi[voiced & (labels == "chest")])
    light_notes = np.round(midi[voiced & np.isin(labels, ["head", "falsetto"])])
    if len(chest_notes) >= 5 and len(light_notes) >= 5:
        top_chest = np.percentile(chest_notes, 95)
        bottom_light = np.percentile(light_notes, 5)
        L.append(f"- Chest to head/falsetto shift: around **{midi_name(top_chest)} / "
                 f"{midi_name(bottom_light)}** (highest chest vs lowest light-register notes)")
    else:
        L.append("- Shift point: not observed (sing a slow slide from low to high to find it)")
    L.append("")

    # --- Voice type ---
    L.append("## Closest voice type (rough)")
    if mhi - mlo < 7:
        L.append("- Range too narrow to judge. Sing a slow slide from your lowest to "
                 "highest comfortable note, then check again.")
    else:
        centre = float(np.median(midi[modal])) if modal.sum() else float(np.median(v))
        ranked = voice_type_scores(mlo, mhi, centre)
        L.append(f"- Uses your range **without falsetto** ({midi_name(mlo)}-{midi_name(mhi)}) "
                 f"and where you sing most (median {midi_name(centre)}), "
                 "because the same note in chest or falsetto doesn't mean the same voice.")
        for name, s in ranked[:3]:
            a, b = VOICE_TYPES[name]
            L.append(f"- {name} ({midi_name(a)}-{midi_name(b)}): match {s:.2f}")
        L.append("- Pitch alone can't separate, e.g., mezzo from tenor; a teacher also "
                 "listens to timbre and where the voice feels comfortable.")
    L.append("")

    # --- Pitch accuracy ---
    segs = held_notes(times, midi)
    L.append("## Pitch accuracy (held notes)")
    if segs:
        cents = np.concatenate([100 * (midi[a:b] - np.round(midi[a:b])) for a, b in segs])
        within = [np.std(100 * (midi[a:b] - np.round(midi[a:b]))) for a, b in segs]
        L.append(f"- {len(segs)} held notes (>= {HELD_MIN_S} s on one semitone)")
        L.append(f"- Average distance from the nearest note: **{np.mean(np.abs(cents)):.0f} cents** "
                 "(100 cents = 1 semitone)")
        L.append(f"- Time within +/-25 cents: **{100 * np.mean(np.abs(cents) <= 25):.0f}%**")
        L.append(f"- Steadiness (median wobble inside a note): {np.median(within):.0f} cents")
        L.append("- Measured against standard A440 tuning, not against the note you meant to sing.")
    else:
        L.append("- No held notes found. Hold a few notes for 1 s or more.")
    L.append("")

    # --- Vibrato ---
    L.append("## Vibrato")
    long_segs = [(a, b) for a, b in segs if times[b - 1] - times[a] >= VIBRATO_MIN_S]
    found = [vibrato(times[a:b], 100 * (midi[a:b] - np.round(midi[a:b]))) for a, b in long_segs]
    found = [f for f in found if f]
    if not long_segs:
        L.append(f"- No notes held >= {VIBRATO_MIN_S} s.")
    else:
        L.append(f"- Detected in {len(found)} of {len(long_segs)} long notes")
        if found:
            L.append(f"- Median rate {np.median([f[0] for f in found]):.1f} Hz, "
                     f"extent +/-{np.median([f[1] for f in found]):.0f} cents")
    L.append("")

    # --- Clarity ---
    L.append("## Clarity")
    h = hnr[voiced & np.isfinite(hnr)]
    if h.size:
        L.append(f"- Harmonic-to-noise (median): **{np.median(h):.1f} dB**. "
                 "Higher = clearer tone, lower = breathier. Depends on mic and room too, "
                 "so compare your own sessions rather than with other people.")
    return "\n".join(L), _figure(times, midi, labels, voiced)


def _figure(times, midi, labels, voiced):
    fig = Figure(figsize=(11, 4.5))
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1]})
    for r in REGISTERS + ["none"]:
        m = voiced & (labels == r)
        if m.any():
            ax1.scatter(times[m], midi[m], s=6, color=COLORS.get(r, COLORS[None]),
                        label=r if r != "none" else "unknown")
    lo, hi = int(np.nanmin(midi[voiced])) - 2, int(np.nanmax(midi[voiced])) + 3
    ticks = [m for m in range(lo, hi) if m % 12 in (0, 4, 7)]
    for ax in (ax1, ax2):
        ax.set_yticks(ticks)
        ax.set_yticklabels([midi_name(m) for m in ticks])
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)
    ax1.set_xlabel("seconds")
    ax1.set_title("Pitch over time, coloured by register")
    ax1.legend(loc="upper right", markerscale=3, fontsize=8)

    notes = np.arange(lo, hi)
    left = np.zeros(len(notes))
    for r in REGISTERS:
        counts = np.array([((np.round(midi) == n) & voiced & (labels == r)).sum() for n in notes])
        if counts.any():
            ax2.barh(notes, counts, left=left, color=COLORS[r], height=0.85)
            left += counts
    ax2.set_title("Time on each note")
    ax2.set_xlabel("frames")
    fig.tight_layout()
    return fig
