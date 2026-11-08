"""Live window layout: pitch graph, current note + register, glowing head."""
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon

from vvc.pitch import FMAX_HZ, FMIN_HZ, NOTE_NAMES, hz_to_midi, hz_to_note
from vvc.register import COLORS, REGISTERS

# Where singers commonly *feel* each register (imagery, not acoustics).
GLOW = {
    "chest":    {"xy": (7.3, 2.4), "r": 2.3, "alpha": 0.55,
                 "text": "Forward from the chest:\nround, full, powerful"},
    "mix":      {"xy": (7.0, 6.1), "r": 1.4, "alpha": 0.55,
                 "text": "In the mask / nose:\nbuzzy, bright, balanced"},
    "head":     {"xy": (6.6, 9.0), "r": 1.7, "alpha": 0.50,
                 "text": "Up and ahead of the forehead:\nlight, ringing"},
    "falsetto": {"xy": (4.4, 11.2), "r": 2.2, "alpha": 0.30,
                 "text": "Floating above the head:\nairy, soft"},
}


def _note_ticks(ax):
    lo, hi = int(hz_to_midi(FMIN_HZ)), int(hz_to_midi(FMAX_HZ)) + 1
    ticks = [m for m in range(lo, hi + 1) if m % 12 in (0, 4, 7)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in ticks])


def _draw_head(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-3.4, 13.6)
    ax.set_aspect("equal")
    ax.axis("off")
    skin = dict(facecolor="#d9d9de", edgecolor="#55555c", lw=1.5, zorder=3)
    ax.add_patch(FancyBboxPatch((1.6, 0.2), 4.4, 3.3,
                                boxstyle="round,pad=0,rounding_size=1.2", **skin))
    ax.add_patch(FancyBboxPatch((3.5, 3.0), 1.3, 1.9,
                                boxstyle="round,pad=0,rounding_size=0.3", **skin))
    ax.add_patch(Ellipse((4.2, 6.6), 3.7, 4.3, **skin))
    ax.add_patch(Polygon([(5.95, 7.0), (6.65, 6.15), (5.98, 5.95)], closed=True, **skin))
    ax.plot([5.55, 5.95], [5.2, 5.25], color="#55555c", lw=1.5, zorder=4)  # mouth
    ax.add_patch(Circle((5.25, 7.5), 0.13, color="#55555c", zorder=4))      # eye

    glows = {}
    for reg, g in GLOW.items():
        rings = []
        for k in range(6):  # stacked translucent rings = soft glow
            ring = Circle(g["xy"], g["r"] * (1 - k * 0.15), color=COLORS[reg],
                          alpha=0.0, lw=0, zorder=2 if reg == "chest" else 5)
            ax.add_patch(ring)
            rings.append(ring)
        glows[reg] = rings
    caption = ax.text(5, -0.3, "", ha="center", va="top", fontsize=10)
    ax.text(5, -2.3, "Singer imagery for where each register is felt.\n"
                     "All sound is made at the vocal folds.",
            ha="center", va="top", fontsize=7.5, color="grey")
    return glows, caption


class LiveView:
    def __init__(self, fig, calibrated):
        gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[1, 2.8],
                              left=0.07, right=0.98, top=0.88, bottom=0.08,
                              wspace=0.12, hspace=0.25)
        self.ax_pitch = fig.add_subplot(gs[:, 0])
        self.ax_now = fig.add_subplot(gs[0, 1])
        self.ax_head = fig.add_subplot(gs[1, 1])

        ax = self.ax_pitch
        self.scatter = ax.scatter([], [], s=10)
        _note_ticks(ax)
        ax.set_ylim(hz_to_midi(110), hz_to_midi(880))
        ax.set_xlabel("seconds")
        ax.set_title("Pitch (colour = register)")
        ax.grid(alpha=0.3)
        for r in REGISTERS:
            ax.scatter([], [], color=COLORS[r], label=r)
        ax.legend(loc="upper left", fontsize=8, ncol=4)

        a = self.ax_now
        a.axis("off")
        self.note_text = a.text(0.5, 0.78, "—", ha="center", va="center",
                                fontsize=34, fontweight="bold", transform=a.transAxes)
        self.reg_text = a.text(0.5, 0.40, "", ha="center", va="center",
                               fontsize=18, fontweight="bold", transform=a.transAxes)
        self.info_text = a.text(0.5, 0.08, "", ha="center", va="center",
                                fontsize=9, color="grey", transform=a.transAxes)
        self.calibrated = calibrated

        self.glows, self.caption = _draw_head(self.ax_head)
        fig.suptitle("VVC live" + ("" if calibrated else
                     "   (register model uncalibrated: run calibrate.py)"),
                     fontsize=12, y=0.97)

    def update(self, times, midis, labels, hz, label, conf, proc_ms, window_s=10):
        times = np.asarray(times)
        midis = np.asarray(midis)
        now = times[-1] if len(times) else 0
        ok = np.isfinite(midis)
        if ok.any():
            self.scatter.set_offsets(np.column_stack([times[ok], midis[ok]]))
            self.scatter.set_color([COLORS[labels[i]] for i in np.flatnonzero(ok)])
            lo = min(np.nanmin(midis) - 2, hz_to_midi(110))
            hi = max(np.nanmax(midis) + 2, hz_to_midi(880))
            self.ax_pitch.set_ylim(lo, hi)
        self.ax_pitch.set_xlim(max(0, now - window_s), max(window_s, now))

        note, cents = hz_to_note(hz)
        self.note_text.set_text(f"{note} {cents:+d}¢" if note else "—")
        if label:
            self.reg_text.set_text(f"{label}  ({conf * 100:.0f}%)")
            self.reg_text.set_color(COLORS[label])
        else:
            self.reg_text.set_text("")
        self.info_text.set_text((f"{hz:.1f} Hz · " if np.isfinite(hz) else "")
                                + f"analysis {proc_ms:.1f} ms/update")

        for reg, rings in self.glows.items():
            level = GLOW[reg]["alpha"] * (0.4 + 0.6 * conf) if reg == label else 0.0
            for k, ring in enumerate(rings):
                ring.set_alpha(level * (0.35 + 0.13 * k))
        self.caption.set_text(GLOW[label]["text"] if label else "")
        self.caption.set_color(COLORS[label] if label else "grey")
