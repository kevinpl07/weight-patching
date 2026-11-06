"""Hero figure for BLOG_DRAFT.md: pilot 1, seed 0, held-out accuracy after the edit.

Run: .venv\\Scripts\\python.exe blog\\make_hero.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e1"
BLUE = "#2a78d6"    # unedited
ORANGE = "#eb6834"  # Opus edit

arms = json.load(open(ROOT / "runs/seed0/results.json"))["arms"]
EDIT_STEP = arms["unedited"]["history"][0]["step"]


def curve(name):
    h = arms[name]["history"]
    return [(p["step"] - EDIT_STEP) / 1000 for p in h], [100 * p["test_acc"] for p in h]


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
})
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# Random edits: context, recessive
for i, name in enumerate(["random0", "random1", "random2"]):
    x, y = curve(name)
    ax.plot(x, y, color=MUTED, lw=1.2, alpha=0.9, zorder=2,
            label="Random edits, same size (3 draws)" if i == 0 else None)

# Unedited
x, y = curve("unedited")
ax.plot(x, y, color=BLUE, lw=2.2, zorder=4, label="No edit (plain AdamW)")
ax.plot(x[-1], y[-1], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=5)

# Opus
xo, yo = curve("opus")
ax.plot(xo, yo, color=ORANGE, lw=2.2, zorder=4, label="Opus 5.5 edit")

# 95% target
ax.axhline(95, color=TEXT_2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(61.5, 96.2, "95% target", ha="right", va="bottom", color=TEXT_2, fontsize=10)

# 50% markers
s = arms["opus"]["steps_to"]["test50"] / 1000
u = arms["unedited"]["steps_to"]["test50"] / 1000
ax.annotate(f"Opus reaches 50% after {s:.1f}k steps\n({u / s:.1f}× sooner than no edit)",
            xy=(s, 50), xytext=(13, 14), color=TEXT, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
ax.plot([s], [50], "o", ms=7, color=ORANGE, mec=SURFACE, mew=2, zorder=6)

# Unedited end
t95 = arms["unedited"]["steps_to"]["test95"] / 1000
ax.annotate(f"No edit reaches 95%\nat {t95:.1f}k steps",
            xy=(x[-1], y[-1]), xytext=(x[-1] + 2.5, 68), color=TEXT, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))

# Opus plateau
best = max(yo)
ax.annotate(f"Opus stalls at {best:.0f}%: the 59 neurons it switched off\nnever fire again, so the network can't finish",
            xy=(52, yo[-60]), xytext=(33, 52), color=TEXT, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))

ax.set_xlim(0, 62)
ax.set_ylim(0, 102)
ax.set_xlabel("Training steps after the edit (thousands)")
ax.set_ylabel("Held-out accuracy (%)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0)

leg = ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=TEXT)

fig.text(0.065, 0.955, "The edit gets there faster, then never gets there",
         fontsize=15, weight="bold", color=TEXT, ha="left")
fig.text(0.065, 0.912, "Grokking (a + b) mod 97, seed 0. One Opus 5.5 weight edit at the memorization plateau, then plain AdamW.",
         fontsize=10.5, color=TEXT_2, ha="left")
fig.subplots_adjust(left=0.065, right=0.97, top=0.86, bottom=0.11)

for ext in ("png", "svg"):
    fig.savefig(OUT / f"hero_grokking.{ext}", facecolor=SURFACE)
print("wrote", OUT / "hero_grokking.png")
