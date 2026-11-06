"""Static blog figures for the archived Track A (mid-training edit) experiments.

Reads only site/data/archive/*.json (written by extract_archive_data.py) and writes PNG + SVG into site/img/.
Run from the repository root:  python archive/make_archive_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "site" / "data" / "archive"
IMG = REPO / "site" / "img"
IMG.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e1"
BLUE = "#2a78d6"    # no edit / plain training
ORANGE = "#eb6834"  # Opus edit

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "svg.fonttype": "none",
})


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def header(fig, title, subtitle, x=0.07):
    fig.text(x, 0.955, title, fontsize=15, weight="bold", color=TEXT, ha="left")
    fig.text(x, 0.912, subtitle, fontsize=10.5, color=TEXT_2, ha="left")


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(IMG / f"{name}.{ext}", facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print("wrote", IMG / f"{name}.png")


# --------------------------------------------------------------------------- 1. grokking
def fig_grokking():
    g = load("grokking.json")
    S = g["curves"]["series"]
    seed0 = g["pilot1_per_seed"]["arms"]["seed0"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for i, k in enumerate(["random0", "random1", "random2"]):
        x, y = zip(*S[k])
        ax.plot([v / 1000 for v in x], y, color=MUTED, lw=1.2, zorder=2,
                label="Random edits, same size (3 draws)" if i == 0 else None)
    x, y = zip(*S["unedited"])
    xu = [v / 1000 for v in x]
    ax.plot(xu, y, color=BLUE, lw=2.2, zorder=4, label="No edit (plain AdamW)")
    ax.plot(xu[-1], y[-1], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=5)
    xo, yo = zip(*S["opus"])
    xo = [v / 1000 for v in xo]
    ax.plot(xo, yo, color=ORANGE, lw=2.2, zorder=4, label="Opus 5.5 edit")
    ax.axhline(95, color=TEXT_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(61.5, 96.2, "95% target", ha="right", va="bottom", color=TEXT_2, fontsize=10)
    s50, u50 = seed0["opus"]["steps_to_50"] / 1000, seed0["unedited"]["steps_to_50"] / 1000
    ax.plot([s50], [50], "o", ms=7, color=ORANGE, mec=SURFACE, mew=2, zorder=6)
    ax.annotate(f"Edit reaches 50% after {s50:.1f}k steps\n({u50 / s50:.1f}× sooner than no edit)",
                xy=(s50, 50), xytext=(13, 14), color=TEXT, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    t95 = seed0["unedited"]["steps_to_95"] / 1000
    ax.annotate(f"No edit reaches 95%\nat {t95:.1f}k steps", xy=(xu[-1], y[-1]), xytext=(xu[-1] + 2.5, 68),
                color=TEXT, fontsize=10, arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    best = seed0["opus"]["best_heldout_pct"]
    ax.annotate(f"Edited run stalls at {best:.0f}%: the neurons it switched off\nnever fire again",
                xy=(52, yo[int(len(yo) * 0.83)]), xytext=(31, 54), color=TEXT, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Training steps after the edit (thousands)")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=TEXT)
    header(fig, "The edit gets there faster, then never gets there",
           "Grokking (a + b) mod 97, seed 0. One Opus 5.5 weight edit at the memorization plateau, then plain AdamW.")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.11)
    save(fig, "archive_grokking")


# --------------------------------------------------------------------------- 2. pretraining
def fig_pretrain():
    p = load("pretrain.json")
    S = p["curves"]["series"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.22))
    fig.patch.set_facecolor(SURFACE)
    spec = [("arm_random0", MUTED, 1.4, "Random edit, same size"), ("arm_none", BLUE, 2.2, "No edit"),
            ("arm_agent", ORANGE, 2.2, "Opus 5.5 edit at step 1,000")]
    for ax, (lo, hi) in [(a1, (1000, 4000)), (a2, (1000, 15000))]:
        style(ax)
        for k, c, lw, lab in spec:
            x, y = zip(*[pt for pt in S[k] if lo <= pt[0] <= hi])
            ax.plot(x, y, color=c, lw=lw, label=lab, zorder=4 if c != MUTED else 2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(2.1 if hi <= 4000 else 1.8, 3.4)
        ax.set_xlabel("Training step")
    a1.set_ylabel("Validation loss (nats)")
    a1.set_title("Steps 1,000–4,000: the plateau", fontsize=11, color=TEXT_2, loc="left")
    a2.set_title("Whole run", fontsize=11, color=TEXT_2, loc="left")
    sb = {r["experiment"]: r for r in load("track_a_scoreboard.json")["rows"]}
    r = next(v for k, v in sb.items() if k.startswith("Small-LM"))
    agent = dict(map(tuple, S["arm_agent"]))
    t = r["peak_step"]
    if t in agent:
        a1.annotate(f"{r['peak_lead']:.2f} nats ahead at step {t:,}\n(up to ~{r['max_steps_ahead']} steps ahead)",
                    xy=(t, agent[t]), xytext=(2150, 2.95), color=TEXT, fontsize=10,
                    arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    a1.text(3950, 2.33, "control forms its own\ninduction circuit", ha="right", va="bottom", fontsize=10, color=TEXT)
    fin = {k: S[k][-1][1] for k in S}
    a2.text(14800, 3.3, f"At step 15,000:\nedit {fin['arm_agent']:.3f}\nno edit {fin['arm_none']:.3f}\n"
            f"random {fin['arm_random0']:.3f}", ha="right", va="top", fontsize=10, color=TEXT)
    a1.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=TEXT)
    header(fig, "A real head start on a plateau, closed within ~1,000 steps",
           "2-layer attention-only GPT on Python code, seed 0. The Opus 5.5 edit wrote an induction circuit at step "
           "1,000; training continued unchanged.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.11)
    save(fig, "archive_pretrain")


# --------------------------------------------------------------------------- 3. RL head start
def fig_rl():
    r = load("rl.json")["head_start"]
    E = r["evals"]

    def series(arm, k="symT1_acc"):
        return [e["step"] for e in E[arm]], [e[k] for e in E[arm]]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    xs, _ = series("armA_s0_100")
    # individual seeds, faint
    for arm, c in [("armA_s0_100", BLUE), ("armA_s1_100", BLUE), ("armB_s0_100", ORANGE), ("armB_s1_100", ORANGE)]:
        x, y = series(arm)
        ax.plot(x, y, color=c, lw=1, alpha=0.35, zorder=2)
    meanA = [(a + b) / 2 for a, b in zip(series("armA_s0_100")[1], series("armA_s1_100")[1])]
    meanB = [(a + b) / 2 for a, b in zip(series("armB_s0_100")[1], series("armB_s1_100")[1])]
    xc, yc = series("armC_s0_100")
    ax.plot(xc, yc, color=MUTED, lw=1.6, ls=(0, (4, 2)), marker="s", ms=6, mec=SURFACE, mew=1.5, zorder=3,
            label="Random edit, same size (1 seed)")
    ax.plot(xs, meanA, color=BLUE, lw=2.2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=4, label="No edit (mean of 2 seeds)")
    ax.plot(xs, meanB, color=ORANGE, lw=2.2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=5,
            label="Opus 5.5 edit (mean of 2 seeds)")
    d0 = r["b_minus_a_mean_over_seeds"]["symT1_acc"][0]["diff"]
    dend = r["b_minus_a_mean_over_seeds"]["symT1_acc"][-1]["diff"]
    ax.annotate(f"+{d0:.1f} points at the start", xy=(0, meanB[0]), xytext=(9, 31), color=TEXT, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    ax.annotate("gone by step 15: plain GRPO\nfixes the same rambling", xy=(15, meanA[1]), xytext=(22, 33),
                color=TEXT, fontsize=10, arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    ax.text(99, 62.5, f"edit minus no edit at step 100: {dend:+.1f}", ha="right", fontsize=10, color=TEXT)
    ax.set_xlim(-3, 103)
    ax.set_ylim(20, 65)
    ax.set_xticks(xs)
    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("GSM-Symbolic accuracy, sampled at T=1 (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=TEXT)
    header(fig, "A 7-point head start is worth about 7 RL steps",
           "Qwen3.5-0.8B, full-weight GRPO on GSM8K. Faint lines are individual seeds; 500 held-out problems per point.")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.11)
    save(fig, "archive_rl_head_start")


# --------------------------------------------------------------------------- 4. scoreboard
def fig_scoreboard():
    rows = load("track_a_scoreboard.json")["rows"]
    labels = {
        "Grokking MLP (pilot 1)": ("Grokking MLP", "held-out accuracy, points", "mean of 3 seeds"),
        "LLM RL head start (Qwen3.5-0.8B GRPO)": ("LLM RL (GRPO)", "GSM-Symbolic accuracy, points", "mean of 2 seeds"),
        "Small-LM pretraining (induction plateau)": ("Small-LM pretraining", "validation loss, nats saved", "1 seed"),
        "ViT locality (CIFAR-10)": ("ViT on CIFAR-10", "test accuracy, points", "mean of 3 seeds"),
    }
    notes = {}
    for r in rows:
        if r["experiment"].startswith("Grokking"):
            notes[r["experiment"]] = ("reached 50% held-out " + "/".join(f"{s / 1000:.1f}k" for s in r["steps_saved_to_50pct"])
                                      + " steps sooner; never reached 95%")
        elif r["experiment"].startswith("Small-LM"):
            notes[r["experiment"]] = f"up to ~{r['max_steps_ahead']} steps ahead; final gap is seed noise"
        elif r["experiment"].startswith("LLM RL"):
            notes[r["experiment"]] = "lead worth ~7 steps of training; level from step 15"
        else:
            notes[r["experiment"]] = f"best window {r['peak_window']} steps; no head start beyond noise"
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 6.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, r in zip(axes, rows):
        name, unit, n = labels[r["experiment"]]
        pk, fn = r["peak_lead"], r["final_lead"]
        # common floor per unit so a sub-point ViT lead doesn't fill the row like grokking's 39 points
        floor = {"held-out accuracy, points": 0, "GSM-Symbolic accuracy, points": 0, "test accuracy, points": 2.0}.get(unit, 0)
        span = max(abs(pk) * 1.25, abs(fn) * 1.25, floor)
        style(ax, ygrid=False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_xlim(-span, span)
        ax.axvline(0, color=TEXT_2, lw=1, zorder=1)
        ax.plot([fn, pk], [0, 0], color=GRID, lw=3, zorder=2, solid_capstyle="round")
        ax.plot([pk], [0], "o", ms=11, color=ORANGE, mec=SURFACE, mew=2, zorder=4)
        ax.plot([fn], [0], "o", ms=11, color=TEXT, mec=SURFACE, mew=2, zorder=5)
        fmt = (lambda v: f"{v:+.3f}") if "nats" in unit else (lambda v: f"{v:+.1f}")
        ax.text(pk, 0.45, f"best {fmt(pk)}", ha="center", fontsize=9.5, color=TEXT)
        ax.text(fn, -0.75, f"end {fmt(fn)}", ha="center", fontsize=9.5, color=TEXT)
        ax.text(-span, 0.95, f"{name}", fontsize=11, weight="bold", color=TEXT, va="bottom", ha="left",
                transform=ax.transData)
        ax.text(span, 0.95, f"{unit} ({n}) · {notes[r['experiment']]}", fontsize=9, color=TEXT_2,
                va="bottom", ha="right")
        ax.tick_params(axis="x", labelsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=10, color=ORANGE, mec=SURFACE, label="biggest lead during training"),
               Line2D([], [], marker="o", ls="", ms=10, color=TEXT, mec=SURFACE, label="lead at the end of training")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10, labelcolor=TEXT)
    header(fig, "Every mid-training edit's lead was gone by the end",
           "Edited run minus plain training, in each experiment's own metric (right of zero = edit ahead). Axes differ per row.",
           x=0.05)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.84, bottom=0.12, hspace=1.1)
    save(fig, "archive_track_a_scoreboard")


if __name__ == "__main__":
    fig_grokking()
    fig_pretrain()
    fig_rl()
    fig_scoreboard()
