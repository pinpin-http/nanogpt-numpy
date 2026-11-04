"""Courbes de loss (PNG, sans fenêtre : backend Agg)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
TRAIN = "#2a78d6"
VAL = "#eb6834"
RAW = "#c9c8c3"


def centered_average(x, w):
    """Moyenne glissante centrée ; renvoie (indices, valeurs)."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.arange(len(x)), x
    smooth = np.convolve(x, np.ones(w) / w, mode="valid")
    start = (w - 1) // 2
    return np.arange(start, start + len(smooth)), smooth


def plot_history(history_path, out_path, results_path=None):
    history = json.loads(Path(history_path).read_text())
    results = json.loads(Path(results_path).read_text()) if results_path else None
    ln2 = math.log(2)
    iters = np.asarray(history["iter"])
    raw = np.asarray(history["train_loss"]) / ln2
    idx, smooth = centered_average(raw, 25)
    ev = history["eval"]
    ev_it = [e["iter"] for e in ev]
    ev_val = [e["val"] / ln2 for e in ev]

    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": MUTED})
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(iters, raw, color=RAW, lw=0.6, label="train, batch par batch")
    ax.plot(iters[idx], smooth, color=TRAIN, lw=2, label="train, moyenne glissante (25 it.)")
    ax.plot(ev_it, ev_val, "o-", color=VAL, ms=4, lw=2, label="val, estimation sur 16 batches")

    x_end = iters[-1]
    levels = [smooth.min(), min(ev_val)]
    if results:
        for key, label, style in (("unigram", "unigramme", ":"), ("bigram", "bigramme", "--")):
            b = results["baselines"][key]["bpc"]
            levels.append(b)
            ax.axhline(b, color=MUTED, ls=style, lw=1)
            ax.text(x_end, b + 0.03, f"{label} {b:.2f} bpc", va="bottom", ha="right", color=MUTED)
        m = results["model"]["bpc"]
        levels.append(m)
        ax.plot([x_end], [m], "D", color=VAL, ms=7, mec=SURFACE, mew=1.5, clip_on=False, zorder=5)
        ax.text(x_end, m - 0.05, f"val complète, fin : {m:.2f} bpc", va="top", ha="right", color=MUTED)

    # Le départ (~6,7 bpc, prédiction uniforme) écraserait le reste : on coupe à 5.
    ax.set_ylim(min(levels) - 0.3, 5.0)
    ax.set_xlim(0, x_end)
    ax.set_xlabel("itération")
    ax.set_ylabel("bits par caractère")
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
