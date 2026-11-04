"""Courbes de loss (PNG, sans fenêtre : backend Agg)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_history(history_path, out_path, results_path=None):
    history = json.loads(Path(history_path).read_text())
    results = json.loads(Path(results_path).read_text()) if results_path else None
    ln2 = math.log(2)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    w = 25
    smooth = moving_average(history["train_loss"], w)
    ax.plot(history["iter"], np.asarray(history["train_loss"]) / ln2, color="#c9c9c9", lw=0.6, label="train (batch)")
    ax.plot(history["iter"][w - 1 :], smooth / ln2, color="#1f5aa6", lw=1.4, label=f"train (moyenne sur {w} it.)")
    ev = history["eval"]
    ax.plot([e["iter"] for e in ev], [e["val"] / ln2 for e in ev], "o-", color="#d1495b", ms=3, lw=1.4, label="val (estimation)")
    if results:
        for key, label, style in (("unigram", "unigramme", ":"), ("bigram", "bigramme", "--")):
            b = results["baselines"][key]["bpc"]
            ax.axhline(b, color="#555555", ls=style, lw=1)
            ax.text(history["iter"][-1], b, f" {label} {b:.2f}", va="bottom", ha="right", fontsize=8, color="#555555")
        m = results["model"]["bpc"]
        ax.axhline(m, color="#d1495b", ls="-.", lw=0.8)
        ax.text(history["iter"][-1], m, f" val complète {m:.2f}", va="top", ha="right", fontsize=8, color="#d1495b")
    ax.set_xlabel("itération")
    ax.set_ylabel("bits par caractère")
    ax.set_ylim(bottom=max(0.0, min(smooth / ln2) - 0.3) if len(smooth) else 0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("nanogpt-numpy sur Les Misérables")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
