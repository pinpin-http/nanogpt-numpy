"""Mesures : loss du modèle sur tout un split et baselines n-grammes sur les mêmes cibles.

Toutes les mesures portent sur la prédiction de ids[1:] (chaque caractère sauf
le premier, qui n'a pas de contexte), en nats par caractère. On convertit en
bits par caractère en divisant par ln 2.
"""

from __future__ import annotations

import math

import numpy as np

from .layers import log_softmax


def bits_per_char(nats: float) -> float:
    return nats / math.log(2)


def window_plan(n: int, block_size: int, stride: int):
    """Découpe en fenêtres glissantes pour évaluer ids[1:] une seule fois chacun.

    Renvoie une liste de (début, première position notée dans la fenêtre). Hormis
    la première fenêtre, chaque cible notée dispose d'au moins block_size - stride
    caractères de contexte.
    """
    if n < block_size + 1:
        raise ValueError("split trop court pour une fenêtre complète")
    last = n - 1 - block_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    plan, scored_end = [], 0  # scored_end : indice de la dernière cible déjà notée
    for s in starts:
        first_target = max(scored_end + 1, s + 1)
        plan.append((s, first_target - s - 1))
        scored_end = s + block_size
    return plan


def model_nll(model, ids: np.ndarray, block_size: int, stride: int | None = None, batch_size: int = 32):
    """NLL moyenne (nats/caractère) du modèle sur ids[1:], fenêtres glissantes."""
    stride = stride or block_size // 2
    plan = window_plan(len(ids), block_size, stride)
    offsets = np.arange(block_size + 1)
    total, count = 0.0, 0
    for i in range(0, len(plan), batch_size):
        chunk = plan[i : i + batch_size]
        starts = np.array([s for s, _ in chunk])
        windows = ids[starts[:, None] + offsets[None, :]]
        x, y = windows[:, :-1], windows[:, 1:]
        logits, _ = model.forward(x)
        logp = log_softmax(logits.astype(np.float64))
        picked = np.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
        for row, (_, first) in enumerate(chunk):
            total -= picked[row, first:].sum()
            count += block_size - first
    assert count == len(ids) - 1
    return total / count


def unigram_nll(train: np.ndarray, val: np.ndarray, vocab_size: int, alpha: float) -> float:
    counts = np.bincount(train, minlength=vocab_size) + alpha
    logp = np.log(counts / counts.sum())
    return float(-logp[val[1:]].mean())


def bigram_nll(train: np.ndarray, val: np.ndarray, vocab_size: int, alpha: float) -> float:
    pairs = train[:-1] * vocab_size + train[1:]
    counts = np.bincount(pairs, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size)
    counts = counts + alpha
    logp = np.log(counts / counts.sum(axis=1, keepdims=True))
    return float(-logp[val[:-1], val[1:]].mean())


ALPHAS = (1e-3, 1e-2, 0.1, 0.3, 1.0)


def baselines(train: np.ndarray, val: np.ndarray, vocab_size: int):
    """Unigramme et bigramme à lissage additif.

    Le coefficient de lissage est choisi au mieux *sur la validation*, ce qui
    avantage les baselines : la comparaison est donc prudente pour le modèle.
    """
    out = {}
    for name, fn in (("unigram", unigram_nll), ("bigram", bigram_nll)):
        scores = {a: fn(train, val, vocab_size, a) for a in ALPHAS}
        best = min(scores, key=scores.get)
        out[name] = {"nats": scores[best], "bpc": bits_per_char(scores[best]), "alpha": best}
    return out
