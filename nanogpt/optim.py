"""AdamW, planning du taux d'apprentissage et écrêtage du gradient."""

from __future__ import annotations

import math

import numpy as np


class AdamW:
    """Adam avec weight decay découplé (Loshchilov & Hutter, 2019).

    Le weight decay ne s'applique qu'aux tenseurs de dimension >= 2 (matrices et
    embeddings), pas aux biais ni aux gains des LayerNorm, comme dans GPT-2.
    """

    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.items = [(name, p, g) for name, p, g in named_params]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for _, p, _ in self.items]
        self.v = [np.zeros_like(p) for _, p, _ in self.items]
        self.t = 0

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else float(lr)
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        corr1 = 1.0 - b1**self.t
        corr2 = 1.0 - b2**self.t
        for (_, p, g), m, v in zip(self.items, self.m, self.v):
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            if p.ndim >= 2 and self.weight_decay:
                p *= 1.0 - lr * self.weight_decay
            p -= lr * (m / corr1) / (np.sqrt(v / corr2) + self.eps)


def cosine_lr(it: int, progress: float, max_lr: float, min_lr: float, warmup: int) -> float:
    """Montée linéaire sur `warmup` itérations, puis cosinus de max_lr à min_lr.

    `progress` va de 0 à 1 : c'est la part du budget consommée, en itérations
    ou en temps (voir train.py).
    """
    if it < warmup:
        return max_lr * (it + 1) / warmup
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def clip_grad_norm(grads, max_norm: float) -> float:
    """Ramène la norme globale des gradients à max_norm si elle la dépasse. Renvoie la norme avant écrêtage."""
    total = math.sqrt(sum(float(np.vdot(g, g)) for g in grads))
    if max_norm > 0 and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g *= scale
    return total
