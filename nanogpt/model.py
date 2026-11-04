"""Le modèle GPT : embeddings, blocs transformer, normalisation finale, tête de sortie."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .layers import Block, Embedding, LayerNorm, Linear, Module, SoftmaxCrossEntropy


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    tie_weights: bool = True
    dtype: str = "float32"


class GPT(Module):
    def __init__(self, config: GPTConfig, seed: int = 0):
        super().__init__()
        self.config = config
        dtype = np.dtype(config.dtype)
        rng = np.random.default_rng(seed)
        C = config.n_embd
        # Initialisation GPT-2 : les projections qui alimentent le flux résiduel
        # sont réduites d'un facteur sqrt(2 * n_layer).
        proj_std = 0.02 / math.sqrt(2 * config.n_layer)
        self.wte = Embedding(config.vocab_size, C, rng, dtype)
        self.wpe = Embedding(config.block_size, C, rng, dtype)
        self.blocks = [Block(C, config.n_head, rng, dtype, proj_std) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(C, dtype)
        if not config.tie_weights:
            self.lm_head = Linear(C, config.vocab_size, rng, dtype, bias=False)
        self.loss_fn = SoftmaxCrossEntropy()

    # ------------------------------------------------------------------ entraînement

    def _head_forward(self, x):
        if self.config.tie_weights:
            self.head_x = x
            return x @ self.wte.params["W"].T
        return self.lm_head.forward(x)

    def _head_backward(self, dlogits):
        if self.config.tie_weights:
            W = self.wte.params["W"]
            V, C = W.shape
            self.wte.grads["W"] += dlogits.reshape(-1, V).T @ self.head_x.reshape(-1, C)
            return dlogits @ W
        return self.lm_head.backward(dlogits)

    def forward(self, idx, targets=None):
        """idx : (B, T) entiers. Renvoie (logits (B, T, V), loss ou None)."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"séquence de {T} > block_size={self.config.block_size}")
        x = self.wte.forward(idx) + self.wpe.forward(np.arange(T))
        for block in self.blocks:
            x = block.forward(x)
        logits = self._head_forward(self.ln_f.forward(x))
        loss = None if targets is None else self.loss_fn.forward(logits, targets)
        return logits, loss

    def backward(self):
        """Rétropropagation depuis la loss du dernier forward (avec cibles)."""
        dx = self.ln_f.backward(self._head_backward(self.loss_fn.backward()))
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.wte.backward(dx)
        self.wpe.backward(dx.sum(axis=0))

    # ------------------------------------------------------------------ inférence

    def new_cache(self):
        return [{"k": None, "v": None} for _ in self.blocks]

    @staticmethod
    def cache_len(cache) -> int:
        return 0 if cache[0]["k"] is None else cache[0]["k"].shape[2]

    def forward_cached(self, idx, cache):
        """Ne calcule que les nouvelles positions `idx` (B, T_new), en réutilisant le cache."""
        past = self.cache_len(cache)
        T = idx.shape[1]
        if past + T > self.config.block_size:
            raise ValueError("le cache dépasse block_size, il faut le reconstruire")
        x = self.wte.forward(idx) + self.wpe.forward(np.arange(past, past + T))
        for block, c in zip(self.blocks, cache):
            x = block.forward_cached(x, c)
        return self._head_forward(self.ln_f.forward(x))

    # ------------------------------------------------------------------ sauvegarde

    def state_dict(self):
        return {name: p for name, p, _ in self.named_parameters()}

    def load_state_dict(self, state):
        for name, p, _ in self.named_parameters():
            if state[name].shape != p.shape:
                raise ValueError(f"forme incompatible pour {name}")
            p[...] = state[name]

    def save(self, path, extra: dict | None = None):
        """Écrit les poids dans un .npz et la config (+ métadonnées) dans un .json voisin."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"), **self.state_dict())
        meta = {"config": asdict(self.config), **(extra or {})}
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        model = cls(GPTConfig(**meta["config"]))
        with np.load(path.with_suffix(".npz")) as state:
            model.load_state_dict(state)
        return model, meta
