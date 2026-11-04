"""Couches de base, chacune avec un forward et un backward écrits à la main.

Convention : `forward` garde en mémoire ce dont `backward` a besoin, et
`backward(dout)` renvoie le gradient par rapport à l'entrée tout en
*accumulant* les gradients des paramètres dans `self.grads`. Grâce à
l'accumulation, un même poids peut servir à deux endroits (embeddings liés à
la tête de sortie). Il faut donc appeler `zero_grad()` avant chaque passe.
"""

from __future__ import annotations

import math

import numpy as np


class Module:
    """Conteneur de paramètres, à la manière de torch.nn.Module en très réduit."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def add_param(self, name: str, value: np.ndarray) -> None:
        self.params[name] = value
        self.grads[name] = np.zeros_like(value)

    def children(self) -> list[tuple[str, Module]]:
        out = []
        for name, attr in vars(self).items():
            if isinstance(attr, Module):
                out.append((name, attr))
            elif isinstance(attr, list):
                out.extend((f"{name}.{i}", m) for i, m in enumerate(attr) if isinstance(m, Module))
        return out

    def named_parameters(self, prefix: str = ""):
        """Itère sur (nom, paramètre, gradient), sous-modules compris."""
        for name, p in self.params.items():
            yield prefix + name, p, self.grads[name]
        for cname, child in self.children():
            yield from child.named_parameters(f"{prefix}{cname}.")

    def zero_grad(self) -> None:
        for _, _, g in self.named_parameters():
            g.fill(0)

    def num_params(self) -> int:
        return sum(p.size for _, p, _ in self.named_parameters())


class Linear(Module):
    """y = x @ W + b, sur la dernière dimension de x."""

    def __init__(self, n_in, n_out, rng, dtype=np.float32, bias=True, std=0.02):
        super().__init__()
        self.add_param("W", (rng.standard_normal((n_in, n_out)) * std).astype(dtype))
        if bias:
            self.add_param("b", np.zeros(n_out, dtype=dtype))

    def forward(self, x):
        self.x = x
        y = x @ self.params["W"]
        if "b" in self.params:
            y += self.params["b"]
        return y

    def backward(self, dy):
        W = self.params["W"]
        x2 = self.x.reshape(-1, W.shape[0])
        dy2 = dy.reshape(-1, W.shape[1])
        self.grads["W"] += x2.T @ dy2
        if "b" in self.params:
            self.grads["b"] += dy2.sum(axis=0)
        return dy @ W.T


class Embedding(Module):
    """Table de vecteurs indexée par des entiers."""

    def __init__(self, n_rows, dim, rng, dtype=np.float32, std=0.02):
        super().__init__()
        self.add_param("W", (rng.standard_normal((n_rows, dim)) * std).astype(dtype))

    def forward(self, idx):
        self.idx = idx
        return self.params["W"][idx]

    def backward(self, dout):
        dim = self.params["W"].shape[1]
        # Un même indice peut apparaître plusieurs fois : add.at somme les contributions.
        np.add.at(self.grads["W"], self.idx.ravel(), dout.reshape(-1, dim))
        return None


class LayerNorm(Module):
    """Normalisation sur la dernière dimension, avec gain et biais appris."""

    def __init__(self, dim, dtype=np.float32, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.add_param("g", np.ones(dim, dtype=dtype))
        self.add_param("b", np.zeros(dim, dtype=dtype))

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + self.eps)
        xhat = xc * rstd
        self.xhat, self.rstd = xhat, rstd
        return xhat * self.params["g"] + self.params["b"]

    def backward(self, dy):
        xhat, rstd = self.xhat, self.rstd
        dim = xhat.shape[-1]
        self.grads["g"] += (dy * xhat).reshape(-1, dim).sum(axis=0)
        self.grads["b"] += dy.reshape(-1, dim).sum(axis=0)
        dxhat = dy * self.params["g"]
        # Dérivée de (x - mean) / std : on retire la composante moyenne et la
        # composante colinéaire à xhat.
        return rstd * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
        )


_GELU_K = math.sqrt(2.0 / math.pi)
_GELU_C = 0.044715


class GELU(Module):
    """GELU, approximation tanh (celle de GPT-2)."""

    def forward(self, x):
        t = np.tanh(_GELU_K * (x + _GELU_C * x * x * x))
        self.x, self.t = x, t
        return 0.5 * x * (1.0 + t)

    def backward(self, dy):
        x, t = self.x, self.t
        du = _GELU_K * (1.0 + 3.0 * _GELU_C * x * x)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


class SoftmaxCrossEntropy:
    """Entropie croisée moyenne (en nats) entre softmax(logits) et des cibles entières."""

    def forward(self, logits, targets):
        V = logits.shape[-1]
        logp = log_softmax(logits.reshape(-1, V))
        t = targets.reshape(-1)
        self.logp, self.t, self.shape = logp, t, logits.shape
        return float(-logp[np.arange(t.size), t].mean())

    def backward(self):
        n = self.t.size
        d = np.exp(self.logp)
        d[np.arange(n), self.t] -= 1.0
        d /= n
        return d.reshape(self.shape)


class CausalSelfAttention(Module):
    """Attention multi-têtes masquée : la position i ne voit que les positions <= i."""

    def __init__(self, n_embd, n_head, rng, dtype=np.float32, proj_std=0.02):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.c_attn = Linear(n_embd, 3 * n_embd, rng, dtype)
        self.c_proj = Linear(n_embd, n_embd, rng, dtype, std=proj_std)

    def _split_heads(self, x):
        B, T, C = x.shape
        return x.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(x):
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    def _attend(self, q, k, v, offset):
        """q : (B,H,Tq,D), k et v : (B,H,Tk,D). La requête i est à la position offset + i."""
        Tq, Tk, D = q.shape[2], k.shape[2], q.shape[3]
        scale = 1.0 / math.sqrt(D)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        future = np.arange(Tk)[None, :] > (offset + np.arange(Tq))[:, None]
        scores = np.where(future, -np.inf, scores)
        att = softmax(scores)
        return att, att @ v

    def forward(self, x):
        qkv = self.c_attn.forward(x)
        q, k, v = (self._split_heads(t) for t in np.split(qkv, 3, axis=-1))
        att, y = self._attend(q, k, v, offset=0)
        self.q, self.k, self.v, self.att = q, k, v, att
        return self.c_proj.forward(self._merge_heads(y))

    def backward(self, dout):
        q, k, v, att = self.q, self.k, self.v, self.att
        scale = 1.0 / math.sqrt(q.shape[-1])
        dy = self._split_heads(self.c_proj.backward(dout))
        datt = dy @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dy
        # Jacobien du softmax ligne par ligne. Les positions masquées ont att = 0,
        # leur gradient est donc nul sans traitement particulier.
        dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True)) * scale
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q
        dqkv = np.concatenate([self._merge_heads(t) for t in (dq, dk, dv)], axis=-1)
        return self.c_attn.backward(dqkv)

    def forward_cached(self, x, cache):
        """Passe avant pour l'inférence avec cache de clés et de valeurs.

        `cache` est un dict {"k", "v"} (None au départ) complété en place.
        x ne contient que les nouvelles positions.
        """
        qkv = self.c_attn.forward(x)
        q, k, v = (self._split_heads(t) for t in np.split(qkv, 3, axis=-1))
        if cache["k"] is not None:
            k = np.concatenate([cache["k"], k], axis=2)
            v = np.concatenate([cache["v"], v], axis=2)
        cache["k"], cache["v"] = k, v
        offset = k.shape[2] - q.shape[2]
        _, y = self._attend(q, k, v, offset)
        return self.c_proj.forward(self._merge_heads(y))


class MLP(Module):
    def __init__(self, n_embd, rng, dtype=np.float32, proj_std=0.02):
        super().__init__()
        self.fc = Linear(n_embd, 4 * n_embd, rng, dtype)
        self.act = GELU()
        self.proj = Linear(4 * n_embd, n_embd, rng, dtype, std=proj_std)

    def forward(self, x):
        return self.proj.forward(self.act.forward(self.fc.forward(x)))

    def backward(self, dy):
        return self.fc.backward(self.act.backward(self.proj.backward(dy)))


class Block(Module):
    """Bloc transformer pre-LN : x + attn(ln(x)), puis x + mlp(ln(x))."""

    def __init__(self, n_embd, n_head, rng, dtype=np.float32, proj_std=0.02):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, dtype)
        self.attn = CausalSelfAttention(n_embd, n_head, rng, dtype, proj_std)
        self.ln2 = LayerNorm(n_embd, dtype)
        self.mlp = MLP(n_embd, rng, dtype, proj_std)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        return x + self.mlp.forward(self.ln2.forward(x))

    def backward(self, dx):
        dx = dx + self.ln2.backward(self.mlp.backward(dx))
        return dx + self.ln1.backward(self.attn.backward(dx))

    def forward_cached(self, x, cache):
        x = x + self.attn.forward_cached(self.ln1.forward(x), cache)
        return x + self.mlp.forward(self.ln2.forward(x))
