"""Génération autorégressive : température, top-k, et cache de clés/valeurs."""

from __future__ import annotations

import numpy as np

from .layers import softmax


def sample_next(logits: np.ndarray, rng: np.random.Generator, temperature: float = 1.0, top_k: int | None = None) -> int:
    """Tire un indice à partir d'un vecteur de logits (V,). temperature=0 donne l'argmax."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None and 0 < top_k < logits.size:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = softmax(logits)
    return int(rng.choice(logits.size, p=probs))


def generate(
    model,
    prompt_ids,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
    return_logits: bool = False,
):
    """Prolonge prompt_ids de max_new_tokens caractères.

    Sans cache, chaque pas relance le modèle sur les block_size derniers
    caractères. Avec cache, on ne calcule que la nouvelle position ; quand le
    cache est plein (les positions sont absolues), on le reconstruit à partir de
    la deuxième moitié de la fenêtre. Tant que la séquence tient dans
    block_size, les deux chemins donnent exactement les mêmes logits.
    """
    block_size = model.config.block_size
    rng = np.random.default_rng(seed)
    tokens = [int(t) for t in prompt_ids]
    if not tokens:
        raise ValueError("il faut au moins un caractère d'amorce")
    history = []

    if use_cache:
        cache = model.new_cache()
        logits = model.forward_cached(np.array([tokens[-block_size:]]), cache)[0, -1]
    for step in range(max_new_tokens):
        if not use_cache:
            logits = model.forward(np.array([tokens[-block_size:]]))[0][0, -1]
        if return_logits:
            history.append(np.array(logits, copy=True))
        nxt = sample_next(logits, rng, temperature, top_k)
        tokens.append(nxt)
        if use_cache and step < max_new_tokens - 1:
            if model.cache_len(cache) >= block_size:
                cache = model.new_cache()
                logits = model.forward_cached(np.array([tokens[-(block_size // 2) :]]), cache)[0, -1]
            else:
                logits = model.forward_cached(np.array([[nxt]]), cache)[0, -1]
    return (tokens, history) if return_logits else tokens
