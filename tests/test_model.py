import numpy as np
import pytest
from gradcheck import numerical_grad, rel_error

from nanogpt.generate import generate, sample_next
from nanogpt.model import GPT, GPTConfig
from nanogpt.optim import AdamW, clip_grad_norm, cosine_lr


def tiny_model(tie=True, seed=0, block_size=8, dtype="float64", **kw):
    cfg = GPTConfig(vocab_size=11, block_size=block_size, n_layer=2, n_head=2, n_embd=8, tie_weights=tie, dtype=dtype)
    for k, v in kw.items():
        setattr(cfg, k, v)
    model = GPT(cfg, seed=seed)
    rng = np.random.default_rng(seed + 100)
    # Poids plus grands que l'initialisation GPT-2 pour des gradients bien mesurables.
    for _, p, _ in model.named_parameters():
        p += rng.standard_normal(p.shape) * 0.3
    return model


@pytest.mark.parametrize("tie", [True, False])
def test_gradient_modele_complet(tie):
    model = tiny_model(tie=tie)
    rng = np.random.default_rng(1)
    x = rng.integers(0, 11, size=(2, 8))
    y = rng.integers(0, 11, size=(2, 8))

    model.zero_grad()
    model.forward(x, y)
    model.backward()
    analytic = {name: g.copy() for name, _, g in model.named_parameters()}

    def loss():
        return model.forward(x, y)[1]

    errors = {name: rel_error(analytic[name], numerical_grad(loss, p)) for name, p, _ in model.named_parameters()}
    bad = {k: v for k, v in errors.items() if v > 1e-6}
    assert not bad, f"gradients faux : {bad}"
    assert ("lm_head.W" in errors) == (not tie)


def test_surapprentissage_petite_sequence():
    """Sur une seule séquence répétée, la loss doit tomber près de zéro."""
    text = "le vieux lion dort au soleil."
    chars = sorted(set(text))
    ids = np.array([chars.index(c) for c in text])
    x, y = ids[None, :-1], ids[None, 1:]
    cfg = GPTConfig(vocab_size=len(chars), block_size=len(text) - 1, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg, seed=0)
    named = list(model.named_parameters())
    opt = AdamW(named, lr=1e-2, weight_decay=0.0)
    first = None
    for it in range(150):
        model.zero_grad()
        _, loss = model.forward(x, y)
        first = loss if first is None else first
        model.backward()
        clip_grad_norm([g for _, _, g in named], 1.0)
        opt.step()
    assert first > 2.0
    assert loss < 0.05, f"loss finale {loss:.3f}"
    # Le modèle restitue la phrase en décodage glouton.
    out = generate(model, ids[:1], len(text) - 1, temperature=0)
    assert "".join(chars[i] for i in out) == text


def test_cache_kv_memes_logits():
    model = tiny_model(block_size=16)
    rng = np.random.default_rng(3)
    seq = rng.integers(0, 11, size=(2, 16))
    full, _ = model.forward(seq)

    cache = model.new_cache()
    parts = [model.forward_cached(seq[:, :5], cache)]  # amorce de 5 positions d'un coup
    for t in range(5, 16):
        parts.append(model.forward_cached(seq[:, t : t + 1], cache))  # puis une à la fois
    cached = np.concatenate(parts, axis=1)
    np.testing.assert_allclose(cached, full, rtol=0, atol=1e-10)
    assert model.cache_len(cache) == 16
    with pytest.raises(ValueError):
        model.forward_cached(seq[:, :1], cache)


def test_generation_cache_identique_sans_cache():
    model = tiny_model(block_size=16)
    prompt = [1, 2, 3]
    kw = dict(temperature=1.0, top_k=5, seed=7, return_logits=True)
    a, la = generate(model, prompt, 13, use_cache=True, **kw)  # 3 + 13 = 16 = block_size
    b, lb = generate(model, prompt, 13, use_cache=False, **kw)
    assert a == b
    np.testing.assert_allclose(np.array(la), np.array(lb), atol=1e-10)


def test_generation_au_dela_du_bloc():
    # Au-delà de block_size le cache est reconstruit : la génération continue sans erreur.
    model = tiny_model(block_size=8)
    out = generate(model, [0], 40, temperature=0.9, seed=0)
    assert len(out) == 41 and all(0 <= t < 11 for t in out)


def test_sample_next_top_k_et_temperature():
    rng = np.random.default_rng(0)
    logits = np.array([0.0, 5.0, 4.0, -1.0])
    assert sample_next(logits, rng, temperature=0) == 1
    draws = {sample_next(logits, rng, temperature=1.0, top_k=2) for _ in range(200)}
    assert draws <= {1, 2}


def test_sauvegarde_rechargement(tmp_path):
    model = tiny_model(dtype="float32")
    model.save(tmp_path / "m", extra={"vocab": list("abcdefghijk")})
    loaded, meta = GPT.load(tmp_path / "m")
    x = np.array([[1, 2, 3]])
    np.testing.assert_array_equal(model.forward(x)[0], loaded.forward(x)[0])
    assert meta["vocab"][0] == "a"


def test_cosine_lr():
    # Montée linéaire pendant le warmup, puis cosinus en fonction de la progression.
    assert cosine_lr(0, 0.0, 1.0, 0.1, 10) == pytest.approx(0.1)
    assert cosine_lr(9, 0.09, 1.0, 0.1, 10) == pytest.approx(1.0)
    assert cosine_lr(10, 0.0, 1.0, 0.1, 10) == pytest.approx(1.0)
    assert cosine_lr(50, 0.5, 1.0, 0.1, 10) == pytest.approx(0.55)
    assert cosine_lr(99, 1.0, 1.0, 0.1, 10) == pytest.approx(0.1)
    assert cosine_lr(99, 1.7, 1.0, 0.1, 10) == pytest.approx(0.1)  # progression bornée à 1


def test_adamw_un_pas():
    # Au premier pas, Adam déplace chaque coordonnée d'environ lr * signe(g).
    p = np.array([[1.0, -2.0]])
    g = np.array([[0.5, -3.0]])
    opt = AdamW([("W", p, g)], lr=0.1, weight_decay=0.0)
    opt.step()
    np.testing.assert_allclose(p, [[0.9, -1.9]], atol=1e-6)
    # Weight decay découplé : p <- p * (1 - lr * wd) avant le pas d'Adam.
    p2, g2 = np.array([[1.0]]), np.array([[0.0]])
    AdamW([("W", p2, g2)], lr=0.1, weight_decay=0.5).step()
    np.testing.assert_allclose(p2, [[0.95]])


def test_clip_grad_norm():
    grads = [np.array([3.0]), np.array([[4.0]])]
    assert clip_grad_norm(grads, 1.0) == pytest.approx(5.0)
    total = np.sqrt(sum((g**2).sum() for g in grads))
    assert total == pytest.approx(1.0, abs=1e-5)
