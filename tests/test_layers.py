import numpy as np
import pytest
from gradcheck import check_layer, numerical_grad, rel_error

from nanogpt.layers import (
    GELU,
    MLP,
    Block,
    CausalSelfAttention,
    Embedding,
    LayerNorm,
    Linear,
    SoftmaxCrossEntropy,
)

F64 = np.float64


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def randomize(layer, rng, scale=0.5):
    """Remplace l'initialisation par des valeurs quelconques, pour que les
    gains, biais et poids aient des gradients non triviaux."""
    for _, p, _ in layer.named_parameters():
        p[...] = rng.standard_normal(p.shape) * scale + (1.0 if p.ndim == 1 and np.all(p == 1) else 0.0)
    return layer


def test_linear(rng):
    layer = randomize(Linear(5, 7, rng, F64), rng)
    check_layer(layer, rng.standard_normal((2, 3, 5)), rng)


def test_linear_sans_biais(rng):
    layer = randomize(Linear(4, 6, rng, F64, bias=False), rng)
    check_layer(layer, rng.standard_normal((3, 4)), rng)


def test_layernorm(rng):
    layer = randomize(LayerNorm(6, F64), rng)
    check_layer(layer, rng.standard_normal((2, 4, 6)) * 3 + 1, rng)


def test_gelu(rng):
    check_layer(GELU(), rng.standard_normal((3, 4, 5)) * 2, rng)


def test_gelu_valeurs():
    # Valeurs de référence de l'approximation tanh.
    y = GELU().forward(np.array([-3.0, -1.0, 0.0, 1.0, 3.0]))
    np.testing.assert_allclose(y, [-0.00363739, -0.15880801, 0.0, 0.84119199, 2.99636261], atol=1e-7)


def test_embedding_indices_repetes(rng):
    layer = Embedding(5, 3, rng, F64)
    idx = np.array([[0, 2, 2], [4, 0, 2]])
    G = rng.standard_normal((2, 3, 3))

    def loss():
        return float((layer.forward(idx) * G).sum())

    layer.zero_grad()
    layer.forward(idx)
    layer.backward(G)
    numeric = numerical_grad(loss, layer.params["W"])
    assert rel_error(layer.grads["W"], numeric) < 1e-7
    assert np.all(layer.grads["W"][[1, 3]] == 0)  # lignes jamais utilisées


def test_softmax_cross_entropy(rng):
    ce = SoftmaxCrossEntropy()
    logits = rng.standard_normal((2, 5, 7)) * 3
    targets = rng.integers(0, 7, size=(2, 5))
    ce.forward(logits, targets)
    analytic = ce.backward()
    numeric = numerical_grad(lambda: ce.forward(logits, targets), logits)
    assert rel_error(analytic, numeric) < 1e-7


def test_softmax_cross_entropy_stable():
    # Logits énormes : pas de débordement, et la loss vaut log(2) pour deux classes égales.
    ce = SoftmaxCrossEntropy()
    loss = ce.forward(np.array([[1e4, 1e4]]), np.array([0]))
    assert np.isclose(loss, np.log(2))


def test_attention(rng):
    layer = randomize(CausalSelfAttention(8, 2, rng, F64), rng)
    check_layer(layer, rng.standard_normal((2, 5, 8)), rng)


def test_attention_causale(rng):
    # Modifier le dernier pas de temps ne doit rien changer aux sorties précédentes.
    layer = randomize(CausalSelfAttention(8, 2, rng, F64), rng)
    x = rng.standard_normal((1, 6, 8))
    y1 = layer.forward(x).copy()
    x[0, -1] += 10.0
    y2 = layer.forward(x)
    np.testing.assert_allclose(y1[0, :-1], y2[0, :-1], atol=1e-12)
    assert not np.allclose(y1[0, -1], y2[0, -1])


def test_mlp(rng):
    layer = randomize(MLP(6, rng, F64), rng)
    check_layer(layer, rng.standard_normal((2, 3, 6)), rng)


def test_block(rng):
    layer = randomize(Block(8, 2, rng, F64), rng, scale=0.3)
    check_layer(layer, rng.standard_normal((2, 4, 8)), rng)


def test_dtype_float32_conserve(rng):
    # Aucune opération ne doit remonter silencieusement en float64.
    block = Block(8, 2, rng, np.float32)
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    y = block.forward(x)
    assert y.dtype == np.float32
    assert block.backward(np.ones_like(y)).dtype == np.float32
