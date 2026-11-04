import math

import numpy as np
import pytest

from nanogpt.data import CharTokenizer, get_batch, strip_gutenberg, unwrap_paragraphs
from nanogpt.evaluate import baselines, bigram_nll, model_nll, unigram_nll, window_plan
from nanogpt.layers import log_softmax
from nanogpt.model import GPT, GPTConfig

RAW = (
    "The Project Gutenberg eBook\r\n\r\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n\r\n"
    "Produced by quelqu'un\r\n\r\n"
    "Première ligne d'un paragraphe\r\ncoupé en deux.\r\n\r\n\r\n"
    "--Une réplique.\r\n\r\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\nlicence...\r\n"
)


def test_nettoyage_gutenberg():
    text = unwrap_paragraphs(strip_gutenberg(RAW))
    assert text == "Première ligne d'un paragraphe coupé en deux.\n--Une réplique.\n"


def test_tokenizer_aller_retour():
    tok = CharTokenizer.from_text("éléphant à l'eau")
    s = "l'éléphant"
    assert tok.decode(tok.encode(s)) == s
    with pytest.raises(ValueError):
        tok.encode("z")


def test_get_batch_decalage():
    ids = np.arange(100)
    x, y = get_batch(ids, 4, 10, np.random.default_rng(0))
    assert x.shape == y.shape == (4, 10)
    np.testing.assert_array_equal(y, x + 1)


@pytest.mark.parametrize("n,block,stride", [(50, 8, 4), (51, 8, 8), (9, 8, 3), (200, 16, 5)])
def test_window_plan_note_chaque_cible_une_fois(n, block, stride):
    seen = np.zeros(n, dtype=int)
    for s, first in window_plan(n, block, stride):
        targets = np.arange(s + 1, s + block + 1)[first:]
        seen[targets] += 1
        if s > 0:
            assert first >= block - stride or s == n - 1 - block
    assert seen[0] == 0 and np.all(seen[1:] == 1)


def test_model_nll_egale_calcul_direct():
    # Sur une séquence qui tient dans une fenêtre, l'évaluation par fenêtres
    # doit retrouver la moyenne des -log p calculée à la main.
    cfg = GPTConfig(vocab_size=7, block_size=12, n_layer=1, n_head=2, n_embd=8, dtype="float64")
    model = GPT(cfg, seed=0)
    ids = np.random.default_rng(0).integers(0, 7, size=13)
    logits, _ = model.forward(ids[None, :-1])
    logp = log_softmax(logits[0])
    expected = -logp[np.arange(12), ids[1:]].mean()
    assert model_nll(model, ids, 12) == pytest.approx(expected, rel=1e-12)


def test_baselines():
    rng = np.random.default_rng(0)
    # Chaîne de Markov très déterministe : le bigramme doit battre largement l'unigramme.
    V = 5
    seq = [0]
    for _ in range(5000):
        seq.append((seq[-1] + 1) % V if rng.random() < 0.95 else int(rng.integers(V)))
    seq = np.array(seq)
    train, val = seq[:4000], seq[4000:]
    assert unigram_nll(train, val, V, 1.0) == pytest.approx(math.log(V), abs=0.02)
    assert bigram_nll(train, val, V, 0.1) < 0.5
    out = baselines(train, val, V)
    assert out["bigram"]["bpc"] < out["unigram"]["bpc"]
