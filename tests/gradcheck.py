"""Outils de vérification de gradient par différences finies centrées (float64)."""

import numpy as np

EPS = 1e-6
TOL = 1e-7


def numerical_grad(f, x, eps=EPS):
    """Gradient de f() par rapport à x, en modifiant x sur place puis en le restaurant."""
    grad = np.zeros_like(x)
    flat, gflat = x.reshape(-1), grad.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        fp = f()
        flat[i] = old - eps
        fm = f()
        flat[i] = old
        gflat[i] = (fp - fm) / (2 * eps)
    return grad


def rel_error(analytic, numeric):
    """Erreur max rapportée à l'amplitude du gradient."""
    scale = max(np.abs(analytic).max(), np.abs(numeric).max(), 1e-12)
    return float(np.abs(analytic - numeric).max() / scale)


def check_layer(layer, x, rng, tol=TOL):
    """Vérifie dx et les gradients de tous les paramètres d'une couche.

    La loss scalaire est sum(forward(x) * G) avec G aléatoire fixé, si bien que
    le gradient amont reçu par backward vaut G.
    """
    G = rng.standard_normal(layer.forward(x).shape)

    def loss():
        return float((layer.forward(x) * G).sum())

    layer.zero_grad()
    layer.forward(x)
    dx = layer.backward(G)
    analytic = {name: g.copy() for name, _, g in layer.named_parameters()}

    errors = {}
    if dx is not None:
        errors["x"] = rel_error(dx, numerical_grad(loss, x))
    for name, p, _ in layer.named_parameters():
        errors[name] = rel_error(analytic[name], numerical_grad(loss, p))
    bad = {k: v for k, v in errors.items() if v > tol}
    assert not bad, f"gradients faux : {bad}"
    return errors
