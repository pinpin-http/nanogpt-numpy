"""Boucle d'entraînement."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA_DIR, get_batch, load_splits
from .evaluate import baselines, bits_per_char, model_nll
from .model import GPT, GPTConfig
from .optim import AdamW, clip_grad_norm, cosine_lr


@dataclass
class TrainConfig:
    # modèle
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    tie_weights: bool = True
    # optimisation
    batch_size: int = 32
    max_iters: int = 5000
    # Budget de temps en minutes. À 0 (défaut), seul max_iters arrête
    # l'entraînement, qui est alors reproductible au pas près.
    time_budget_min: float = 0.0
    lr: float = 2e-3
    min_lr: float = 2e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # suivi
    eval_interval: int = 250
    eval_batches: int = 16
    log_interval: int = 10
    seed: int = 1337
    data_dir: str = str(DEFAULT_DATA_DIR)
    out: str = "checkpoints/miserables"


def estimate_loss(model, ids, cfg: TrainConfig, seed: int) -> float:
    """Loss moyenne sur des batches tirés avec une graine fixe (mêmes fenêtres à chaque éval)."""
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(ids, cfg.batch_size, cfg.block_size, rng)
        losses.append(model.forward(x, y)[1])
    return float(np.mean(losses))


def train(cfg: TrainConfig) -> dict:
    tok, train_ids, val_ids = load_splits(cfg.data_dir)
    print(f"corpus : {len(train_ids):,} car. train, {len(val_ids):,} car. val, vocabulaire {tok.vocab_size}")

    mcfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        tie_weights=cfg.tie_weights,
    )
    model = GPT(mcfg, seed=cfg.seed)
    print(f"modèle : {model.num_params():,} paramètres")
    named = list(model.named_parameters())
    grads = [g for _, _, g in named]
    opt = AdamW(named, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)

    history = {"iter": [], "train_loss": [], "lr": [], "grad_norm": [], "eval": []}
    out = Path(cfg.out)
    t0 = time.perf_counter()
    budget_s = cfg.time_budget_min * 60

    def progress(it: int) -> float:
        # Part du budget consommée : itérations ou temps, la plus avancée des deux.
        # Le cosinus du taux d'apprentissage suit cette progression, si bien que
        # la décroissance va à son terme même sur une machine chargée.
        p = it / cfg.max_iters
        if budget_s > 0:
            p = max(p, (time.perf_counter() - t0) / budget_s)
        return p

    it = 0
    while True:
        p = progress(it)
        if it % cfg.eval_interval == 0 or p >= 1.0:
            tr = estimate_loss(model, train_ids, cfg, seed=1)
            va = estimate_loss(model, val_ids, cfg, seed=2)
            elapsed = time.perf_counter() - t0
            history["eval"].append({"iter": it, "train": tr, "val": va, "time_s": elapsed})
            print(f"[éval] it {it:5d} | train {tr:.4f} | val {va:.4f} ({bits_per_char(va):.3f} bpc) | {elapsed / 60:.1f} min")
        if p >= 1.0:
            break

        lr = cosine_lr(it, p, cfg.lr, cfg.min_lr, cfg.warmup_iters)
        x, y = get_batch(train_ids, cfg.batch_size, cfg.block_size, rng)
        model.zero_grad()
        _, loss = model.forward(x, y)
        model.backward()
        norm = clip_grad_norm(grads, cfg.grad_clip)
        opt.step(lr)

        history["iter"].append(it)
        history["train_loss"].append(loss)
        history["lr"].append(lr)
        history["grad_norm"].append(norm)
        if it % cfg.log_interval == 0:
            dt = (time.perf_counter() - t0) / (it + 1)
            print(f"it {it:5d} | loss {loss:.4f} | lr {lr:.2e} | |g| {norm:.2f} | {dt:.2f} s/it | {p:.0%}", flush=True)
        it += 1

    train_time = time.perf_counter() - t0
    meta = {
        "vocab": tok.chars,
        "train_config": asdict(cfg),
        "iters_done": it,
        "train_time_s": train_time,
        "num_params": model.num_params(),
    }
    model.save(out, extra=meta)
    (out.parent / f"{out.name}_history.json").write_text(json.dumps(history))
    print(f"entraînement : {train_time / 60:.1f} min, checkpoint dans {out.with_suffix('.npz')}")
    return {"model": model, "tokenizer": tok, "history": history, "meta": meta}


def final_report(model, data_dir: str, block_size: int, stride: int | None = None) -> dict:
    """Loss sur toute la validation, comparée aux baselines unigramme et bigramme."""
    tok, train_ids, val_ids = load_splits(data_dir)
    t0 = time.perf_counter()
    val_nats = model_nll(model, val_ids, block_size, stride)
    eval_time = time.perf_counter() - t0
    base = baselines(train_ids, val_ids, tok.vocab_size)
    return {
        "val_chars": int(len(val_ids)),
        "train_chars": int(len(train_ids)),
        "vocab_size": tok.vocab_size,
        "stride": stride or block_size // 2,
        "model": {"nats": val_nats, "bpc": bits_per_char(val_nats)},
        "baselines": base,
        "eval_time_s": eval_time,
    }
