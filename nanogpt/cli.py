"""Interface en ligne de commande : python -m nanogpt <commande>."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

from .train import TrainConfig

DEFAULT_CKPT = "checkpoints/miserables"  # sortie de `train`
SHIPPED_CKPT = "models/miserables"  # poids fournis avec le dépôt


def _add_train_args(p: argparse.ArgumentParser) -> None:
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        value = getattr(defaults, f.name)
        if isinstance(value, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "oui", "yes"), default=value)
        else:
            p.add_argument(flag, type=type(value), default=value)


def cmd_download(args):
    from .data import download_corpus

    download_corpus(Path(args.data_dir), force=args.force)


def cmd_train(args):
    from .train import train

    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    train(cfg)


def cmd_baselines(args):
    from .data import load_splits
    from .evaluate import baselines

    tok, train_ids, val_ids = load_splits(args.data_dir)
    for name, b in baselines(train_ids, val_ids, tok.vocab_size).items():
        print(f"{name:8s} {b['nats']:.4f} nats/car. = {b['bpc']:.4f} bpc (lissage alpha={b['alpha']})")


def _load(ckpt):
    from .data import CharTokenizer
    from .model import GPT

    model, meta = GPT.load(ckpt)
    return model, CharTokenizer(meta["vocab"]), meta


def cmd_sample(args):
    from .generate import generate

    model, tok, _ = _load(args.ckpt)
    top_k = args.top_k if args.top_k > 0 else None
    for i in range(args.num_samples):
        t0 = time.perf_counter()
        ids = generate(
            model,
            tok.encode(args.prompt),
            args.length,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed + i,
            use_cache=not args.no_cache,
        )
        dt = time.perf_counter() - t0
        print(tok.decode(ids))
        print(f"--- {args.length} caractères en {dt:.2f} s ({args.length / dt:.0f} car./s)\n")


def cmd_report(args):
    from .generate import generate
    from .plot import plot_history
    from .train import final_report

    model, tok, meta = _load(args.ckpt)
    cfg = meta["train_config"]
    print("évaluation sur toute la validation…")
    results = final_report(model, cfg["data_dir"], model.config.block_size)
    results.update(
        num_params=meta["num_params"],
        iters_done=meta["iters_done"],
        train_time_s=meta["train_time_s"],
        train_config=cfg,
    )
    docs = Path(args.docs)
    docs.mkdir(parents=True, exist_ok=True)

    # Vitesse de génération avec et sans cache, sur la même graine.
    speed = {}
    for use_cache in (True, False):
        t0 = time.perf_counter()
        generate(model, tok.encode("\n"), 300, temperature=0.8, top_k=20, seed=0, use_cache=use_cache)
        speed["avec_cache" if use_cache else "sans_cache"] = 300 / (time.perf_counter() - t0)
    results["generation_chars_per_s"] = speed

    (docs / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    history = Path(args.ckpt).parent / f"{Path(args.ckpt).name}_history.json"
    plot_history(history, docs / "loss.png", docs / "results.json")

    samples = []
    for i, prompt in enumerate(args.prompts):
        ids = generate(model, tok.encode(prompt), args.length, temperature=0.8, top_k=20, seed=i)
        samples.append({"prompt": prompt, "temperature": 0.8, "top_k": 20, "seed": i, "text": tok.decode(ids)})
    (docs / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))

    m, b = results["model"], results["baselines"]
    print(f"modèle    {m['nats']:.4f} nats/car. = {m['bpc']:.4f} bpc")
    print(f"bigramme  {b['bigram']['nats']:.4f} nats/car. = {b['bigram']['bpc']:.4f} bpc")
    print(f"unigramme {b['unigram']['nats']:.4f} nats/car. = {b['unigram']['bpc']:.4f} bpc")
    print(f"génération : {speed['avec_cache']:.0f} car./s avec cache, {speed['sans_cache']:.0f} sans")
    print(f"écrit : {docs / 'results.json'}, {docs / 'loss.png'}, {docs / 'samples.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m nanogpt", description="Un petit GPT en NumPy pur.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="télécharger et nettoyer le corpus")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train", help="entraîner un modèle")
    _add_train_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("baselines", help="loss des modèles unigramme et bigramme sur la validation")
    p.add_argument("--data-dir", default="data")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("sample", help="générer du texte depuis un checkpoint")
    p.add_argument("--ckpt", default=SHIPPED_CKPT)
    p.add_argument("--prompt", default="\n")
    p.add_argument("--length", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20, help="0 pour désactiver")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--no-cache", action="store_true", help="recalculer toute la fenêtre à chaque pas")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("report", help="évaluation complète, courbes et échantillons dans docs/")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--docs", default="docs")
    p.add_argument("--length", type=int, default=400)
    p.add_argument(
        "--prompts",
        nargs="+",
        default=["Jean Valjean", "Javert regarda", "La nuit était", "--Monsieur, dit"],
    )
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)
