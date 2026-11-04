"""Corpus : téléchargement depuis Project Gutenberg, nettoyage, vocabulaire, découpage."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

# Les Misérables de Victor Hugo (1862), en cinq tomes, domaine public.
GUTENBERG_IDS = {
    17489: "Tome I : Fantine",
    17493: "Tome II : Cosette",
    17494: "Tome III : Marius",
    17518: "Tome IV : L'idylle rue Plumet et l'épopée rue Saint-Denis",
    17519: "Tome V : Jean Valjean",
}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

DEFAULT_DATA_DIR = Path("data")
CORPUS_FILE = "miserables.txt"
VAL_FRACTION = 0.1


def strip_gutenberg(raw: str) -> str:
    """Garde le texte entre les marqueurs START/END de Gutenberg et retire les crédits."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^\*\*\* ?START OF.*$", text, flags=re.M)
    end = re.search(r"^\*\*\* ?END OF.*$", text, flags=re.M)
    if start is None or end is None:
        raise ValueError("marqueurs Project Gutenberg introuvables")
    body = text[start.end() : end.start()]
    body = re.sub(r"^Produced by.*$", "", body, flags=re.M)
    return body


def unwrap_paragraphs(text: str) -> str:
    """Les fichiers Gutenberg sont coupés à ~70 colonnes. On recolle chaque
    paragraphe sur une seule ligne et on sépare les paragraphes par un saut de ligne."""
    paragraphs = re.split(r"\n\s*\n", text)
    lines = [" ".join(line.strip() for line in p.splitlines() if line.strip()) for p in paragraphs]
    return "\n".join(line for line in lines if line) + "\n"


def download_corpus(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    """Télécharge les cinq tomes (une seule fois) et écrit le corpus nettoyé."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for gid, title in GUTENBERG_IDS.items():
        raw_path = raw_dir / f"pg{gid}.txt"
        if force or not raw_path.exists():
            print(f"téléchargement de {title} (Gutenberg #{gid})")
            req = urllib.request.Request(URL.format(id=gid), headers={"User-Agent": "nanogpt-numpy"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_path.write_bytes(resp.read())
        parts.append(unwrap_paragraphs(strip_gutenberg(raw_path.read_text(encoding="utf-8"))))
    text = "".join(parts)
    out = data_dir / CORPUS_FILE
    out.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    print(f"corpus : {out} ({len(text):,} caractères, sha256 {digest}…)")
    return out


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> np.ndarray:
        unknown = set(s) - self.stoi.keys()
        if unknown:
            raise ValueError(f"caractères absents du vocabulaire : {sorted(unknown)!r}")
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def to_json(self) -> str:
        return json.dumps(self.chars, ensure_ascii=False)


def load_splits(data_dir: Path = DEFAULT_DATA_DIR, val_fraction: float = VAL_FRACTION):
    """Renvoie (tokenizer, train_ids, val_ids).

    Le vocabulaire est construit sur tout le corpus pour que la validation ne
    contienne aucun caractère inconnu ; les statistiques (baselines, modèle) ne
    sont, elles, apprises que sur la partie train. Le découpage est contigu :
    les derniers 10 % du texte (fin du tome V) servent de validation.
    """
    path = Path(data_dir) / CORPUS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancer d'abord `python -m nanogpt download`")
    text = path.read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    n_val = int(len(ids) * val_fraction)
    return tok, ids[:-n_val], ids[-n_val:]


def get_batch(ids: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator):
    """Tire batch_size fenêtres au hasard ; y est x décalé d'un caractère."""
    starts = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    offsets = np.arange(block_size + 1)
    windows = ids[starts[:, None] + offsets[None, :]]
    return windows[:, :-1], windows[:, 1:]
