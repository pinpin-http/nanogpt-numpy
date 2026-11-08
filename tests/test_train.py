import json

import numpy as np

from nanogpt.data import CORPUS_FILE
from nanogpt.train import TrainConfig, train


def tiny_config(tmp_path, **kw):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / CORPUS_FILE).write_text("Jean Valjean regarda la rue, puis Cosette.\n" * 40, encoding="utf-8")
    cfg = dict(
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        batch_size=4,
        max_iters=12,
        eval_interval=5,
        eval_batches=2,
        log_interval=100,
        data_dir=str(data),
        out=str(tmp_path / "ckpt" / "m"),
    )
    cfg.update(kw)
    return TrainConfig(**cfg)


def test_defauts_sans_budget_de_temps():
    cfg = TrainConfig()
    assert cfg.time_budget_min == 0
    assert cfg.max_iters == 5000
    assert cfg.eval_interval == 250


def test_arret_a_max_iters_et_reproductible(tmp_path):
    # Sans budget de temps, l'entraînement fait exactement max_iters pas, et
    # deux runs de même graine donnent les mêmes losses au pas près.
    runs = []
    for name in ("a", "b"):
        cfg = tiny_config(tmp_path, out=str(tmp_path / name / "m"))
        assert cfg.time_budget_min == 0
        res = train(cfg)
        assert res["meta"]["iters_done"] == cfg.max_iters
        hist = res["history"]
        assert hist["iter"] == list(range(cfg.max_iters))
        assert [e["iter"] for e in hist["eval"]] == [0, 5, 10, 12]
        saved = json.loads((tmp_path / name / "m_history.json").read_text())
        assert saved["iter"] == hist["iter"]
        runs.append(hist)
    np.testing.assert_array_equal(runs[0]["train_loss"], runs[1]["train_loss"])
    assert [e["val"] for e in runs[0]["eval"]] == [e["val"] for e in runs[1]["eval"]]
