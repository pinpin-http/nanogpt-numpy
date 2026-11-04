# Limite NumPy/OpenBLAS à 2 threads par défaut (make train THREADS=8 pour changer).
THREADS ?= 2
export OMP_NUM_THREADS := $(THREADS)
export OPENBLAS_NUM_THREADS := $(THREADS)
export MKL_NUM_THREADS := $(THREADS)

PY ?= .venv/bin/python
CKPT ?= checkpoints/miserables
SAMPLE_CKPT ?= models/miserables

.PHONY: install test data baselines train report sample all

install:
	test -d .venv || uv venv --python 3.12
	uv pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

data:
	$(PY) -m nanogpt download

baselines:
	$(PY) -m nanogpt baselines

train:
	$(PY) -m nanogpt train --out $(CKPT)

report:
	$(PY) -m nanogpt report --ckpt $(CKPT)

sample:
	$(PY) -m nanogpt sample --ckpt $(SAMPLE_CKPT) --prompt "Jean Valjean" --num-samples 3

all: data train report
