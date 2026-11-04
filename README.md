# nanogpt-numpy

Un petit GPT au niveau caractère écrit en NumPy pur : la passe avant et la
rétropropagation de chaque couche sont écrites à la main, sans framework de
deep learning.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```
