# SSViT

## Setup

```
uv python install 3.13
uv venv -p 3.13 env
source env/bin/activate
uv pip install -r requirements.txt [-r requirements-dev.txt]
```

Note that pandas v2.3.3 has a bug that causes segmentation faults when reading some CSV files. For example:
```python
import pandas as pd
file = "./data/data/tr/size/size-0000572.csv"
pd.read_csv(file, engine='c')
```
will segmentation fault 100% of the time.

## Tests

```
OMP_NUM_THREADS=1 pytest -n auto --ignore tests/trainer tests/
OMP_NUM_THREADS=$(nproc --all) pytest -n 1 tests/trainer
```

## Lint

```
ruff check .
```

```
mypy .
```

## Format

```
isort --force-single-line-imports .
```
