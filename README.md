# SSViT

## Setup

```
uv python install 3.13
uv venv -p 3.13 env
source env/bin/activate
uv pip install -r requirements.txt [-r requirements-dev.txt]
```

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
