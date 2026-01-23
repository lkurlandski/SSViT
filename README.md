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
OMP_NUM_THREADS=1 pytest -n auto --ignore tests/test_trainer.py --ignore tests/test_architectures.py --ignore tests/test_patch_encoders.py tests/
OMP_NUM_THREADS=$(nproc --all) pytest tests/test_trainer.py tests/test_architectures.py tests/test_patch_encoders.py
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

## Usage

```
python scripts/create.py
```

```
bash run/experiment-0.sh
```

## Notes

Note that pandas v2.3.3 has a bug that causes segmentation faults when reading some CSV files. For example:
```python
import pandas as pd
file = "./data/data/tr/size/size-0000572.csv"
pd.read_csv(file, engine='c')
```
will segmentation fault 100% of the time.

Upon further analysis it seems the issue is more complex than this. Rather than just being from pandas, it looks like its caused by a combinations of various other third-party software in conjunction with pandas v2.3.3. Nonetheless, simply downgrading to panadas v2.2.2 solves the issue regardless.

## To Do

- Review the ConvViT (CvT) architecture, from ICCV '21.
- Review and adjust the overlap concept for the low-memory convolution.
- Improve the modularity of selecting structures.
- Improve the modularity of selecting guides.
- Improve the calculation for the total number of steps in the lr scheduler then remove the warnings in trainer.
- Add a real logging system with DEBUG, INFO, and WARN modes.
- There might be O(C^2) memory complexity in the low-mem implementations when computing winner positions.
- Resuming from a checkpoint saved at the end of one epoch but not before the start of the next epoch requires iterating through the entire epoch.
- Tests with pin_memory should only run if GPUs available.
