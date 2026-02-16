OMP_NUM_THREADS=1 pytest -n 'auto' \
    --ignore tests/test_architectures.py \
    --ignore tests/test_patch_encoders.py \
    --ignore tests/test_main.py \
    --ignore tests/test_trainer.py \
    tests/

OMP_NUM_THREADS=$(nproc --all) pytest -n 0 \
    tests/test_architectures.py \
    tests/test_patch_encoders.py \
    tests/test_main.py \
    tests/test_trainer.py
