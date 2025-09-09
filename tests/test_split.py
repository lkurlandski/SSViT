"""
Tests.
"""

import numpy as np
import pytest
from typing import Any

from src.split import tr_vl_ts_split


# Extensively typing numpy arrays is a pain the ass.
# mypy: disable-error-code="type-arg,no-untyped-call"


# ---------------- helpers ----------------

def hamilton_counts(n: int, sizes: tuple[float, float, float]) -> tuple[int, int, int, int]:
    props = np.array(sizes, dtype=float)
    raw = n * props
    base = np.floor(raw).astype(int)
    target_total = int(round(raw.sum()))
    give = target_total - int(base.sum())
    if give > 0:
        order = np.argsort(-(raw - base))
        base[order[:give]] += 1
    elif give < 0:
        order = np.argsort(raw - base)
        base[order[: -give]] -= 1
    tr, vl, ts = map(int, base)
    rm = n - (tr + vl + ts)
    return tr, vl, ts, rm


def class_dist(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    s = counts.sum()
    return counts / s if s > 0 else np.ones_like(counts) / counts.size


def pairwise_max_abs_diff(dists: list[np.ndarray]) -> float:
    m = 0.0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            m = max(m, float(np.max(np.abs(dists[i] - dists[j]))))
    return m


def make_temporal_drift(n: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    timestamps = np.arange(n, dtype=np.int64)
    labels = np.empty(n, dtype=int)

    def fill(a, b, zeros_ratio):
        block = b - a
        k = int(round(5 * zeros_ratio))
        pattern = np.array([0] * k + [1] * (5 - k), dtype=int)
        labels[a:b] = np.tile(pattern, block // 5 + 1)[:block]

    fill(0, 400, 0.8)
    fill(400, 600, 0.5)
    fill(600, 1000, 0.2)
    return idx, timestamps, labels


# --------------- fixtures ----------------

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


# --------------- core invariants (Cartesian) ---------------

@pytest.mark.parametrize("sizes", [(0.6, 0.2, 0.2), (0.33, 0.33, 0.33), (0.6, 0.2, 0.1)])
@pytest.mark.parametrize("seed", [0, 7, 123])
@pytest.mark.parametrize("shuffle", [True, False])
def test_basic_invariants_random(sizes: tuple[float, float, float], seed: int, shuffle: bool) -> None:
    idx = np.arange(1000, dtype=np.int64)
    tr, vl, ts = tr_vl_ts_split(idx, *sizes, shuffle=shuffle, random_state=seed)
    all_sets = np.concatenate([tr, vl, ts])
    # disjoint
    assert len(np.intersect1d(tr, vl)) == 0
    assert len(np.intersect1d(tr, ts)) == 0
    assert len(np.intersect1d(vl, ts)) == 0
    # subset
    assert set(all_sets).issubset(set(idx))
    # lengths
    exp_tr, exp_vl, exp_ts, exp_rm = hamilton_counts(len(idx), sizes)
    assert (len(tr), len(vl), len(ts)) == (exp_tr, exp_vl, exp_ts)
    assert len(all_sets) == len(idx) - exp_rm


@pytest.mark.parametrize("sizes", [(0.5, 0.25, 0.25), (0.6, 0.2, 0.2)])
def test_shuffle_false_preserves_order_when_non_temporal(sizes: tuple[float, float, float]) -> None:
    idx = np.arange(20, dtype=np.int64)
    tr, vl, ts = tr_vl_ts_split(idx, *sizes, shuffle=False)
    tr_n, vl_n, ts_n, _ = hamilton_counts(len(idx), sizes)
    assert np.array_equal(tr, idx[:tr_n])
    assert np.array_equal(vl, idx[tr_n:tr_n + vl_n])
    assert np.array_equal(ts, idx[tr_n + vl_n:tr_n + vl_n + ts_n])


@pytest.mark.parametrize("sizes", [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.33, 0.33, 0.33)])
@pytest.mark.parametrize("seed", [0, 42, 999])
def test_determinism_same_seed_same_split(sizes: tuple[float, float, float], seed: int) -> None:
    idx = np.arange(200, dtype=np.int64)
    args: dict[str, Any] = dict(tr_size=sizes[0], vl_size=sizes[1], ts_size=sizes[2], shuffle=True, random_state=seed)
    out1 = tr_vl_ts_split(idx, **args)
    out2 = tr_vl_ts_split(idx, **args)
    assert all(np.array_equal(a, b) for a, b in zip(out1, out2))


@pytest.mark.parametrize("seed_a", [0, 1])
@pytest.mark.parametrize("seed_b", [2, 999])
def test_determinism_shuffle_false_ignores_seed(seed_a: int, seed_b: int) -> None:
    idx = np.arange(200, dtype=np.int64)
    a = tr_vl_ts_split(idx, 0.6, 0.2, 0.2, shuffle=False, random_state=seed_a)
    b = tr_vl_ts_split(idx, 0.6, 0.2, 0.2, shuffle=False, random_state=seed_b)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --------------- stratified (non-temporal) ---------------

@pytest.mark.parametrize("n", [6000, 8000])
@pytest.mark.parametrize("probs", [np.array([0.2, 0.3, 0.5])])
@pytest.mark.parametrize("sizes", [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25)])
@pytest.mark.parametrize("seed", [7, 11])
def test_stratified_matches_empirical_ratios_large_n(n: int, probs: np.ndarray, sizes: tuple[float, float, float], seed: int) -> None:
    classes = np.arange(len(probs))
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    labels = rng.choice(classes, size=n, p=probs)
    tr, vl, ts = tr_vl_ts_split(idx, *sizes, labels=labels, shuffle=True, random_state=seed)
    for part in (tr, vl, ts):
        d = class_dist(labels[part], classes)
        assert np.max(np.abs(d - probs)) < 0.02


@pytest.mark.parametrize("n", [4000, 6000])
@pytest.mark.parametrize("ratios", [np.array([0.5, 0.5])])   # feasible with balanced data
@pytest.mark.parametrize("sizes", [(0.5, 0.25, 0.25), (0.6, 0.2, 0.2)])
@pytest.mark.parametrize("seed", [5, 123])
def test_stratified_with_explicit_ratios_balanced_data(n: int, ratios: np.ndarray, sizes: tuple[float, float, float], seed: int) -> None:
    idx = np.arange(n)
    labels = np.tile(np.array([0, 1]), n // 2)  # globally 50/50
    tr, vl, ts = tr_vl_ts_split(idx, *sizes, labels=labels, ratios=ratios,
                                shuffle=True, random_state=seed)
    classes = np.array([0, 1])
    for part in (tr, vl, ts):
        d = class_dist(labels[part], classes)
        assert np.max(np.abs(d - ratios)) <= 0.02


# Optional: demonstrate graceful behavior with **infeasible** ratios
@pytest.mark.parametrize("ratios", [np.array([0.2, 0.8]), np.array([0.8, 0.2])])
def test_stratified_with_infeasible_ratios_no_crash(ratios: np.ndarray) -> None:
    n = 4000
    idx = np.arange(n)
    labels = np.tile(np.array([0, 1]), n // 2)  # 50/50 globally
    tr, vl, ts = tr_vl_ts_split(idx, 0.6, 0.2, 0.2, labels=labels, ratios=ratios,
                                shuffle=True, random_state=7)
    # Just sanity checks (can't enforce desired ratios strictly):
    assert len(tr) + len(vl) + len(ts) == n
    for part in (tr, vl, ts):
        d = class_dist(labels[part], np.array([0, 1]))
        assert np.all((0.0 <= d) & (d <= 1.0))


# --------------- temporal strict ---------------

@pytest.mark.parametrize("sizes", [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25)])
@pytest.mark.parametrize("seed", [0, 1])
def test_temporal_strict_enforces_order(sizes: tuple[float, float, float], seed: int) -> None:
    n = 500
    idx = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    timestamps = rng.standard_normal(n).cumsum()
    tr, vl, ts = tr_vl_ts_split(idx, *sizes, timestamps=timestamps,
                                temporal_mode="strict", shuffle=True, random_state=seed)
    assert timestamps[tr].max() <= timestamps[vl].min()
    assert timestamps[vl].max() <= timestamps[ts].min()


@pytest.mark.parametrize("sizes", [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.33, 0.33, 0.33)])
def test_temporal_strict_lengths(sizes: tuple[float, float, float]) -> None:
    idx = np.arange(1000, dtype=np.int64)
    timestamps = np.arange(1000)
    tr, vl, ts = tr_vl_ts_split(idx, *sizes, timestamps=timestamps, temporal_mode="strict",
                                shuffle=False)
    exp_tr, exp_vl, exp_ts, _ = hamilton_counts(len(idx), sizes)
    assert (len(tr), len(vl), len(ts)) == (exp_tr, exp_vl, exp_ts)
    assert np.all(tr < vl[0])
    assert np.all(vl < ts[0])


# --------------- temporal balanced ---------------

@pytest.mark.parametrize("max_shift", [25, 50])
@pytest.mark.parametrize("grid_step", [1, 2])
def test_temporal_balanced_improves_class_similarity(max_shift: int, grid_step: int) -> None:
    idx, timestamps, labels = make_temporal_drift(1000)
    sizes = (0.5, 0.2, 0.2)

    tr_s, vl_s, ts_s = tr_vl_ts_split(idx, *sizes, labels=labels, timestamps=timestamps,
                                      temporal_mode="strict", shuffle=False)
    classes = np.array([0, 1])
    dists_strict = [
        class_dist(labels[tr_s], classes),
        class_dist(labels[vl_s], classes),
        class_dist(labels[ts_s], classes),
    ]
    strict_pair_gap = pairwise_max_abs_diff(dists_strict)

    tr_b, vl_b, ts_b = tr_vl_ts_split(
        idx, *sizes, labels=labels, timestamps=timestamps,
        temporal_mode="balanced", max_shift=max_shift, grid_step=grid_step,
        shuffle=False, random_state=0
    )
    dists_bal = [
        class_dist(labels[tr_b], classes),
        class_dist(labels[vl_b], classes),
        class_dist(labels[ts_b], classes),
    ]
    bal_pair_gap = pairwise_max_abs_diff(dists_bal)
    assert bal_pair_gap <= strict_pair_gap + 1e-12


@pytest.mark.parametrize("tol", [0.08, 0.10])
@pytest.mark.parametrize("max_shift", [60, 80])
@pytest.mark.parametrize("grid_step", [1, 2])
def test_temporal_balanced_respects_tolerance_when_feasible(tol: float, max_shift: int, grid_step: int) -> None:
    idx, timestamps, labels = make_temporal_drift(1000)
    sizes = (0.5, 0.2, 0.2)
    ratios = np.array([0.5, 0.5])
    tr, vl, ts = tr_vl_ts_split(
        idx, *sizes, labels=labels, ratios=ratios, timestamps=timestamps,
        temporal_mode="balanced", max_shift=max_shift, grid_step=grid_step,
        ratio_tolerance=tol, shuffle=False
    )
    classes = np.array([0, 1])
    for part in (tr, vl, ts):
        d = class_dist(labels[part], classes)
        assert np.max(np.abs(d - ratios)) <= tol


# Force trimming by eliminating boundary movement (max_shift=0)
@pytest.mark.parametrize("tol", [0.02, 0.03])
@pytest.mark.parametrize("max_shift", [0, 5])
@pytest.mark.parametrize("grid_step", [1])
def test_temporal_balanced_emits_warning_when_trimming(tol: float, max_shift: int, grid_step: int) -> None:
    idx, timestamps, labels = make_temporal_drift(1000)
    sizes = (0.5, 0.2, 0.2)
    ratios = np.array([0.5, 0.5])
    with pytest.warns(UserWarning):
        _ = tr_vl_ts_split(
            idx, *sizes, labels=labels, ratios=ratios, timestamps=timestamps,
            temporal_mode="balanced", max_shift=max_shift, grid_step=grid_step,
            ratio_tolerance=tol, shuffle=False
        )


@pytest.mark.parametrize("size_penalty", [0.05, 0.1])
@pytest.mark.parametrize("max_shift", [80, 120])
def test_temporal_balanced_size_penalty_keeps_sizes_close(size_penalty: float, max_shift: int) -> None:
    idx, timestamps, labels = make_temporal_drift(1000)
    sizes = (0.5, 0.2, 0.2)
    tr, vl, ts = tr_vl_ts_split(
        idx, *sizes, labels=labels, timestamps=timestamps,
        temporal_mode="balanced", max_shift=max_shift, grid_step=1,
        ratio_tolerance=None, size_penalty=size_penalty, shuffle=False
    )
    exp_tr, exp_vl, exp_ts, exp_rm = hamilton_counts(len(idx), sizes)
    assert abs(len(tr) - exp_tr) <= 5
    assert abs(len(vl) - exp_vl) <= 5
    assert abs(len(ts) - exp_ts) <= 5
    assert (len(tr) + len(vl) + len(ts)) >= len(idx) - exp_rm - 5


# --------------- alignment ---------------

@pytest.mark.parametrize("subset", [300, 600, 900])
@pytest.mark.parametrize("seed", [99, 123])
def test_alignment_with_global_arrays(subset: int, seed: int) -> None:
    n = 1000
    rng = np.random.default_rng(seed)
    global_labels = rng.integers(0, 3, size=n)
    global_times = rng.integers(0, 10_000, size=n)
    idx = rng.choice(np.arange(n), size=subset, replace=False)
    rng.shuffle(idx)

    tr, vl, ts = tr_vl_ts_split(
        idx, 0.6, 0.2, 0.2, labels=global_labels, timestamps=global_times,
        temporal_mode="strict", shuffle=True, random_state=seed
    )

    assert set(tr).issubset(set(idx))
    assert set(vl).issubset(set(idx))
    assert set(ts).issubset(set(idx))
    if len(tr) and len(vl):
        assert global_times[tr].max() <= global_times[vl].min()
    if len(vl) and len(ts):
        assert global_times[vl].max() <= global_times[ts].min()


# --------------- rounding / leftovers ---------------

@pytest.mark.parametrize("sizes", [(0.6, 0.2, 0.1), (0.55, 0.25, 0.1)])
def test_leftovers_when_sum_lt_one(sizes: tuple[float, float, float]) -> None:
    idx = np.arange(100, dtype=np.int64)
    tr, vl, ts = tr_vl_ts_split(idx, *sizes, shuffle=False)
    tr_n, vl_n, ts_n, rm_n = hamilton_counts(len(idx), sizes)
    assert (len(tr), len(vl), len(ts)) == (tr_n, vl_n, ts_n)
    used = np.sort(np.concatenate([tr, vl, ts]))
    assert len(used) == len(idx) - rm_n
    assert np.array_equal(used, idx[: len(idx) - rm_n])


@pytest.mark.parametrize("n", [9, 10, 11])
def test_rounding_fairness_small_n(n: int) -> None:
    idx = np.arange(n, dtype=np.int64)
    sizes = (0.33, 0.33, 0.33)
    tr, vl, ts = tr_vl_ts_split(idx, *sizes, shuffle=True, random_state=1)
    lens = sorted([len(tr), len(vl), len(ts)])
    assert max(lens) - min(lens) <= 1


# --------------- validation / errors ---------------

@pytest.mark.parametrize("sizes", [(0.5, 0.5, 0.5), (0.8, 0.3, 0.1)])
def test_error_sizes_sum_gt_one(sizes: tuple[float, float, float]) -> None:
    idx = np.arange(10, dtype=np.int64)
    total = sum(sizes)
    if total > 1.0:
        with pytest.raises(ValueError):
            tr_vl_ts_split(idx, *sizes)


@pytest.mark.parametrize("k", [2, 4])
def test_error_bad_ratios_length(k: int) -> None:
    n = 100
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=n)
    bad = np.ones(k) / k
    with pytest.raises(ValueError):
        tr_vl_ts_split(idx, 0.6, 0.2, 0.2, labels=labels, ratios=bad)


@pytest.mark.parametrize("sizes", [(0.6, 0.2, 0.2), (0.33, 0.33, 0.33)])
def test_empty_idx_edge_case(sizes: tuple[float, float, float]) -> None:
    idx = np.array([], dtype=np.int64)
    tr, vl, ts = tr_vl_ts_split(idx, *sizes)
    assert len(tr) == len(vl) == len(ts) == 0
