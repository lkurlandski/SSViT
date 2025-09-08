"""
Tests.
"""

import numpy as np
import pytest

from src.split import tr_vl_ts_split


# --------------------------- helpers -----------------------------------------

def hamilton_counts(n: int, sizes: tuple[float, float, float]) -> tuple[int, int, int, int]:
    """Reference rounding: Hamilton (largest remainder)."""
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


# --------------------------- fixtures ----------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def make_idx():
    def _make_idx(n=1000):
        return np.arange(n, dtype=np.int64)
    return _make_idx


# --------------------------- core invariants ---------------------------------

@pytest.mark.parametrize("sizes", [(0.6, 0.2, 0.2), (0.33, 0.33, 0.33), (0.6, 0.2, 0.1)])
def test_basic_invariants_random(make_idx, sizes):
    idx = make_idx(1000)
    tr, vl, ts, rm = tr_vl_ts_split(idx, *sizes, shuffle=True, random_state=0)
    # Disjointness & coverage
    all_sets = np.concatenate([tr, vl, ts, rm])
    assert np.array_equal(np.sort(all_sets), np.sort(idx))
    assert len(np.intersect1d(tr, vl)) == 0
    assert len(np.intersect1d(tr, ts)) == 0
    assert len(np.intersect1d(vl, ts)) == 0
    # Length expectations
    exp_tr, exp_vl, exp_ts, exp_rm = hamilton_counts(len(idx), sizes)
    assert (len(tr), len(vl), len(ts), len(rm)) == (exp_tr, exp_vl, exp_ts, exp_rm)


def test_shuffle_false_preserves_order_when_non_temporal(make_idx):
    idx = make_idx(20)
    tr, vl, ts, rm = tr_vl_ts_split(idx, 0.5, 0.25, 0.25, shuffle=False)
    assert np.array_equal(tr, idx[:10])
    assert np.array_equal(vl, idx[10:15])
    assert np.array_equal(ts, idx[15:20])
    assert len(rm) == 0


def test_determinism_same_seed_same_split(make_idx):
    idx = make_idx(200)
    args = dict(tr_size=0.6, vl_size=0.2, ts_size=0.2, shuffle=True, random_state=42)
    out1 = tr_vl_ts_split(idx, **args)
    out2 = tr_vl_ts_split(idx, **args)
    assert all(np.array_equal(a, b) for a, b in zip(out1, out2))


def test_determinism_shuffle_false_ignores_seed(make_idx):
    idx = make_idx(200)
    a = tr_vl_ts_split(idx, 0.6, 0.2, 0.2, shuffle=False, random_state=0)
    b = tr_vl_ts_split(idx, 0.6, 0.2, 0.2, shuffle=False, random_state=999)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --------------------------- stratified (non-temporal) ------------------------

def test_stratified_matches_empirical_ratios_large_n(rng):
    n = 10000
    idx = np.arange(n)
    # Generate labels with known empirical ratios: 3 classes, probs 0.2, 0.3, 0.5
    classes = np.array([0, 1, 2])
    probs = np.array([0.2, 0.3, 0.5])
    labels = rng.choice(classes, size=n, p=probs)
    tr, vl, ts, rm = tr_vl_ts_split(idx, 0.6, 0.2, 0.2, labels=labels, shuffle=True, random_state=7)
    for part in (tr, vl, ts):
        d = class_dist(labels[part], classes)
        # Expect each within ~1.5 percentage points of empirical with large n
        assert np.max(np.abs(d - probs)) < 0.015
    assert len(rm) == 0


def test_stratified_with_explicit_ratios_balanced_data(rng):
    n = 4000
    idx = np.arange(n)
    classes = np.array([0, 1])
    # Balanced dataset overall
    labels = np.tile(np.array([0, 1]), n // 2)
    # Ask for 50/50 in each set (feasible)
    ratios = np.array([0.5, 0.5])
    tr, vl, ts, rm = tr_vl_ts_split(idx, 0.5, 0.25, 0.25, labels=labels, ratios=ratios,
                                    shuffle=True, random_state=123)
    for part in (tr, vl, ts):
        d = class_dist(labels[part], classes)
        assert np.max(np.abs(d - ratios)) <= 0.02  # tight tolerance
    assert len(rm) == 0


# --------------------------- temporal strict ----------------------------------

def test_temporal_strict_enforces_order(rng):
    n = 500
    idx = np.arange(n)
    timestamps = rng.standard_normal(n).cumsum()  # strictly increasing w.h.p.
    tr, vl, ts, rm = tr_vl_ts_split(idx, 0.6, 0.2, 0.2, timestamps=timestamps,
                                    temporal_mode="strict", shuffle=True, random_state=0)
    assert timestamps[tr].max() <= timestamps[vl].min()
    assert timestamps[vl].max() <= timestamps[ts].min()
    assert len(rm) == 0


def test_temporal_strict_lengths(make_idx, rng):
    idx = make_idx(1000)
    timestamps = np.arange(1000)  # clean monotone
    sizes = (0.6, 0.2, 0.2)
    tr, vl, ts, rm = tr_vl_ts_split(idx, *sizes, timestamps=timestamps, temporal_mode="strict",
                                    shuffle=False)
    exp_tr, exp_vl, exp_ts, exp_rm = hamilton_counts(len(idx), sizes)
    assert (len(tr), len(vl), len(ts), len(rm)) == (exp_tr, exp_vl, exp_ts, exp_rm)
    # Order respected
    assert np.all(tr < vl[0])
    assert np.all(vl < ts[0])


# --------------------------- temporal balanced --------------------------------

def _make_temporal_drift_data(n=1000):
    """
    Build labels that drift over time:
      0..399 : 80% class 0
      400..599: ~50/50
      600..999: 80% class 1
    Deterministic pattern without RNG.
    """
    idx = np.arange(n)
    timestamps = np.arange(n, dtype=np.int64)
    labels = np.empty(n, dtype=int)

    def fill_block(a, b, zeros_ratio):
        block = b - a
        k = int(round(5 * zeros_ratio))
        pattern = np.array([0] * k + [1] * (5 - k), dtype=int)  # length 5
        labels[a:b] = np.tile(pattern, block // 5 + 1)[:block]

    fill_block(0, 400, 0.8)     # mostly zeros
    fill_block(400, 600, 0.5)   # balanced
    fill_block(600, 1000, 0.2)  # mostly ones
    return idx, timestamps, labels


def test_temporal_balanced_improves_class_similarity():
    idx, timestamps, labels = _make_temporal_drift_data(1000)
    sizes = (0.5, 0.2, 0.2)  # 500 / 200 / 200
    # Strict
    tr_s, vl_s, ts_s, _ = tr_vl_ts_split(idx, *sizes, labels=labels, timestamps=timestamps,
                                         temporal_mode="strict", shuffle=False)
    classes = np.array([0, 1])
    dists_strict = [
        class_dist(labels[tr_s], classes),
        class_dist(labels[vl_s], classes),
        class_dist(labels[ts_s], classes),
    ]
    strict_pair_gap = pairwise_max_abs_diff(dists_strict)

    # Balanced (search within ±50)
    tr_b, vl_b, ts_b, _ = tr_vl_ts_split(
        idx, *sizes, labels=labels, timestamps=timestamps,
        temporal_mode="balanced", max_shift=50, grid_step=1, shuffle=False, random_state=0
    )
    dists_bal = [
        class_dist(labels[tr_b], classes),
        class_dist(labels[vl_b], classes),
        class_dist(labels[ts_b], classes),
    ]
    bal_pair_gap = pairwise_max_abs_diff(dists_bal)

    # Expect improvement or equal (never worse ideally)
    assert bal_pair_gap <= strict_pair_gap + 1e-12


def test_temporal_balanced_respects_tolerance_when_feasible():
    idx, timestamps, labels = _make_temporal_drift_data(1000)
    sizes = (0.5, 0.2, 0.2)
    # The global distribution is ~50/50; with enough shift, each set can be close to 50/50.
    ratios = np.array([0.5, 0.5])
    tr, vl, ts, _ = tr_vl_ts_split(
        idx, *sizes, labels=labels, ratios=ratios, timestamps=timestamps,
        temporal_mode="balanced", max_shift=80, grid_step=2, ratio_tolerance=0.08, shuffle=False
    )
    classes = np.array([0, 1])
    for part in (tr, vl, ts):
        d = class_dist(labels[part], classes)
        assert np.max(np.abs(d - ratios)) <= 0.08


def test_temporal_balanced_size_penalty_keeps_sizes_close():
    idx, timestamps, labels = _make_temporal_drift_data(1000)
    sizes = (0.5, 0.2, 0.2)
    tr, vl, ts, rm = tr_vl_ts_split(
        idx, *sizes, labels=labels, timestamps=timestamps,
        temporal_mode="balanced", max_shift=120, grid_step=1,
        ratio_tolerance=None, size_penalty=0.1, shuffle=False
    )
    exp_tr, exp_vl, exp_ts, exp_rm = hamilton_counts(len(idx), sizes)
    # Allow small deviation due to search/penalty trade-off
    assert abs(len(tr) - exp_tr) <= 5
    assert abs(len(vl) - exp_vl) <= 5
    assert abs(len(ts) - exp_ts) <= 5
    # Coverage still exact
    assert len(tr) + len(vl) + len(ts) + len(rm) == len(idx)


# --------------------------- alignment behavior -------------------------------

def test_alignment_with_global_arrays(rng):
    n = 1000
    global_labels = rng.integers(0, 3, size=n)  # 3 classes
    global_times = rng.integers(0, 10_000, size=n)

    # Take a subset of indices in arbitrary order
    idx = rng.choice(np.arange(n), size=600, replace=False)
    rng.shuffle(idx)

    tr, vl, ts, rm = tr_vl_ts_split(
        idx, 0.6, 0.2, 0.2, labels=global_labels, timestamps=global_times,
        temporal_mode="strict", shuffle=True, random_state=99
    )

    # Ensure outputs are from idx and respect time ordering
    assert set(tr).issubset(set(idx))
    assert set(vl).issubset(set(idx))
    assert set(ts).issubset(set(idx))
    # temporal ordering w.r.t subset times
    tmap = global_times  # same array; function indexes it internally
    if len(tr) and len(vl):
        assert tmap[tr].max() <= tmap[vl].min()
    if len(vl) and len(ts):
        assert tmap[vl].max() <= tmap[ts].min()
    assert len(rm) == 0  # sizes sum to 1 → no leftovers


# --------------------------- rounding / leftovers -----------------------------

def test_leftovers_when_sum_lt_one(make_idx):
    idx = make_idx(100)
    sizes = (0.6, 0.2, 0.1)  # sum = 0.9
    tr, vl, ts, rm = tr_vl_ts_split(idx, *sizes, shuffle=False)
    assert (len(tr), len(vl), len(ts), len(rm)) == (60, 20, 10, 10)
    # Leftovers are the tail in order because shuffle=False
    assert np.array_equal(rm, idx[90:])


def test_rounding_fairness_approx(make_idx):
    idx = make_idx(10)
    sizes = (0.33, 0.33, 0.33)  # sums to ~9.9 -> 10 after rounding
    tr, vl, ts, rm = tr_vl_ts_split(idx, *sizes, shuffle=True, random_state=1)
    lens = sorted([len(tr), len(vl), len(ts)])
    assert lens == [3, 3, 4]
    assert len(rm) == 0


# --------------------------- validation / errors ------------------------------

def test_error_sizes_sum_gt_one(make_idx):
    idx = make_idx(10)
    with pytest.raises(ValueError):
        tr_vl_ts_split(idx, 0.5, 0.5, 0.5)  # sum > 1


def test_error_bad_ratios_length(rng):
    n = 100
    idx = np.arange(n)
    labels = rng.integers(0, 3, size=n)
    with pytest.raises(ValueError):
        tr_vl_ts_split(idx, 0.6, 0.2, 0.2, labels=labels, ratios=np.array([0.5, 0.5]))


def test_error_negative_or_zero_grid_params(make_idx):
    idx = make_idx(100)
    timestamps = np.arange(100)
    labels = np.zeros(100, dtype=int)
    with pytest.raises(ValueError):
        tr_vl_ts_split(idx, 0.6, 0.2, 0.2, labels=labels, timestamps=timestamps,
                       temporal_mode="balanced", max_shift=-1)
    with pytest.raises(ValueError):
        tr_vl_ts_split(idx, 0.6, 0.2, 0.2, labels=labels, timestamps=timestamps,
                       temporal_mode="balanced", grid_step=0)


def test_empty_idx_edge_case():
    idx = np.array([], dtype=np.int64)
    tr, vl, ts, rm = tr_vl_ts_split(idx, 0.6, 0.2, 0.2)
    assert len(tr) == len(vl) == len(ts) == 0
    assert np.array_equal(rm, idx)
