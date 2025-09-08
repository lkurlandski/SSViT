"""
Split datasets into training/validation/testing with spatial and temporal constraints.
"""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt


__all__ = [
    "tr_vl_ts_split",
]


def tr_vl_ts_split(
    idx: npt.NDArray[np.integer],
    tr_size: float = 0.0,
    vl_size: float = 0.0,
    ts_size: float = 0.0,
    labels: Optional[npt.NDArray[np.integer]] = None,
    ratios: Optional[npt.NDArray[np.floating]] = None,
    timestamps: Optional[npt.NDArray[np.number]] = None,
    shuffle: bool = True,
    *,
    random_state: Optional[int] = None,
    temporal_mode: Literal["strict", "balanced"] = "strict",
    max_shift: int = 0,
    grid_step: int = 1,
    ratio_tolerance: Optional[float] = None,
    size_penalty: float = 0.05,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Split indices into train/val/test with optional stratification or temporal ordering.

    Args:
        idx: 1D array of indices to split.
        tr_size: Proportion of `idx` to assign to training set. Must be within [0, 1].
        vl_size: Proportion of `idx` to assign to validation set. Must be within [0, 1].
        ts_size: Proportion of `idx` to assign to test set. Must be within [0, 1].
        labels: Optional 1D array of class labels for stratified splitting.
        ratios: Optional 1D array of desired class ratios for stratified splitting.
        timestamps: Optional 1D array of timestamps for temporal splitting.
        shuffle: Whether to shuffle samples within each split.
        random_state: Optional random seed for reproducibility.
        temporal_mode: If `timestamps` is given, either `"strict"` or `"balanced"`.
        max_shift: For `temporal_mode="balanced"`, max boundary shift (in samples).
        grid_step: For `temporal_mode="balanced"`, step size when searching boundaries.
        ratio_tolerance: For `temporal_mode="balanced"`, optional max class-ratio error tolerance.
        size_penalty: For `temporal_mode="balanced"`, penalty factor for size deviation.

    Returns:
        tr: Indices assigned to training set.
        vl: Indices assigned to validation set.
        ts: Indices assigned to test set.
    """
    rng = np.random.default_rng(random_state)

    idx = _to_1d_array(np.asarray(idx), "idx")
    n = int(idx.size)

    if n == 0 or (tr_size == 0 and vl_size == 0 and ts_size == 0):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    tr_n, vl_n, ts_n, expected_rm = _largest_remainder_counts(n, (tr_size, vl_size, ts_size))

    # ---------------- Temporal path ----------------
    timestamps_aligned = _align_feature(idx, timestamps, n, "timestamps")
    if timestamps_aligned is not None:
        order = _time_order(timestamps_aligned)
        oidx = idx[order]

        if temporal_mode == "strict" or labels is None:
            tr, vl, ts = _temporal_strict_split(oidx, tr_n, vl_n, ts_n, shuffle, rng)
            _warn_if_extra_removed(n, expected_rm, tr, vl, ts, reason=None)
            return tr, vl, ts

        # balanced temporal (labels required)
        y = _align_feature(idx, labels, n, "labels")
        assert y is not None  # mypy

        tr, vl, ts, extra_reason = _temporal_balanced_split(
            oidx=oidx,
            order=order,
            y_all=y,
            ratios=ratios,
            tr_n=tr_n,
            vl_n=vl_n,
            ts_n=ts_n,
            n=n,
            max_shift=max_shift,
            grid_step=grid_step,
            ratio_tolerance=ratio_tolerance,
            size_penalty=size_penalty,
            shuffle=shuffle,
            rng=rng,
        )
        _warn_if_extra_removed(n, expected_rm, tr, vl, ts, reason=extra_reason)
        return tr, vl, ts

    # ---------------- Non-temporal path ----------------
    if labels is not None:
        y = _align_feature(idx, labels, n, "labels")
        assert y is not None
        tr, vl, ts = _stratified_split(idx, y, tr_n, vl_n, ts_n, ratios, shuffle, rng)
        # By design, we do not warn in non-temporal stratified mode if some
        # samples remain unassigned (e.g., sizes < 1.0 or class availability).
        return tr, vl, ts

    # Random (no labels, no timestamps)
    work = idx.copy()
    _shuffle_inplace(rng, work, enable=shuffle)
    tr = work[:tr_n]
    vl = work[tr_n : tr_n + vl_n]
    ts = work[tr_n + vl_n : tr_n + vl_n + ts_n]
    # No warning here; leftovers (if any) are implied by proportions.
    return tr, vl, ts


# --------------------------- Private helpers ----------------------------------


def _to_1d_array(a: npt.NDArray, name: str) -> npt.NDArray:
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {a.shape}.")
    return a


def _align_feature(
    idx: npt.NDArray[np.integer],
    feat: Optional[npt.NDArray],
    n: int,
    name: str,
) -> Optional[npt.NDArray]:
    if feat is None:
        return None
    arr = np.asarray(feat)
    arr = _to_1d_array(arr, name)
    if arr.shape[0] == n:
        return arr
    try:
        return arr[idx]
    except Exception as e:
        raise ValueError(
            f"{name} must either be length len(idx) or indexable by idx. "
            f"Got length {arr.shape[0]}; indexing failed: {e!r}"
        ) from e


def _largest_remainder_counts(n: int, sizes: Tuple[float, float, float]) -> tuple[int, int, int, int]:
    props = np.array(sizes, dtype=float)
    if np.any((props < 0) | (props > 1)):
        raise ValueError("Sizes must be within [0, 1].")
    total_prop = float(props.sum())
    if total_prop > 1 + 1e-12:
        raise ValueError(f"tr_size+vl_size+ts_size must be <= 1.0; got {total_prop:.6f}")

    raw = props * n
    base = np.floor(raw).astype(int)
    target_total = int(round(raw.sum()))
    give = target_total - int(base.sum())

    if give > 0:
        order = np.argsort(-(raw - base))
        base[order[:give]] += 1
    elif give < 0:
        order = np.argsort(raw - base)
        base[order[: -give]] -= 1

    tr_n, vl_n, ts_n = map(int, base)
    rm_n = n - (tr_n + vl_n + ts_n)
    return tr_n, vl_n, ts_n, rm_n


def _shuffle_inplace(rng: np.random.Generator, a: npt.NDArray, *, enable: bool) -> None:
    if enable and a.size > 1:
        rng.shuffle(a)


def _time_order(timestamps: npt.NDArray[np.number]) -> npt.NDArray[np.int64]:
    n = timestamps.shape[0]
    return np.lexsort((np.arange(n, dtype=np.int64), timestamps)).astype(np.int64)


def _empirical_ratios(y: npt.NDArray, classes: npt.NDArray) -> npt.NDArray[np.floating]:
    counts = np.array([(y == c).sum() for c in classes], dtype=float)
    s = float(counts.sum())
    if s <= 0:
        return np.ones_like(counts, dtype=float) / float(len(counts))
    return counts / s


def _build_pos_lists(y_sorted: npt.NDArray, classes: npt.NDArray) -> list[npt.NDArray[np.int64]]:
    out: list[npt.NDArray[np.int64]] = []
    for c in classes:
        out.append(np.flatnonzero(y_sorted == c).astype(np.int64))
    return out


def _segment_counts(a: int, b: int, pos_lists: Sequence[npt.NDArray[np.int64]]) -> npt.NDArray[np.int64]:
    if b <= a:
        return np.zeros(len(pos_lists), dtype=np.int64)
    counts = np.empty(len(pos_lists), dtype=np.int64)
    for i, pos in enumerate(pos_lists):
        lo = int(np.searchsorted(pos, a, side="left"))
        hi = int(np.searchsorted(pos, b, side="left"))
        counts[i] = hi - lo
    return counts


def _dist(counts: npt.NDArray[np.integer]) -> npt.NDArray[np.floating]:
    s = int(counts.sum())
    if s <= 0:
        return np.ones(counts.size, dtype=float) / float(counts.size)
    return counts.astype(float) / float(s)


def _prepare_desired_ratios(
    y_sorted: npt.NDArray[np.integer],
    classes: npt.NDArray[np.integer],
    ratios: Optional[npt.NDArray[np.floating]],
) -> npt.NDArray[np.floating]:
    if ratios is None:
        return _empirical_ratios(y_sorted, classes)
    r = np.asarray(ratios, dtype=float)
    if r.ndim != 1 or r.size != classes.size or (r < 0).any() or r.sum() <= 0:
        raise ValueError("ratios must be 1D, non-negative, and match number of classes.")
    return r / r.sum()


def _ratio_error(dists: Sequence[npt.NDArray[np.floating]], target: Optional[npt.NDArray[np.floating]]) -> float:
    if target is not None:
        return float(sum(np.abs(d - target).sum() for d in dists))
    err = 0.0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            err += float(np.abs(dists[i] - dists[j]).sum())
    return err


def _ratio_ok(dists: Sequence[npt.NDArray[np.floating]], target: Optional[npt.NDArray[np.floating]], tol: float) -> bool:
    if tol < 0:
        return False
    if target is not None:
        return all(float(np.max(np.abs(d - target))) <= tol for d in dists)
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            if float(np.max(np.abs(dists[i] - dists[j]))) > tol:
                return False
    return True


# --------------------------- Split strategies ---------------------------------


def _temporal_strict_split(
    oidx: npt.NDArray[np.integer],
    tr_n: int,
    vl_n: int,
    ts_n: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    tr = oidx[:tr_n]
    vl = oidx[tr_n : tr_n + vl_n]
    ts = oidx[tr_n + vl_n : tr_n + vl_n + ts_n]
    _shuffle_inplace(rng, tr, enable=shuffle)
    _shuffle_inplace(rng, vl, enable=shuffle)
    _shuffle_inplace(rng, ts, enable=shuffle)
    return tr, vl, ts


def _temporal_balanced_split(
    *,
    oidx: npt.NDArray[np.integer],
    order: npt.NDArray[np.int64],
    y_all: npt.NDArray[np.integer],
    ratios: Optional[npt.NDArray[np.floating]],
    tr_n: int,
    vl_n: int,
    ts_n: int,
    n: int,
    max_shift: int,
    grid_step: int,
    ratio_tolerance: Optional[float],
    size_penalty: float,
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer], Optional[str]]:
    """Balanced chronological split with local boundary search (+ optional trimming)."""
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0.")
    if grid_step <= 0:
        raise ValueError("grid_step must be >= 1.")

    y_sorted = y_all[order]
    classes = np.unique(y_sorted)
    desired = _prepare_desired_ratios(y_sorted, classes, ratios)

    pos_lists = _build_pos_lists(y_sorted, classes)

    p0 = tr_n
    q0 = tr_n + vl_n

    p_lo = max(0, p0 - max_shift)
    p_hi = min(n, p0 + max_shift)
    q_lo = max(p_lo, q0 - max_shift)
    q_hi = min(n, q0 + max_shift)

    best_any: dict[str, float] = dict(score=np.inf, p=p0, q=q0, len_ts=ts_n)
    best_tol: Optional[dict[str, float]] = None
    tol = ratio_tolerance

    # --- NEW: explicitly evaluate the strict candidate (p0, q0) ---
    len_tr = p0
    len_vl = max(0, q0 - p0)
    len_ts_strict = min(ts_n, max(0, n - q0))
    c_tr = _segment_counts(0, p0, pos_lists)
    c_vl = _segment_counts(p0, q0, pos_lists)
    c_ts = _segment_counts(q0, q0 + len_ts_strict, pos_lists)
    d_tr, d_vl, d_ts = _dist(c_tr), _dist(c_vl), _dist(c_ts)
    size_err = abs(len_tr - tr_n) + abs(len_vl - vl_n) + abs(len_ts_strict - ts_n)
    strict_score = _ratio_error([d_tr, d_vl, d_ts], desired if ratios is not None else None) + size_penalty * size_err
    best_any.update(score=strict_score, p=p0, q=q0, len_ts=len_ts_strict)
    if tol is not None and _ratio_ok([d_tr, d_vl, d_ts], desired if ratios is not None else None, tol):
        best_tol = dict(score=strict_score, p=p0, q=q0, len_ts=len_ts_strict)

    # Grid search around target boundaries
    for p in range(p_lo, p_hi + 1, grid_step):
        q_start = max(p, q_lo)
        for q in range(q_start, q_hi + 1, grid_step):
            len_tr = p
            len_vl = max(0, q - p)
            len_ts = min(ts_n, max(0, n - q))

            c_tr = _segment_counts(0, p, pos_lists)
            c_vl = _segment_counts(p, q, pos_lists)
            c_ts = _segment_counts(q, q + len_ts, pos_lists)

            d_tr, d_vl, d_ts = _dist(c_tr), _dist(c_vl), _dist(c_ts)
            dists = [d_tr, d_vl, d_ts]

            size_err = abs(len_tr - tr_n) + abs(len_vl - vl_n) + abs(len_ts - ts_n)
            score = _ratio_error(dists, desired if ratios is not None else None) + size_penalty * size_err

            ok = (tol is not None) and _ratio_ok(dists, desired if ratios is not None else None, tol)
            if ok:
                if (best_tol is None) or (float(best_tol["score"]) > score):
                    best_tol = dict(score=score, p=p, q=q, len_ts=len_ts)

            if float(best_any["score"]) > score:
                best_any.update(score=score, p=p, q=q, len_ts=len_ts)

    chosen = best_tol if best_tol is not None else best_any
    p = int(chosen["p"])
    q = int(chosen["q"])
    len_ts = int(chosen["len_ts"])

    # Minimal trimming fallback if tolerance requested but unmet
    if best_tol is None and ratio_tolerance is not None:
        tr, vl, ts, ok = _trim_to_tolerance(
            oidx=oidx,
            n=n,
            p=p,
            q=q,
            len_ts=len_ts,
            pos_lists=pos_lists,
            desired=desired if ratios is not None else None,
            ratios_given=(ratios is not None),
            tol=ratio_tolerance,
            max_shift=max_shift,
            grid_step=grid_step,
            shuffle=shuffle,
            rng=rng,
        )
        if ok:
            return tr, vl, ts, "temporal-balanced trimming to satisfy ratio_tolerance"

    # Construct final split (no trimming)
    tr = oidx[:p]
    vl = oidx[p:q]
    ts = oidx[q : q + len_ts]
    _shuffle_inplace(rng, tr, enable=shuffle)
    _shuffle_inplace(rng, vl, enable=shuffle)
    _shuffle_inplace(rng, ts, enable=shuffle)
    return tr, vl, ts, "temporal-balanced boundary deviation" if (q + len_ts) < n else None


def _trim_to_tolerance(
    *,
    oidx: npt.NDArray[np.integer],
    n: int,
    p: int,
    q: int,
    len_ts: int,
    pos_lists: Sequence[npt.NDArray[np.int64]],
    desired: Optional[npt.NDArray[np.floating]],
    ratios_given: bool,
    tol: float,
    max_shift: int,
    grid_step: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer], bool]:
    """Attempt minimal trimming from temporal boundaries to satisfy tolerance."""
    def seg_ok(a: int, b: int) -> bool:
        cnt = _segment_counts(a, b, pos_lists)
        d = _dist(cnt)
        return _ratio_ok([d], desired, tol) if ratios_given else _ratio_ok([d], None, tol)

    step = max(1, grid_step)

    # Train: can trim up to entire prefix of length p
    best_s: Optional[int] = None
    upper_s = p
    for s in range(0, upper_s + 1, step):
        if seg_ok(s, p):
            best_s = s
            break
    s = best_s if best_s is not None else 0

    # Val: can trim up to its entire length (q - p)
    best_t_v: Optional[int] = None
    upper_t_v = max(0, q - max(p, 0))
    for t_v in range(0, upper_t_v + 1, step):
        if seg_ok(p, q - t_v):
            best_t_v = t_v
            break
    t_v = best_t_v if best_t_v is not None else 0

    # Test: can trim up to its current length len_ts
    best_t_t: Optional[int] = None
    upper_t_t = len_ts
    for t_t in range(0, upper_t_t + 1, step):
        if seg_ok(q, q + (len_ts - t_t)):
            best_t_t = t_t
            break
    t_t = best_t_t if best_t_t is not None else 0

    # Validate
    d_tr = _dist(_segment_counts(s, p, pos_lists))
    d_vl = _dist(_segment_counts(p, q - t_v, pos_lists))
    d_ts = _dist(_segment_counts(q, q + (len_ts - t_t), pos_lists))
    all_ok = _ratio_ok([d_tr, d_vl, d_ts], desired if ratios_given else None, tol)
    if not all_ok:
        return (
            np.empty(0, dtype=oidx.dtype),
            np.empty(0, dtype=oidx.dtype),
            np.empty(0, dtype=oidx.dtype),
            False,
        )

    tr = oidx[s:p]
    vl = oidx[p : q - t_v]
    ts = oidx[q : q + (len_ts - t_t)]
    _shuffle_inplace(rng, tr, enable=shuffle)
    _shuffle_inplace(rng, vl, enable=shuffle)
    _shuffle_inplace(rng, ts, enable=shuffle)
    return tr, vl, ts, True


def _stratified_split(
    idx: npt.NDArray[np.integer],
    y: npt.NDArray[np.integer],
    tr_n: int,
    vl_n: int,
    ts_n: int,
    ratios: Optional[npt.NDArray[np.floating]],
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    classes = np.unique(y)
    desired = _prepare_desired_ratios(y, classes, ratios)

    work = idx.copy()
    if shuffle:
        perm = rng.permutation(work.size)
        work = work[perm]
        y = y[perm]

    pools: dict[int, npt.NDArray] = {int(c): work[y == c] for c in classes}
    if shuffle:
        for c in classes:
            _shuffle_inplace(rng, pools[int(c)], enable=True)

    def take(n_target: int) -> npt.NDArray:
        if n_target <= 0:
            return np.empty(0, dtype=work.dtype)
        avail = np.array([pools[int(c)].size for c in classes], dtype=int)
        raw = desired * n_target
        take_counts = np.minimum(np.floor(raw).astype(int), avail)

        left = n_target - int(take_counts.sum())
        rema = raw - np.floor(raw)
        while left > 0 and np.any(avail - take_counts > 0):
            candidates = np.where(avail - take_counts > 0)[0]
            best_i = int(candidates[np.argmax(rema[candidates])])
            take_counts[best_i] += 1
            left -= 1

        chosen: list[npt.NDArray] = []
        for c, k in zip(classes, take_counts.tolist()):
            if k > 0:
                c_int = int(c)
                chosen.append(pools[c_int][:k])
                pools[c_int] = pools[c_int][k:]
        if not chosen:
            return np.empty(0, dtype=work.dtype)
        out = np.concatenate(chosen, axis=0)
        _shuffle_inplace(rng, out, enable=shuffle)
        return out

    tr = take(tr_n)
    vl = take(vl_n)
    ts = take(ts_n)
    # Any remainder (sizes < 1.0 or class shortage) is intentionally dropped without warning.
    return tr, vl, ts


def _warn_if_extra_removed(
    n: int,
    expected_rm: int,
    tr: npt.NDArray[np.integer],
    vl: npt.NDArray[np.integer],
    ts: npt.NDArray[np.integer],
    reason: Optional[str],
) -> None:
    """
    Warn when:
      1) trimming or boundary-driven discards occurred (reason is not None), OR
      2) even without an explicit reason, more items were dropped than proportions imply.
    """
    actual_removed = n - (int(tr.size) + int(vl.size) + int(ts.size))
    extra = actual_removed - expected_rm
    if reason is not None:
        msg = (
            f"tr_vl_ts_split removed samples to meet constraints "
            f"(total removed = {actual_removed}, expected from proportions = {expected_rm})."
            f" Reason: {reason}."
        )
        warnings.warn(msg, UserWarning)
    elif extra > 0:
        warnings.warn(
            f"tr_vl_ts_split removed {extra} additional sample(s) beyond the requested proportions "
            f"(total removed = {actual_removed}).",
            UserWarning,
        )
