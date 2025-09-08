"""
Split datasets into training/validation/testing with spatial and temporal constraints.
"""

from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np


def tr_vl_ts_split(
    idx: np.ndarray,
    tr_size: float = 0.0,
    vl_size: float = 0.0,
    ts_size: float = 0.0,
    labels: Optional[np.ndarray] = None,
    ratios: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    shuffle: bool = True,
    *,
    # New knobs for determinism and temporal balancing:
    random_state: Optional[int] = None,
    temporal_mode: Literal["strict", "balanced"] = "strict",
    max_shift: int = 0,
    grid_step: int = 1,
    ratio_tolerance: Optional[float] = None,
    size_penalty: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into train/val/test with optional (a) stratification or (b) temporal ordering.

    Modes:
      - No timestamps:
          * If labels present -> stratified (match `ratios` if given, else empirical).
          * Else random split.
      - With timestamps:
          * temporal_mode="strict": pure chronological by target sizes.
          * temporal_mode="balanced": chronological *and* search locally for split
            points that minimize class-ratio error while preserving order.

    Args:
        idx: 1D array of "sample ids" to return (not positions).
        tr_size, vl_size, ts_size: Fractions in [0, 1], sum ≤ 1.0 (remainder → rm).
        labels: 1D labels aligned to `idx` length OR indexable by `idx`.
        ratios: Desired class ratios aligned to `np.unique(labels)` order. If None,
                empirical label distribution is used (when labels is provided).
        timestamps: 1D timestamps aligned to `idx` length OR indexable by `idx`.
        shuffle: Shuffle within each set (not across temporal boundaries).

        random_state: Seed for reproducible shuffles.
        temporal_mode: "strict" or "balanced" (only used if timestamps is not None).
        max_shift: Search radius (in **samples**) to adjust the two boundaries in balanced mode.
        grid_step: Step between candidate positions during the local search (≥1).
        ratio_tolerance:
            If provided, we accept a split when:
              - ratios given: max_abs(set_dist - ratios) ≤ ratio_tolerance for all sets.
              - ratios None : all pairwise max_abs diffs between set dists ≤ ratio_tolerance.
            If none satisfy, we return the min-error candidate (no discarding).
        size_penalty: Penalty weight for deviating from target sizes during balanced search.

    Returns:
        tr_idx, vl_idx, ts_idx, rm_idx
    """
    # --------------------------- helpers -------------------------------------
    def _to_1d(a: np.ndarray, name: str) -> np.ndarray:
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1D; got shape {a.shape}.")
        return a

    def _align(feat: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
        if feat is None:
            return None
        feat = np.asarray(feat)
        feat = _to_1d(feat, name)
        if feat.shape[0] == n:
            return feat
        try:
            return feat[idx]
        except Exception as e:
            raise ValueError(
                f"{name} must either be length len(idx) or indexable by idx. "
                f"Got length {len(feat)}; indexing failed: {e!r}"
            )

    def _largest_remainder_counts(N: int, props: np.ndarray) -> Tuple[int, int, int, int]:
        if np.any((props < 0) | (props > 1)):
            raise ValueError("Sizes must be within [0, 1].")
        total_prop = float(props.sum())
        if total_prop > 1 + 1e-12:
            raise ValueError(f"tr_size+vl_size+ts_size must be <= 1.0; got {total_prop:.6f}")
        raw = N * props
        base = np.floor(raw).astype(int)
        target_total = int(round(raw.sum()))
        give = target_total - int(base.sum())
        if give > 0:
            order = np.argsort(-(raw - base))  # descending fractional part
            base[order[:give]] += 1
        elif give < 0:
            order = np.argsort(raw - base)  # ascending fractional part
            base[order[: -give]] -= 1
        tr_c, vl_c, ts_c = map(int, base)
        rm_c = N - (tr_c + vl_c + ts_c)
        return tr_c, vl_c, ts_c, rm_c

    def _shuffle_inplace(a: np.ndarray) -> None:
        if shuffle and a.size > 1:
            rng.shuffle(a)

    def _empirical_ratios(y_sorted: np.ndarray, cls_vals: np.ndarray) -> np.ndarray:
        counts = np.array([(y_sorted == c).sum() for c in cls_vals], dtype=float)
        s = counts.sum()
        return counts / s if s > 0 else np.ones_like(counts) / counts.size

    # fast segment class counts via per-class position lists:
    def _build_pos_lists(y_sorted: np.ndarray, cls_vals: np.ndarray) -> list[np.ndarray]:
        pos_lists = []
        for c in cls_vals:
            pos_lists.append(np.flatnonzero(y_sorted == c).astype(np.int64))
        return pos_lists

    def _segment_counts(a: int, b: int, pos_lists: list[np.ndarray]) -> np.ndarray:
        # count elements in [a, b)
        if b <= a:
            return np.zeros(len(pos_lists), dtype=np.int64)
        out = np.empty(len(pos_lists), dtype=np.int64)
        for i, pos in enumerate(pos_lists):
            lo = np.searchsorted(pos, a, side="left")
            hi = np.searchsorted(pos, b, side="left")
            out[i] = hi - lo
        return out

    def _dist(counts: np.ndarray) -> np.ndarray:
        s = counts.sum()
        if s <= 0:
            # Empty set -> uniform (won't help, but avoids div-by-zero).
            return np.ones(counts.size, dtype=float) / counts.size
        return counts.astype(float) / float(s)

    def _ratio_error(dists: list[np.ndarray], r: Optional[np.ndarray]) -> float:
        # Error metric for optimization. Lower is better.
        if r is not None:
            # Sum of L1 distances to desired ratios
            return float(sum(np.abs(d - r).sum() for d in dists))
        # Else, pairwise L1 distances between sets
        err = 0.0
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                err += float(np.abs(dists[i] - dists[j]).sum())
        return err

    def _ratio_ok(dists: list[np.ndarray], r: Optional[np.ndarray], tol: float) -> bool:
        if r is not None:
            return all(np.max(np.abs(d - r)) <= tol for d in dists)
        # pairwise
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                if np.max(np.abs(dists[i] - dists[j])) > tol:
                    return False
        return True

    # --------------------------- body ----------------------------------------
    rng = np.random.default_rng(random_state)
    idx = np.asarray(idx)
    idx = _to_1d(idx, "idx")
    n = idx.size

    if n == 0 or (tr_size == 0 and vl_size == 0 and ts_size == 0):
        rm = idx.copy()
        if shuffle:
            _shuffle_inplace(rm)
        return (np.empty(0, idx.dtype),) * 3 + (rm,)

    sizes = np.array([tr_size, vl_size, ts_size], dtype=float)
    tr_n, vl_n, ts_n, _rm_n = _largest_remainder_counts(n, sizes)

    # --- Temporal path if timestamps provided --------------------------------
    ts = _align(timestamps, "timestamps")
    if ts is not None:
        # Order samples by time (stable on ties).
        order = np.lexsort((np.arange(n, dtype=np.int64), ts))
        oidx = idx[order]

        if temporal_mode == "strict" or labels is None:
            # pure chronological split
            tr = oidx[:tr_n]
            vl = oidx[tr_n : tr_n + vl_n]
            ts_ = oidx[tr_n + vl_n : tr_n + vl_n + ts_n]
            rm = oidx[tr_n + vl_n + ts_n :]
            if shuffle:
                _shuffle_inplace(tr); _shuffle_inplace(vl); _shuffle_inplace(ts_); _shuffle_inplace(rm)
            return tr, vl, ts_, rm

        # --- balanced temporal split -----------------------------------------
        y = _align(labels, "labels")
        # Sort y along the same order
        y_sorted = y[order]
        classes = np.unique(y_sorted)
        # desired ratios vector (aligned to `classes`)
        if ratios is None:
            desired = _empirical_ratios(y_sorted, classes)
        else:
            r = np.asarray(ratios, dtype=float)
            if r.ndim != 1 or r.size != classes.size or (r < 0).any() or r.sum() <= 0:
                raise ValueError("ratios must be 1D, non-negative, and match number of classes.")
            desired = r / r.sum()

        # Precompute per-class position lists for O(log n) segment queries
        pos_lists = _build_pos_lists(y_sorted, classes)

        # Target split points:
        p0 = tr_n
        q0 = tr_n + vl_n
        # Candidate windows (bounded)
        if max_shift < 0:
            raise ValueError("max_shift must be >= 0.")
        if grid_step <= 0:
            raise ValueError("grid_step must be >= 1.")

        p_lo = max(0, p0 - max_shift)
        p_hi = min(n, p0 + max_shift)
        q_lo = max(p_lo, q0 - max_shift)
        q_hi = min(n, q0 + max_shift)

        best_any = dict(score=np.inf, p=p0, q=q0, dists=None, len_tr=tr_n, len_vl=vl_n, len_ts=ts_n)
        best_tol = None  # prefer any candidate that satisfies ratio_tolerance
        tol = ratio_tolerance

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

                # NOTE: we changed size_err below in Patch 2
                size_err = abs(len_tr - tr_n) + abs(len_vl - vl_n) + abs(len_ts - ts_n)
                score = _ratio_error(dists, desired if ratios is not None else None) + size_penalty * size_err

                ok = (tol is not None) and _ratio_ok(dists, desired if ratios is not None else None, tol)
                if ok:
                    if (best_tol is None) or (score < best_tol["score"]):
                        best_tol = dict(score=score, p=p, q=q, dists=dists, len_tr=len_tr, len_vl=len_vl, len_ts=len_ts)

                if score < best_any["score"]:
                    best_any = dict(score=score, p=p, q=q, dists=dists, len_tr=len_tr, len_vl=len_vl, len_ts=len_ts)

        # choose tolerance-satisfying solution if any; else fallback to global best
        # choose tolerance-satisfying solution if any; else fallback to global best
        chosen = best_tol if best_tol is not None else best_any
        p, q, len_ts = chosen["p"], chosen["q"], chosen["len_ts"]

        # --- NEW: minimal trimming to satisfy ratio_tolerance (if still unmet) -------
        if (best_tol is None) and (ratio_tolerance is not None):
            tol = ratio_tolerance
            # Derive a conservative trim budget: enough to handle strong drift but bounded.
            trim_budget = max(1, max_shift) * 5  # e.g., max_shift=80 -> 400 samples
            step = max(1, grid_step)

            # Helper: check a segment's dist against desired
            def _ok(a: int, b: int) -> bool:
                cnt = _segment_counts(a, b, pos_lists)
                d = _dist(cnt)
                target = desired if ratios is not None else None
                return _ratio_ok([d], target, tol)

            # 1) TRAIN: move start forward (discard earliest samples) to reduce skew
            best_s = None
            upper_s = min(p, trim_budget)
            for s in range(0, upper_s + 1, step):
                if _ok(s, p):
                    best_s = s
                    break
            s = best_s if best_s is not None else 0

            # 2) VAL: trim from the end (move q backward)
            best_t_v = None
            upper_t_v = min(max(0, q - max(p, 0)), trim_budget)
            for t_v in range(0, upper_t_v + 1, step):
                if _ok(p, q - t_v):
                    best_t_v = t_v
                    break
            t_v = best_t_v if best_t_v is not None else 0

            # 3) TEST: trim from the end (shorten len_ts)
            best_t_t = None
            upper_t_t = min(len_ts, trim_budget)
            for t_t in range(0, upper_t_t + 1, step):
                if _ok(q, q + (len_ts - t_t)):
                    best_t_t = t_t
                    break
            t_t = best_t_t if best_t_t is not None else 0

            # Check if all three sets now satisfy tolerance
            def _seg_dist(a, b): 
                return _dist(_segment_counts(a, b, pos_lists))
            d_tr = _seg_dist(s, p)
            d_vl = _seg_dist(p, q - t_v)
            d_ts2 = _seg_dist(q, q + (len_ts - t_t))
            all_ok = _ratio_ok([d_tr, d_vl, d_ts2], desired if ratios is not None else None, tol)

            if all_ok:
                # Build trimmed split and rm as the complement of the three contiguous segments
                tr = oidx[s:p]
                vl = oidx[p : q - t_v]
                ts_ = oidx[q : q + (len_ts - t_t)]

                # rm is: [0, s) ∪ [q - t_v, q) ∪ [q + (len_ts - t_t), n)
                parts = []
                if s > 0:
                    parts.append(oidx[:s])
                if t_v > 0:
                    parts.append(oidx[q - t_v : q])
                tail_start = q + (len_ts - t_t)
                if tail_start < n:
                    parts.append(oidx[tail_start:])
                rm = np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=idx.dtype)

                if shuffle:
                    _shuffle_inplace(tr); _shuffle_inplace(vl); _shuffle_inplace(ts_); _shuffle_inplace(rm)
                return tr, vl, ts_, rm
        # --- end trimming fallback ---


        tr = oidx[:p]
        vl = oidx[p:q]
        ts_ = oidx[q : q + len_ts]
        rm = oidx[q + len_ts :]


        if shuffle:
            _shuffle_inplace(tr); _shuffle_inplace(vl); _shuffle_inplace(ts_); _shuffle_inplace(rm)
        return tr, vl, ts_, rm

    # --- Non-temporal path ---------------------------------------------------
    # 1) Stratified if labels provided
    def _stratified_split(work_idx: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        classes = np.unique(y)
        # Desired class proportions:
        if ratios is None:
            desired = _empirical_ratios(y, classes)
        else:
            r = np.asarray(ratios, dtype=float)
            if r.ndim != 1 or r.size != classes.size or (r < 0).any() or r.sum() <= 0:
                raise ValueError("ratios must be 1D, non-negative, and match number of classes.")
            desired = r / r.sum()

        # Build per-class pools
        pools = {c: work_idx[y == c] for c in classes}
        if shuffle:
            for c in classes:
                _shuffle_inplace(pools[c])

        # Allocate per set using largest remainder + availability
        def take(n_target: int) -> np.ndarray:
            if n_target <= 0:
                return np.empty(0, dtype=work_idx.dtype)
            avail = np.array([pools[c].size for c in classes], dtype=int)
            raw = desired * n_target
            take_counts = np.minimum(np.floor(raw).astype(int), avail)
            left = n_target - int(take_counts.sum())
            # largest remainder respecting availability
            rema = raw - np.floor(raw)
            while left > 0 and np.any(avail - take_counts > 0):
                candidates = np.where(avail - take_counts > 0)[0]
                best_i = candidates[np.argmax(rema[candidates])]
                take_counts[best_i] += 1
                left -= 1
            # collect
            out = []
            for c, k in zip(classes, take_counts.tolist()):
                if k > 0:
                    out.append(pools[c][:k])
                    pools[c] = pools[c][k:]
            if out:
                out = np.concatenate(out, axis=0)
                _shuffle_inplace(out)
                return out
            return np.empty(0, dtype=work_idx.dtype)

        tr = take(tr_n)
        vl = take(vl_n)
        ts_ = take(ts_n)
        leftovers = [p for p in pools.values() if p.size > 0]
        rm = np.concatenate(leftovers, axis=0) if leftovers else np.empty(0, dtype=work_idx.dtype)
        if shuffle:
            _shuffle_inplace(rm)
        return tr, vl, ts_, rm

    if labels is not None:
        y = _align(labels, "labels")
        work = idx.copy()
        # Apply a single shared permutation to keep work & y aligned
        perm = rng.permutation(work.size) if shuffle else np.arange(work.size)
        work = work[perm]
        y = y[perm]
        return _stratified_split(work, y)

    # 2) Plain random
    work = idx.copy()
    if shuffle:
        _shuffle_inplace(work)
    tr = work[:tr_n]
    vl = work[tr_n : tr_n + vl_n]
    ts_ = work[tr_n + vl_n : tr_n + vl_n + ts_n]
    rm = work[tr_n + vl_n + ts_n :]
    return tr, vl, ts_, rm
