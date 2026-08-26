from __future__ import annotations
import numpy as np

def probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15):
    """Convert per-frame play probabilities into merged (start, end) frame
    intervals, end exclusive.

    Boundary semantics:
    - A frame belongs to a run when `p >= threshold`.
    - Two consecutive runs are merged when the gap between them is strictly
      less than `merge_gap` (i.e. `gap < merge_gap`); a gap exactly equal to
      `merge_gap` is NOT merged.
    - A (possibly merged) run is kept when its length is `>= min_len`; a run
      whose length equals `min_len` exactly is kept.
    - A run that is still open when the input ends is closed at `len(p)`.

    This function is non-causal / offline-only by design: computing a run's
    end, and deciding whether to merge it with a later run, requires reading
    frames that occur after the run starts (and, for merging, frames from a
    subsequent run further in the future). It must only be used for offline
    post-processing over a complete probability sequence, never in a
    frame-by-frame online/causal pipeline.
    """
    mask = np.asarray(p) >= threshold
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(mask)))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]

def suppress_events(events, min_gap):
    """Greedy non-max suppression over stroke events.

    Each event is a dict with (at least) keys `frame: int` and
    `confidence: float`; events may carry extra keys, which are preserved
    unchanged (the returned dicts are the same objects as the input dicts,
    i.e. aliases, not copies).

    Boundary semantics:
    - Events are considered in descending-confidence order, tie-broken by
      ascending frame (`key=(-confidence, frame)`), so equal-confidence
      inputs produce an input-order-independent result.
    - A candidate is kept if it is at distance `>= min_gap` from every
      already-kept event; a distance exactly equal to `min_gap` counts as
      far enough, so both events are kept (not suppressed).
    - The result is returned sorted by `frame`, ascending.
    """
    kept = []
    for ev in sorted(events, key=lambda e: (-e["confidence"], e["frame"])):
        if all(abs(ev["frame"] - k["frame"]) >= min_gap for k in kept):
            kept.append(ev)
    return sorted(kept, key=lambda e: e["frame"])
