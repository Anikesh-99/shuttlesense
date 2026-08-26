from __future__ import annotations
import numpy as np

def probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15):
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
    kept = []
    for ev in sorted(events, key=lambda e: -e["confidence"]):
        if all(abs(ev["frame"] - k["frame"]) >= min_gap for k in kept):
            kept.append(ev)
    return sorted(kept, key=lambda e: e["frame"])
