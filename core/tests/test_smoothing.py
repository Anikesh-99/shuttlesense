import copy
import numpy as np
from shuttlesense_core.smoothing import probs_to_intervals, suppress_events

def test_intervals_merge_and_minlen():
    p = np.zeros(200)
    p[10:60] = 0.9        # run 1
    p[70:130] = 0.9       # run 2, gap of 10 (< merge_gap) -> merged with run 1
    p[150:160] = 0.9      # run 3, len 10 (< min_len) -> dropped
    out = probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15)
    assert out == [(10, 130)]

def test_nms_keeps_best_and_orders():
    ev = [
        {"frame": 100, "confidence": 0.7},
        {"frame": 104, "confidence": 0.9},   # wins over 100
        {"frame": 130, "confidence": 0.6},
    ]
    out = suppress_events(ev, min_gap=8)
    assert [e["frame"] for e in out] == [104, 130]

def test_intervals_all_below_threshold_returns_empty():
    p = np.zeros(200)
    assert probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15) == []

def test_intervals_genuinely_empty_array_returns_empty_list():
    assert probs_to_intervals(np.array([])) == []

def test_intervals_run_touching_array_end_is_closed_at_len():
    # A run that never drops below threshold before the array ends must be
    # closed at len(p), not dropped or left open.
    p = np.zeros(100)
    p[70:100] = 0.9   # touches the very end, length 30 == min_len
    out = probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15)
    assert out == [(70, 100)]

def test_intervals_gap_exactly_at_merge_gap_boundary_not_merged():
    # merge condition is `s - prev_end < merge_gap`; a gap exactly equal to
    # merge_gap must NOT be merged (strict inequality).
    p = np.zeros(200)
    p[10:60] = 0.9
    p[75:130] = 0.9   # gap of exactly 15 == merge_gap -> not merged
    out = probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15)
    assert out == [(10, 60), (75, 130)]

def test_intervals_gap_one_less_than_merge_gap_is_merged():
    # gap of merge_gap - 1 must be merged (< merge_gap holds).
    p = np.zeros(200)
    p[10:60] = 0.9
    p[74:130] = 0.9   # gap of exactly 14 < merge_gap(15) -> merged
    out = probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15)
    assert out == [(10, 130)]

def test_intervals_min_len_exact_boundary_is_kept():
    # A run whose length equals min_len exactly must be kept (>= min_len).
    p = np.zeros(100)
    p[10:40] = 0.9   # length exactly 30 == min_len
    out = probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15)
    assert out == [(10, 40)]

def test_nms_empty_input_returns_empty_list():
    assert suppress_events([], min_gap=8) == []

def test_nms_drops_events_within_min_gap_and_preserves_extra_keys():
    ev = [
        {"frame": 100, "confidence": 0.5, "label": "smash"},
        {"frame": 105, "confidence": 0.5, "label": "clear"},  # tie in confidence
    ]
    ev_before = copy.deepcopy(ev)
    out = suppress_events(ev, min_gap=8)
    # Tie-break is (-confidence, frame), so of the two tied-confidence
    # events, the lower-frame one (100, "smash") wins deterministically.
    assert len(out) == 1
    assert out[0]["frame"] == 100
    assert out[0]["label"] == "smash"
    # The input list itself must be untouched (no mutation of the caller's
    # dicts), even though the kept event in `out` aliases the input dict.
    assert ev == ev_before

def test_nms_gap_exactly_at_min_gap_boundary_both_kept():
    # suppression condition requires `abs(diff) >= min_gap` to keep; a gap
    # exactly equal to min_gap means the candidate is NOT within min_gap
    # (distance is not < min_gap), so both should be kept.
    ev = [
        {"frame": 100, "confidence": 0.9},
        {"frame": 108, "confidence": 0.8},   # exactly min_gap away
    ]
    out = suppress_events(ev, min_gap=8)
    assert [e["frame"] for e in out] == [100, 108]

def test_nms_all_far_apart_all_kept_sorted_by_frame():
    ev = [
        {"frame": 300, "confidence": 0.1},
        {"frame": 100, "confidence": 0.99},
        {"frame": 200, "confidence": 0.5},
    ]
    out = suppress_events(ev, min_gap=8)
    assert [e["frame"] for e in out] == [100, 200, 300]
