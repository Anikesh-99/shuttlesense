import numpy as np

from training.extract_poses import assign_players


def test_assign_two_best_by_conf_then_depth_order():
    # Uniform-y people (no lunge ambiguity): mean-y and max-y-of-visible agree here, so
    # this case doesn't distinguish the two rules, but still pins down basic behavior --
    # top-2 selection by confidence, bottom-first depth ordering.
    k = np.zeros((3, 17, 2), dtype=np.float32)
    k[0, :, 1] = 600  # bottom person
    k[1, :, 1] = 100  # top person
    k[2, :, 1] = 350  # mid person, low conf -> excluded
    s = np.ones((3, 17), dtype=np.float32)
    s[2] *= 0.1
    ok, osc = assign_players(k, s)
    assert ok[0, 0, 1] == 600 and ok[1, 0, 1] == 100


def test_assign_handles_empty():
    ok, osc = assign_players(np.zeros((0, 17, 2)), np.zeros((0, 17)))
    assert ok.shape == (2, 17, 2) and (osc == 0).all()


def test_assign_handles_single_person():
    # Only one detected person -> slot 0 filled, slot 1 stays zero-padded.
    k = np.zeros((1, 17, 2), dtype=np.float32)
    k[0, :, 1] = 250
    s = np.ones((1, 17), dtype=np.float32)
    ok, osc = assign_players(k, s)
    assert ok.shape == (2, 17, 2) and osc.shape == (2, 17)
    assert ok[0, 0, 1] == 250
    assert (ok[1] == 0).all() and (osc[1] == 0).all()


def test_assign_conf_selection_independent_of_depth_ordering():
    # CONTROLLER-RULING case: person selection (top-2 by mean confidence) and depth
    # ordering (largest max-visible-y first) are independent steps. Person 0 (far/top,
    # y=100) has much higher confidence than person 1 (near/bottom, y=600), but both
    # still clear the top-2 confidence cut over person 2 (excluded, very low conf).
    # Depth ordering must still put the physically-nearer person 1 in slot 0, regardless
    # of its lower (but still top-2) confidence relative to person 0.
    k = np.zeros((3, 17, 2), dtype=np.float32)
    k[0, :, 1] = 100.0  # top/far player
    k[1, :, 1] = 600.0  # bottom/near player
    k[2, :, 1] = 350.0  # extraneous mid detection, excluded by low confidence
    s = np.zeros((3, 17), dtype=np.float32)
    s[0] = 0.9  # top player: high confidence
    s[1] = 0.35  # bottom player: lower confidence, but still beats person 2
    s[2] = 0.1  # excluded
    ok, osc = assign_players(k, s)
    assert ok[0, 0, 1] == 600.0  # slot 0 = nearer/bottom person, despite lower confidence
    assert ok[1, 0, 1] == 100.0


def test_assign_lunge_feet_ordering_beats_mean_y():
    # Regression test for the CONTROLLER-RULED fix: mean-y is NOT used for depth
    # ordering because it is unreliable for a lunging/diving player. Person 1 here
    # ("near", lunging) has most keypoints raised high (y=100, e.g. outstretched torso
    # and arms) except both ankles (COCO indices 15, 16), which stay planted low
    # (y=400) -- the real near-camera signal. Person 0 ("far", standing) is uniform at
    # y=150.
    k = np.zeros((2, 17, 2), dtype=np.float32)
    s = np.ones((2, 17), dtype=np.float32)
    k[0, :, 1] = 150.0  # far/standing player: mean=150, max=150
    k[1, :, 1] = 100.0  # near/lunging player: most keypoints raised ...
    k[1, 15, 1] = 400.0  # ... except left ankle ...
    k[1, 16, 1] = 400.0  # ... and right ankle, still near the camera

    # Sanity check that this is a genuine mean-y-vs-feet disagreement: under the OLD
    # (now-rejected) mean-y rule, person 0 would have been ranked nearer/bottom because
    # person 1's mean is dragged down by its many raised keypoints.
    assert k[1, :, 1].mean() < k[0, :, 1].mean()

    ok, osc = assign_players(k, s)
    # NEW rule (max-y-of-visible-keypoints, i.e. feet): person 1's planted ankles (400)
    # beat person 0's uniform 150 -> person 1 correctly lands in slot 0 (nearer/bottom).
    assert ok[0, 0, 1] == 100.0  # slot 0 holds person 1's data (nose kept at y=100)
    assert ok[0, 15, 1] == 400.0  # slot 0's ankle is still the planted-low one
    assert ok[1, 0, 1] == 150.0  # slot 1 holds person 0's data


def test_assign_depth_stat_falls_back_to_ungated_max_when_all_below_threshold():
    # If NO keypoint for a person clears score_thr, depth falls back to the ungated
    # max(y) over all keypoints rather than treating the person as depth-less.
    k = np.zeros((2, 17, 2), dtype=np.float32)
    k[0, :, 1] = 200.0
    k[1, :, 1] = 500.0
    s = np.full((2, 17), 0.1, dtype=np.float32)  # below the default score_thr=0.3
    ok, osc = assign_players(k, s)
    assert ok[0, 0, 1] == 500.0
    assert ok[1, 0, 1] == 200.0


def test_assign_gate_independent_per_person_mixed_score():
    # Person 0: one keypoint (idx 5) has the largest y (500) but score < 0.3, so the
    # gated depth stat must exclude it and fall back to the next-highest *scoring* y
    # (200, from the other 16 keypoints, all scored 0.9). Person 1: EVERY keypoint
    # scores below 0.3, so its depth stat must use the ungated max(y) fallback (350).
    # This pins down that the score gate is evaluated independently per person: person
    # 0's low-score outlier must not leak into person 1's fallback decision (or vice
    # versa) -- each person's gated-vs-fallback branch is decided on its own scores.
    k = np.zeros((2, 17, 2), dtype=np.float32)
    s = np.zeros((2, 17), dtype=np.float32)
    k[0, :, 1] = 200.0
    k[0, 5, 1] = 500.0  # highest-y point for person 0, but ...
    s[0, :] = 0.9
    s[0, 5] = 0.1  # ... scored below threshold -> excluded from person 0's gated max
    k[1, :, 1] = 350.0  # uniform y, all below threshold -> ungated fallback = 350
    s[1, :] = 0.1
    ok, osc = assign_players(k, s)
    # person 1's fallback depth (350) > person 0's gated depth (200) -> person 1 in slot 0.
    # (If person 0's gate incorrectly leaked the 500 outlier through, 500 > 350 would
    # wrongly put person 0 in slot 0 instead.)
    assert ok[0, 0, 1] == 350.0
    assert ok[1, 0, 1] == 200.0
