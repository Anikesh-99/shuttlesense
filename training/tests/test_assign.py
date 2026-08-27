import numpy as np

from training.extract_poses import assign_players


def test_assign_two_best_by_conf_then_depth_order():
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
