import numpy as np
import pytest
from shuttlesense_core.homography import COURT_L, COURT_W, fit_homography, to_court


def test_roundtrip_known_projection():
    court = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float64)
    img = np.array([[300, 600], [980, 600], [1180, 100], [100, 100]], dtype=np.float64)
    H = fit_homography(img, court)
    mapped = to_court(H, img)
    np.testing.assert_allclose(mapped, court, atol=1e-6)
    center = to_court(H, np.array([[640.0, 350.0]]))
    assert 0 < center[0, 0] < COURT_W and 0 < center[0, 1] < COURT_L


def test_court_constants_are_doubles_court_meters():
    # Standard badminton doubles court: 6.1m wide x 13.4m long (incl. lines).
    assert COURT_W == 6.1
    assert COURT_L == 13.4


def test_fit_homography_with_more_than_four_points():
    # 5th correspondence (court center -> its true image location under the
    # exact 4-point homography) so all 5 pairs are mutually consistent.
    court = np.array(
        [[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L], [COURT_W / 2, COURT_L / 2]],
        dtype=np.float64,
    )
    img = np.array(
        [[300, 600], [980, 600], [1180, 100], [100, 100], [640.00000652, 406.81817507]],
        dtype=np.float64,
    )
    H = fit_homography(img, court)
    mapped = to_court(H, img)
    assert H.shape == (3, 3)
    np.testing.assert_allclose(mapped, court, atol=1e-2)


def test_fit_homography_raises_on_collinear_points():
    # Four collinear points cannot determine a homography.
    court = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    img = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    with pytest.raises(ValueError):
        fit_homography(img, court)


def test_fit_homography_raises_on_too_few_points():
    # Only 3 correspondences given; a homography needs at least 4.
    court = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L]], dtype=np.float64)
    img = np.array([[300, 600], [980, 600], [1180, 100]], dtype=np.float64)
    with pytest.raises(ValueError):
        fit_homography(img, court)


def test_to_court_output_shape_matches_input_count():
    court = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float64)
    img = np.array([[300, 600], [980, 600], [1180, 100], [100, 100]], dtype=np.float64)
    H = fit_homography(img, court)
    pts = np.array([[640.0, 350.0], [500.0, 400.0], [700.0, 300.0]])
    mapped = to_court(H, pts)
    assert mapped.shape == (3, 2)


def test_to_court_returns_float64_ndarray():
    court = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float64)
    img = np.array([[300, 600], [980, 600], [1180, 100], [100, 100]], dtype=np.float64)
    H = fit_homography(img, court)
    mapped = to_court(H, np.array([[640.0, 350.0]]))
    assert mapped.dtype == np.float64
