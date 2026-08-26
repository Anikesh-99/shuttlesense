"""Court homography math: fit an image-px -> court-meters projective
transform from point correspondences, and apply it to points.

This module is a Phase 2 foundation: it only does the linear-algebra
mapping between two 2D point sets. It does not do any court detection,
tracking, or video I/O.
"""
from __future__ import annotations

import cv2
import numpy as np

# Standard badminton doubles court dimensions (meters), including lines.
COURT_W = 6.1
COURT_L = 13.4


def fit_homography(img_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """Fit a homography mapping image pixel coordinates to court meters.

    Requires >= 4 point correspondences between `img_pts` and `court_pts`
    (each an (N, 2) array, N >= 4). Raises ValueError if fewer than 4
    correspondences are given, or if the points are degenerate (e.g.
    collinear) such that no homography can be fit.
    """
    img = np.asarray(img_pts, dtype=np.float64)
    court = np.asarray(court_pts, dtype=np.float64)
    if img.shape[0] < 4 or court.shape[0] < 4:
        raise ValueError("fit_homography requires >= 4 point correspondences")
    try:
        H, _ = cv2.findHomography(img, court)
    except cv2.error as e:
        raise ValueError(f"homography fit failed: {e}") from e
    if H is None:
        raise ValueError("homography fit failed")
    return H


def to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (N, 2) image pixel points to (N, 2) court-meter points via H."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)
