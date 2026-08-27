"""Shared helpers for the training CLIs (`train_stroke.py`, `train_rally.py`).

Extracted so both CLIs share one hardened implementation of seeding and
splits.json validation rather than duplicating (and risking divergent) logic.
"""
from __future__ import annotations
import json
import random
import sys

import numpy as np
import torch

SPLIT_NAMES = ("train", "val", "test")


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def resolve_splits(all_matches: set, splits_path: str) -> dict:
    """Load and validate a splits.json against the set of match ids actually
    present in the data file.

    Validates, in order:
    - presence: "train"/"val"/"test" must all be keys in the JSON (a key that
      is simply missing is a different, more likely-accidental error than a
      key present with an explicit `[]`, so it gets its own check/message
      instead of silently defaulting to empty via `dict.get`).
    - disjointness: no match id may appear in more than one split (would leak
      match-level data across train/val/test).
    - unknown ids: any listed match id absent from `all_matches` is warned
      about on stderr (but not treated as fatal -- the caller simply won't
      find rows for it).

    Returns the raw `{name: [match_id, ...]}` dict, unfiltered -- filtering
    down to ids actually present in the data is the caller's job.
    """
    splits = json.loads(open(splits_path).read())

    missing_keys = [name for name in SPLIT_NAMES if name not in splits]
    if missing_keys:
        raise ValueError(
            f"{splits_path}: missing required key(s) {missing_keys} -- "
            "'train'/'val'/'test' must all be present (use [] for an "
            "explicitly empty split)"
        )

    sets = {name: set(splits[name]) for name in SPLIT_NAMES}
    for a_name, b_name in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a_name] & sets[b_name]
        if overlap:
            raise ValueError(
                f"{splits_path}: match id(s) {sorted(overlap)} appear in both "
                f"'{a_name}' and '{b_name}' -- splits must be disjoint"
            )

    for name in SPLIT_NAMES:
        unknown = sorted(mid for mid in splits[name] if mid not in all_matches)
        if unknown:
            print(f"WARNING: {splits_path} '{name}' names match id(s) {unknown} "
                  f"that match zero rows in the data's `match` array", file=sys.stderr)

    return splits
