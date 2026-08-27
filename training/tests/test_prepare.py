import math

import numpy as np
import pandas as pd
import pytest

from shuttlesense_core.schemas import STROKE_CLASSES
from training.prepare_shuttleset import STROKE_MAP, convert, resolve_fps


def _raw():
    # Real ShuttleSet set{N}.csv columns per training/notes/shuttleset-format.md:
    # rally, ball_round, time, frame_num, roundscore_A, roundscore_B, player, server,
    # type, ..., getpoint_player, ... (only the columns convert() actually reads are
    # included here). `getpoint_player` is populated only on the rally's final row.
    return pd.DataFrame({
        "rally": [1, 1, 1, 2, 2],
        "ball_round": [1.0, 2.0, 3.0, 1.0, 2.0],
        "frame_num": [100.0, 130.0, 170.0, 400.0, 430.0],
        "player": ["A", "B", "A", "B", "A"],
        "type": ["發短球", "挑球", "殺球", "發長球", "放小球"],
        "getpoint_player": ["A", np.nan, "A", np.nan, "B"],
    })


def test_convert_schema_and_mapping():
    out = convert(_raw(), match_id="m01", fps=30.0)
    assert set(out.columns) == {
        "match_id", "video_file", "fps", "rally_id", "hit_frame", "player",
        "stroke", "rally_start_frame", "rally_end_frame", "rally_winner",
    }
    assert out["stroke"].isin(STROKE_CLASSES).all()

    r1 = out[out.rally_id == out.rally_id.iloc[0]]
    assert r1["rally_start_frame"].iloc[0] <= 100 and r1["rally_end_frame"].iloc[0] >= 170
    assert (r1["rally_winner"] == 0).all()  # player A -> 0 (getpoint_player == 'A' on last row)

    r2 = out[out.rally_id != out.rally_id.iloc[0]]
    assert (r2["rally_winner"] == 1).all()  # getpoint_player == 'B' on last row -> 1


def test_rally_ids_unique_within_match_via_set_num():
    raw = _raw()
    out1 = convert(raw, "m01", 30.0, set_num=1)
    out2 = convert(raw, "m01", 30.0, set_num=2)
    # same underlying `rally` numbers (1, 2) in both set files must not collide
    assert set(out1.rally_id) & set(out2.rally_id) == set()


def test_padding_applied_and_clamped_at_zero():
    raw = _raw()
    out = convert(raw, "m01", 30.0)
    r1 = out[out.hit_frame.isin([100, 130, 170])]
    # PAD_BEFORE=15, PAD_AFTER=30 around first/last hit frame of the rally
    assert r1["rally_start_frame"].iloc[0] == max(100 - 15, 0)
    assert r1["rally_end_frame"].iloc[0] == 170 + 30


def test_unknown_stroke_raises():
    raw = _raw()
    raw.loc[0, "type"] = "???"
    with pytest.raises(KeyError):
        convert(raw, "m01", 30.0)


def test_weizhi_qiuzhong_dropped_and_logged(capsys):
    raw = _raw()
    raw.loc[0, "type"] = "未知球種"  # explicit "unknown shot type" -> dropped, not an error
    out = convert(raw, "m01", 30.0)
    assert 100 not in out["hit_frame"].values
    assert len(out) == 4  # 5 raw rows - 1 dropped
    captured = capsys.readouterr()
    assert "1" in captured.err  # dropped-count logged to stderr


def test_winner_minus_one_when_getpoint_player_all_nan():
    raw = _raw()
    raw["getpoint_player"] = np.nan  # neither rally ever got an outcome annotation
    out = convert(raw, "m01", 30.0)
    assert (out["rally_winner"] == -1).all()


def test_stroke_map_covers_full_real_vocabulary():
    # The 19-value Chinese vocabulary from training/notes/shuttleset-format.md, including the
    # controller-ruling addendum (後場抽平球 -> drive) and 未知球種 -> None (dropped).
    real_vocab = {
        "放小球", "挑球", "擋小球", "推球", "長球", "殺球", "切球", "發短球", "點扣",
        "未知球種", "勾球", "過度切球", "平球", "撲球", "後場抽平球", "防守回抽",
        "發長球", "防守回挑", "小平球",
    }
    assert real_vocab <= STROKE_MAP.keys()
    mapped = {v for k, v in STROKE_MAP.items() if k in real_vocab and v is not None}
    assert mapped <= set(STROKE_CLASSES)
    assert STROKE_MAP["未知球種"] is None
    assert STROKE_MAP["後場抽平球"] == "drive"


def _time_str(total_seconds: int) -> str:
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def test_resolve_fps_snaps_truncated_seconds_to_25_not_25_01():
    # Synthetic data reproducing the documented truncation-bias trap: `time` is floor-truncated
    # to whole seconds while `frame_num` reflects the exact (fractional) elapsed time. A naive
    # per-row ratio frame_num / seconds(time) would be biased high (~25.01+); the whole-match OLS
    # slope must recover the true fps (25.0) and snap to it exactly.
    true_fps = 25.0
    frac = 0.6  # constant fractional-second offset baked into every "true" timestamp
    xs = list(range(0, 3000, 7))  # truncated-second values actually stored in `time`
    frame_nums = [true_fps * (x + frac) for x in xs]
    df = pd.DataFrame({
        "time": [_time_str(x) for x in xs],
        "frame_num": frame_nums,
    })
    fps = resolve_fps([df])
    assert fps == 25.0


def test_resolve_fps_snaps_to_2997_when_slope_closest_to_it():
    true_fps = 29.97
    frac = 0.3
    xs = list(range(0, 2500, 5))
    frame_nums = [true_fps * (x + frac) for x in xs]
    df = pd.DataFrame({
        "time": [_time_str(x) for x in xs],
        "frame_num": frame_nums,
    })
    fps = resolve_fps([df])
    assert fps == 29.97


def test_resolve_fps_pools_multiple_set_files():
    true_fps = 30.0
    frac = 0.1
    xs1 = list(range(0, 1000, 3))
    xs2 = list(range(1000, 2000, 3))
    df1 = pd.DataFrame({
        "time": [_time_str(x) for x in xs1],
        "frame_num": [true_fps * (x + frac) for x in xs1],
    })
    df2 = pd.DataFrame({
        "time": [_time_str(x) for x in xs2],
        "frame_num": [true_fps * (x + frac) for x in xs2],
    })
    fps = resolve_fps([df1, df2])
    assert fps == 30.0
