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
    # PAD_BEFORE=15, PAD_AFTER=30 -- literal expected values, not a re-derivation of the
    # padding formula under test.
    assert r1["rally_start_frame"].iloc[0] == 85
    assert r1["rally_end_frame"].iloc[0] == 200

    # Rally whose first hit is close enough to frame 0 that naive padding (frame - 15) would
    # go negative; rally_start_frame must clamp to 0, not -10.
    early = pd.DataFrame({
        "rally": [3, 3],
        "ball_round": [1.0, 2.0],
        "frame_num": [5.0, 40.0],
        "player": ["B", "A"],
        "type": ["發長球", "殺球"],
        "getpoint_player": [np.nan, "A"],
    })
    raw_with_early = pd.concat([raw, early], ignore_index=True)
    out2 = convert(raw_with_early, "m01", 30.0)
    r3 = out2[out2.hit_frame.isin([5, 40])]
    assert r3["rally_start_frame"].iloc[0] == 0
    assert r3["rally_end_frame"].iloc[0] == 70  # 40 + PAD_AFTER(30), unaffected by clamping


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
    assert "dropped 1 " in captured.err  # exact dropped-count logged to stderr, not a loose "1"


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


def test_convert_raises_on_non_ab_player_set():
    raw = _raw()
    raw.loc[0, "player"] = "C"  # a 3rd letter must not silently participate in a pmap
    with pytest.raises(ValueError):
        convert(raw, "m01", 30.0)


def test_convert_raises_on_single_letter_player_set():
    raw = _raw()
    raw["player"] = "A"  # only one letter present -- must not silently default to id 0 for all
    with pytest.raises(ValueError):
        convert(raw, "m01", 30.0)


def test_convert_raises_on_non_integer_frame_num():
    raw = _raw()
    raw.loc[0, "frame_num"] = 100.5
    with pytest.raises(AssertionError):
        convert(raw, "m01", 30.0)


def _time_str(total_seconds: int) -> str:
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _honest_truncation_df(true_fps: float, n: int, t_start: float, t_end: float,
                           seed: int) -> pd.DataFrame:
    """Simulate `n` stroke rows spread across [t_start, t_end) true elapsed seconds, each with
    its OWN random fractional-second offset (not a single constant frac shared by every row --
    that was an unrealistically clean noise model). `time` gets floor-truncated to whole
    seconds (as the real dataset does); `frame_num` is the true, exact frame index. This is the
    honest version of the documented truncation-bias trap: per-row ratios would each be biased
    high by a different, row-specific amount, but the whole-match OLS slope should still recover
    `true_fps` because the truncation noise is independent of the timestamp itself and averages
    out across many rows."""
    rng = np.random.default_rng(seed)
    true_times = np.sort(rng.uniform(t_start, t_end, size=n))
    frame_nums = np.round(true_fps * true_times)
    truncated_seconds = np.floor(true_times).astype(int)
    return pd.DataFrame({
        "time": [_time_str(int(s)) for s in truncated_seconds],
        "frame_num": frame_nums,
    })


def test_resolve_fps_snaps_truncated_seconds_to_25_not_25_01():
    # Realistic noise model (see _honest_truncation_df): per-row random fractional-second
    # truncation, ~800 strokes spread across a ~50-minute match at true fps 25.0. A naive
    # per-row ratio frame_num / seconds(time) would be biased high (~25.01+) on most rows; the
    # whole-match OLS slope must recover the true fps and snap to exactly 25.0.
    df = _honest_truncation_df(true_fps=25.0, n=800, t_start=0.0, t_end=3000.0, seed=1)
    assert resolve_fps([df]) == 25.0


def test_resolve_fps_snaps_to_2997_when_slope_closest_to_it():
    # Same honest per-row truncation noise model, at true fps 29.97 (the one real match in the
    # dataset that resolves to this rate) -- confirms the 3-way snap distinguishes 29.97 from
    # the much more common 30.0 even under realistic per-row noise.
    df = _honest_truncation_df(true_fps=29.97, n=800, t_start=0.0, t_end=3000.0, seed=2)
    assert resolve_fps([df]) == 29.97


def test_resolve_fps_pools_multiple_set_files():
    # `time`/`frame_num` are continuous across a match's set files (not reset per set, per the
    # notes), so set2's true elapsed time picks up where set1 left off.
    true_fps = 30.0
    df1 = _honest_truncation_df(true_fps, n=400, t_start=0.0, t_end=1500.0, seed=3)
    df2 = _honest_truncation_df(true_fps, n=400, t_start=1500.0, t_end=3000.0, seed=4)
    assert resolve_fps([df1, df2]) == 30.0


def test_resolve_fps_raises_on_fewer_than_two_rows():
    df = pd.DataFrame({"time": ["0:00:01"], "frame_num": [25.0]})
    with pytest.raises(ValueError):
        resolve_fps([df])


def test_resolve_fps_raises_on_constant_time():
    df = pd.DataFrame({"time": ["0:00:01", "0:00:01", "0:00:01"], "frame_num": [25.0, 26.0, 27.0]})
    with pytest.raises(ValueError):
        resolve_fps([df])


def test_resolve_fps_raises_when_slope_implausible():
    # A slope nowhere near 25/29.97/30 (e.g. a bogus 60fps-ish series) must raise rather than
    # silently snapping to the nearest (still wildly wrong) candidate.
    xs = np.arange(0, 200, 1.0)
    df = pd.DataFrame({
        "time": [_time_str(int(x)) for x in xs],
        "frame_num": 60.0 * xs,
    })
    with pytest.raises(ValueError):
        resolve_fps([df])
