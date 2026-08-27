"""Adapter: raw ShuttleSet CSVs -> canonical training label schema.

See `training/notes/shuttleset-format.md` for the full dataset-format investigation this
module implements against. Key facts (do not re-derive from the brief's illustrative code,
which was written before the dataset was inspected):

- Raw per-set-file columns actually used here: `rally`, `ball_round`, `time`, `frame_num`,
  `player` ('A'/'B'), `type` (Chinese shot-type text), `getpoint_player` ('A'/'B'/NaN).
- `rally` is only unique *within one set{N}.csv file*, not across a match's sets, hence the
  `set_num` parameter on `convert()` (see the rally_id scheme below).
- `type` uses a 19-value Chinese vocabulary; `STROKE_MAP` implements the controller-ruled
  19->8 mapping (canonical mapping section of the notes file, plus the addendum ruling
  後場抽平球 -> drive). `未知球種` ("unknown shot type") maps to `None` and its rows are
  dropped (with a count logged to stderr) rather than raising.
- `fps` is per-match, not a global constant, and must be derived from a whole-match OLS
  regression of `frame_num` on `seconds(time)` (see `resolve_fps`) because `time` is
  floor-truncated to whole seconds while `frame_num` is exact -- a naive per-row ratio is
  biased high. The regression slope is then snapped to the nearest of the three plausible
  broadcast frame rates {25, 29.97, 30}.
- `getpoint_player` (the rally winner letter) is NaN on every row except (usually) the
  rally's last stroke; 174/3683 real rallies have it NaN on every row (no annotated outcome)
  -> `rally_winner = -1` sentinel for those.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Controller-ruled 19 -> 8 canonical stroke mapping -----------------------------------
# Source: training/notes/shuttleset-format.md, "Canonical 8-class stroke-type mapping
# (CONTROLLER RULING)" section + addendum (後場抽平球 -> drive). `未知球種` maps to None and
# is dropped (logged, not silently discarded, not an error -- unlike any *other* unmapped
# key, which raises KeyError by design, per the Task 6 contract).
STROKE_MAP: dict[str, str | None] = {
    "殺球": "smash", "點扣": "smash", "撲球": "smash",
    "長球": "clear",
    "切球": "drop", "過度切球": "drop",
    "放小球": "net", "擋小球": "net", "勾球": "net",
    "挑球": "lift", "防守回挑": "lift",
    "平球": "drive", "小平球": "drive", "推球": "drive", "防守回抽": "drive",
    "後場抽平球": "drive",  # controller addendum ruling
    "發短球": "serve", "發長球": "serve",
    "未知球種": None,  # dropped by design, counted + logged, not a KeyError
}

PAD_BEFORE, PAD_AFTER = 15, 30  # frames of context padded around first/last hit of a rally

_OUTPUT_COLUMNS = [
    "match_id", "video_file", "fps", "rally_id", "hit_frame", "player",
    "stroke", "rally_start_frame", "rally_end_frame", "rally_winner",
]

_FPS_CANDIDATES = (25.0, 29.97, 30.0)


def _parse_time_seconds(time_col: pd.Series) -> np.ndarray:
    """Parse `H:MM:SS`/`HH:MM:SS` strings to whole seconds. Splits on ':' rather than
    assuming a fixed string width, per the notes ('time' is inconsistently zero-padded)."""
    parts = time_col.astype(str).str.split(":", expand=True).astype(int)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy(dtype=float)


def resolve_fps(set_dfs: list[pd.DataFrame]) -> float:
    """Robust per-match fps: pool every stroke row across all of a match's set{N}.csv files,
    OLS-regress frame_num (y) on seconds(time) (x), then snap the fitted slope to the nearest
    of {25, 29.97, 30}. See module docstring / notes §(e) for why a naive per-row ratio is
    biased high and must not be used."""
    xs = np.concatenate([_parse_time_seconds(df["time"]) for df in set_dfs])
    ys = np.concatenate([df["frame_num"].to_numpy(dtype=float) for df in set_dfs])
    if len(xs) < 2:
        raise ValueError(f"resolve_fps needs at least 2 (time, frame_num) rows, got {len(xs)}")
    if np.ptp(xs) == 0:
        raise ValueError(
            "resolve_fps: all seconds(time) values are identical -- cannot fit a slope "
            f"(x={xs[0]!r})"
        )
    slope, _intercept = np.polyfit(xs, ys, 1)
    snapped = min(_FPS_CANDIDATES, key=lambda c: abs(c - slope))
    if abs(slope - snapped) > 0.5:
        raise ValueError(
            f"resolve_fps: fitted slope {slope!r} is not plausibly close to any nominal fps "
            f"in {_FPS_CANDIDATES} (nearest candidate {snapped}, off by "
            f"{abs(slope - snapped):.4f} > 0.5 tolerance)"
        )
    return snapped


def convert(raw: pd.DataFrame, match_id: str, fps: float, set_num: int = 1) -> pd.DataFrame:
    """Convert one raw set{N}.csv dataframe to the canonical label schema.

    rally_id scheme: `rally` in the raw data is only unique within a single set file (it
    resets to 1 in set2.csv, set3.csv, ...). To make rally_id unique within a match, combine
    the 1-based `set_num` (passed by the caller, one set file at a time) with the raw `rally`
    number: `rally_id = set_num * 1000 + rally`. This is collision-free since no set file in
    the real dataset exceeds ~50 rallies (max observed: 48) and no match exceeds 3 sets.
    `set_num` defaults to 1 so the 3-arg call from the Task 6 contract/tests still works for
    single-set-file conversions.
    """
    players = sorted(p for p in raw["player"].dropna().unique())
    if set(players) != {"A", "B"}:
        raise ValueError(
            f"match {match_id!r} set_num={set_num}: expected exactly players {{'A', 'B'}}, "
            f"got {players!r} -- refusing to guess a pmap and risk a silent id sign-flip"
        )
    pmap: dict[str, int] = {"A": 0, "B": 1}

    assert (raw["frame_num"].dropna() % 1 == 0).all(), (
        f"match {match_id!r} set_num={set_num}: non-integer frame_num values found -- "
        "frame_num is expected to be an integer-valued float in the raw ShuttleSet CSVs"
    )

    dropped_unknown = 0
    rows: list[dict] = []
    for rally_num, g in raw.groupby("rally", sort=True):
        g = g.sort_values("ball_round")
        first_frame = int(g["frame_num"].min())
        last_frame = int(g["frame_num"].max())
        start = max(first_frame - PAD_BEFORE, 0)
        end = last_frame + PAD_AFTER

        gp = g["getpoint_player"].dropna()
        winner = pmap[gp.iloc[-1]] if len(gp) else -1

        rally_id = set_num * 1000 + int(rally_num)
        for _, r in g.iterrows():
            raw_type = str(r["type"]).strip()
            stroke = STROKE_MAP[raw_type]  # KeyError on truly unmapped type -- by design
            if stroke is None:
                dropped_unknown += 1
                continue
            rows.append({
                "match_id": match_id,
                "video_file": f"{match_id}.mp4",
                "fps": float(fps),
                "rally_id": rally_id,
                "hit_frame": int(r["frame_num"]),
                "player": pmap[r["player"]],
                "stroke": stroke,
                "rally_start_frame": start,
                "rally_end_frame": end,
                "rally_winner": winner,
            })

    if dropped_unknown:
        print(
            f"[{match_id} set {set_num}] dropped {dropped_unknown} 未知球種 (unknown shot "
            "type) rows",
            file=sys.stderr,
        )

    out = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    # Cast unconditionally (including the empty-frame case, e.g. an all-未知球種 set file) so
    # every convert() call returns identically-dtyped columns -- an object-dtyped empty frame
    # concatenated against int64-dtyped frames elsewhere would silently upcast/poison dtypes.
    out = out.astype({
        "match_id": str, "video_file": str, "fps": float, "rally_id": int,
        "hit_frame": int, "player": int, "stroke": str,
        "rally_start_frame": int, "rally_end_frame": int, "rally_winner": int,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="training/data/raw/shuttleset")
    ap.add_argument("--out", default="training/data/processed/labels.parquet")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    match_csv = pd.read_csv(raw_dir / "set" / "match.csv")

    frames: list[pd.DataFrame] = []
    total_raw_rows = 0
    total_dropped = 0
    for _, row in match_csv.iterrows():
        match_id = str(row["video"])  # folder name under set/, unique per match (verified 44/44)
        folder = raw_dir / "set" / match_id
        set_files = sorted(folder.glob("set*.csv"))
        if not set_files:
            print(f"WARNING: no set*.csv files found for match {match_id!r}", file=sys.stderr)
            continue
        set_dfs = [pd.read_csv(f) for f in set_files]
        fps = resolve_fps(set_dfs)
        for set_num, df in enumerate(set_dfs, start=1):
            total_raw_rows += len(df)
            converted = convert(df, match_id, fps, set_num=set_num)
            total_dropped += len(df) - len(converted)
            frames.append(converted)

    if not frames:
        raise ValueError(
            "no set-file dataframes were converted (frames list is empty) -- check "
            f"--raw-dir={args.raw_dir!r} points at a valid ShuttleSet layout with "
            "set/match.csv and set/<video_folder>/set*.csv files"
        )
    out = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(
        f"{len(out)} strokes ({total_raw_rows} raw rows, {total_dropped} dropped as "
        f"未知球種), {out.match_id.nunique()} matches -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
