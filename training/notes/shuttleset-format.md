# ShuttleSet stroke-level dataset — actual format notes

Source: `wywyWang/CoachAI-Projects` GitHub repo, subdirectory `ShuttleSet/` (top-level dir,
sibling to `Movement Forecasting`, `Stroke Forecasting`, etc.). This is the official dataset
release for the KDD 2023 paper *"ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for
Badminton Tactical Analysis"* (https://arxiv.org/abs/2306.04948).

The brief's guess (CSVs directly at repo root) was wrong in detail but right in spirit: the CSVs
are in the CoachAI-Projects repo, just nested under a `ShuttleSet/` folder. No need to fall back
to a separate `wywyWang/ShuttleSet` repo — it wasn't used; the copy here comes straight from
CoachAI-Projects.

Copied into `training/data/raw/shuttleset/` (mirrors the source layout exactly):
```
training/data/raw/shuttleset/
  README.md                     # original dataset README (schema doc, kept verbatim)
  set/
    match.csv                   # 44 rows, one per match: metadata + video URL
    homography.csv              # 44 rows, one per match: court homography for pixel<->real coords
    <PlayerA>_<PlayerB>_<Tournament>_<Round>/
      set1.csv, set2.csv[, set3.csv]   # one CSV per played set, stroke-level rows
```
Note there is also a `CoachAI-Challenge-IJCAI2023/ShuttleSet22/` directory in the same repo
containing a larger, differently-shaped superset (more 2022-era matches, `set/` subfolders with
no top-level `match.csv`/`homography.csv` inspected in this task). We did NOT copy it — out of
scope for this task, but flagging it as a candidate for later expansion if more training data is
needed. It appeared to follow the same set1.csv/set2.csv convention per match folder, unverified
column-for-column.

**Real counts (measured, not from README):** 44 matches, 104 sets, 3,683 rallies, 36,484 strokes.
The dataset README claims "3,685 rallies, and 36,492 strokes" — off by a small amount (2 rallies,
8 strokes) from what's actually in the files. Treat the README's headline numbers as approximate;
trust direct inspection.

## (a) Exact column names

### `set/match.csv` (44 rows, one per match)
```
id, video, tournament, round, year, month, day, set, duration, winner, loser, downcourt, url
```
- `video`: folder name under `set/` for this match (exact string match verified for all 44 rows —
  zero folders missing from match.csv and vice versa).
- `id`: serial match number (1..44).
- `set`: number of sets played in the match (2 or 3). Distribution: 28 matches with 2 sets, 16 with 3.
- `duration`: match duration in minutes.
- `winner` / `loser`: real player names as free text, e.g. `"Kento MOMOTA"`, `"CHOU Tien Chen"`.
- `downcourt`: 0/1 flag (meaning not documented in README; likely "far/near court side" for camera
  orientation — not verified, treat as advisory only).
- `url`: YouTube URL of the broadcast video (`https://www.youtube.com/watch?v=...`). **One row
  (match id=12, SHI Yuqi's match) has a NULL url** — no video source recorded for that match.

Example rows (real output):
```
   id                                                             video          tournament           round  year  month  day  set  duration          winner             loser  downcourt                                          url
0   1               Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals    Fuzhou Open 2019          Finals  2019     11   10    3        83    Kento MOMOTA    CHOU Tien Chen          0  https://www.youtube.com/watch?v=O669aZhH0LI
1   2            CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage   World Tour Finals     Group-Stage  2019     12   13    2        52       CHEN Long    CHOU Tien Chen          1  https://www.youtube.com/watch?v=-aOI9_JxoWc
2   3                 Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final     KOREA OPEN 2019          Finals  2019      9   29    2        53    Kento MOMOTA    CHOU Tien Chen          1  https://www.youtube.com/watch?v=eugfCRwSBJo
3   4           CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal   Denmark Open 2019  Quarter-finals  2019     10   17    2        54       CHEN Long    CHOU Tien Chen          0  https://www.youtube.com/watch?v=y6QbtrTV-K0
```

### `set/homography.csv` (44 rows, one per match)
```
id, video, homography_matrix, upleft_x, upright_x, downleft_x, downright_x, upleft_y, upright_y, downleft_y, downright_y
```
`homography_matrix` is a stringified 3x3 nested list (Python list-of-lists literal, needs
`ast.literal_eval` or `json.loads` after replacing nothing — it's valid Python/JSON syntax with
scientific notation). Per README: "can be used to transform coordinates from the real-world
system back to the camera system by p = H^-1 p'". The 8 `up*/down*` columns give the four
court-corner pixel coordinates in camera space.

Example (row 0):
```
homography_matrix = [[1.0606627138451856, 0.3439667138673875, -497.8978610611467],
                      [0.011919581725440269, 4.586662980083068, -1176.8442597836183],
                      [4.8478780487874024e-06, 0.002012572611469505, 1.0]]
upleft_x=407.6 upright_x=867.0 downleft_x=307.2 downright_x=973.0
upleft_y=308.6 upright_y=307.4 downleft_y=671.2 downright_y=669.4
```

### `set/<match_folder>/set{1,2,3}.csv` (stroke-level rows; 104 files, 36,484 rows total)

**Actual columns (30 — more than the README documents):**
```
rally, ball_round, time, frame_num, roundscore_A, roundscore_B, player, server, type,
aroundhead, backhand, hit_height, hit_area, hit_x, hit_y, landing_height, landing_area,
landing_x, landing_y, lose_reason, win_reason, getpoint_player, flaw,
player_location_area, player_location_x, player_location_y,
opponent_location_area, opponent_location_x, opponent_location_y, db
```

**Discrepancy vs README:** the README's per-stroke field list omits `server`, `hit_height`,
`hit_area`, `hit_x`, `hit_y`, `win_reason`, `flaw`, and `db` entirely. These are real columns
present in every file. Conversely the README calls out `landing_height`/`landing_area` clearly —
those do exist and behave as documented. Do not rely on the README's column list alone; the
column names above were read directly from the CSVs.

Column semantics (inferred + partially README-confirmed):
- `rally`: 1-based rally number, scoped to the file (i.e. resets to 1 in each set1.csv/set2.csv —
  **not globally unique across sets or matches**). To get a globally unique rally id, key on
  `(match_folder, set_file, rally)`.
- `ball_round`: 1-based stroke index within the rally (float dtype but integer-valued, e.g. 1.0, 2.0, ...).
- `time`: wall-clock hit time as `H:MM:SS` or `HH:MM:SS` string — **format is inconsistent within
  the same column** (see (c) below); parse by splitting on `:` and casting each part to int, do
  not regex-match a fixed width.
- `frame_num`: float, the absolute video frame number at which the shot was hit (`frame_num =
  seconds_since_video_start * fps`, confirmed below — NOT rally-relative, NOT reset per rally/set).
- `roundscore_A`, `roundscore_B`: running set score for player A / player B at the time of the shot.
- `player`: `'A'` or `'B'` — the player who hit this shot. **Per README, `A` always denotes the
  player who wins the overall match, `B` the loser** — this assignment is fixed for the whole
  match (all sets), not per-rally. To resolve to a real name, join on `match.csv` by `video`
  (folder name) and use `winner` for A, `loser` for B.
- `server`: integer 1/2/3. Empirically **not** "which player is serving" — it correlates with
  shot position in the rally: value 1 on the first shot of a rally in 3659/3683 rallies (99.3%),
  value 3 on the last shot of a rally (when rally length > 1) in 3508/3654 eligible rallies
  (~96%), value 2 in between. Treat as a weak/imperfect redundant signal for rally-boundary
  detection, not authoritative — use `rally` groupby + `getpoint_player` (below) as the real
  source of truth for boundaries.
- `type`: shot type, **in Chinese**, not English (see (b) below).
- `aroundhead`, `backhand`: 1.0 flag when true, `NaN` otherwise (i.e. boolean-as-optional-1,
  not 0/1). `aroundhead` is NaN 33148/36484 rows (~91%), `backhand` is NaN 22785/36484 (~62%).
- `hit_height`, `hit_area`, `hit_x`, `hit_y`: where/how the shot was hit (hit_height: 1=below net
  rim, 2=above, matching landing_height's convention below). `hit_area`/`hit_x`/`hit_y` are NaN on
  the serve's first datapoint in some rows (~4700/36484, ~13%) — these look like they're NaN
  specifically for the very first shot of a rally (no "incoming" shot to describe hit position
  for) in most cases; not exhaustively verified row-by-row beyond spot checks.
- `landing_height`: 1 = below net height, 2 = above (per README). NaN on 6318/36484 rows (~17%),
  typically the last shot of a rally when the shuttle is not tracked to a landing spot (e.g. hit
  out of bounds by definition, or rally-ending error).
- `landing_area`, `hit_area`: integer domain **1–16** (a 4x4-or-similar court-zone grid on each
  side, not documented further in README beyond "the grid of the shuttle destinations").
- `landing_x`, `landing_y`, `player_location_x/y`, `opponent_location_x/y`: pixel coordinates in
  camera space (same coordinate system as `homography.csv`'s corner points); use the per-match
  homography matrix to project to real-world court coordinates if needed.
- `lose_reason` / `win_reason`: Chinese free-text describing how the rally ended, NaN except on
  the rally-ending row (32975/36484 NaN, i.e. present on ~3509 rows ≈ rally count). Vocabulary
  (paired win/lose framing of the same event) — real value_counts:
  ```
  lose_reason                         win_reason
  對手落地致勝  1189                    落地致勝        1189
  出界         1048                    對手出界        1048
  掛網          841                    對手掛網         841
  未過網         376                    對手未過網        376
  落點判斷失誤     44                    對手落點判斷失誤    44
  對手落地判斷失誤    7                    落地判斷失誤        7
  犯規            4                    對手犯規           4
  ```
- `getpoint_player`: `'A'`/`'B'` — **the player who won the rally** (see (f) below for exact
  encoding/boundary behavior).
- `flaw`: 1.0 flag, present on 1443/36484 rows (~4%), NaN otherwise. Not documented in README;
  appears to mark some kind of annotated shot execution error, unverified beyond that it exists
  and is sparse.
- `player_location_area`, `opponent_location_area`: integer court-zone ids (1–16-ish range) for
  where the hitting player and the opponent were standing, analogous to `landing_area`.
- `db`: constant `0` for every single row in every file (36484/36484). Dead/legacy column — no
  observed information content in this data snapshot.

## (b) Full stroke-type vocabulary with counts

**Real value_counts across all 104 files, 36,484 rows** (`type` column, Chinese text):
```
type
放小球      6290
挑球       5331
擋小球      3620
推球       2925
長球       2922
殺球       2586
切球       2144
發短球      2051
點扣       1648
未知球種     1407
勾球       1371
過度切球     1356
平球        700
撲球        512
後場抽平球     473
防守回抽      406
發長球       373
防守回挑      301
小平球        68
```
That's **19 distinct values**, not the 18 the README's translation table lists. Two
discrepancies vs the README:
1. `未知球種` ("unknown shot type") appears 1,407 times (3.9% of all strokes) and is **not in the
   README's translation table at all**. It must be treated as an explicit "unlabeled/unknown"
   class, not an error — it's common enough (14th-highest by count) to require its own handling
   in Task 6 (either a dedicated UNKNOWN class or filtered out, but do not silently drop or crash).
2. The README's table has `過渡切球` ("passive drop") but the actual data column value is
   `過度切球` (different character: 度 "degree/excessive" vs 渡 "transition/crossing") — likely a
   transcription typo in one place or the other. Confirmed via literal byte comparison that the
   CSV files consistently use `過度切球` (1,356 occurrences, no `過渡切球` variant anywhere in the
   data). Task 6's stroke-type mapping table should key off the CSV value `過度切球` and map it
   to "passive drop" in English, ignoring the README spelling.

English translation table (from README, for the 18 it documents; `未知球種` = "unknown" added by us):
| Chinese (as it appears in CSV) | English |
|---|---|
| 放小球 | net shot |
| 擋小球 | return net |
| 殺球 | smash |
| 點扣 | wrist smash |
| 挑球 | lob |
| 防守回挑 | defensive return lob |
| 長球 | clear |
| 平球 | drive |
| 小平球 | driven flight |
| 後場抽平球 | back-court drive |
| 切球 | drop |
| 過度切球 (README spells 過渡切球) | passive drop |
| 推球 | push |
| 撲球 | rush |
| 防守回抽 | defensive return drive |
| 勾球 | cross-court net shot |
| 發短球 | short service |
| 發長球 | long service |
| 未知球種 (undocumented in README) | unknown |

## (c) Rally boundaries + hit frame/time encoding

- Rallies are delimited by the `rally` column: consecutive rows with the same `rally` value,
  ordered by `ball_round`, form one rally. `rally` resets to 1 at the start of each `set*.csv`
  file — it is **not** globally unique; use `(match_folder, set_file, rally)` as the composite key.
- **Rally winner** is recorded via `getpoint_player`, which is NaN on every row except the last
  stroke of the rally. Verified programmatically across all 3,683 rallies:
  - In 3,508 rallies, `getpoint_player` is non-null exactly and only on the final `ball_round` row.
  - In **174 rallies (4.7%)**, `getpoint_player` is NaN on every row of the rally — i.e. the
    outcome of that rally was never annotated. Task 6 must treat these as "unknown winner", not
    crash/impute.
  - In exactly **1 rally** out of 3,683, `getpoint_player` appears on a non-last row (a genuine
    data quirk, negligible — treat as noise, don't build logic around it).
- **Hit time**: `time` column is a string `H:MM:SS` or `HH:MM:SS` (both forms occur — 4,423 rows
  use the single-digit-hour form `H:MM:SS`, 32,061 use `HH:MM:SS`; **do not assume fixed string
  width**, split on `:` instead). Time is the wall-clock position in the **full match video**
  (not rally-relative, not reset per set) — see (e).
- **Hit frame**: `frame_num` is a float but integer-valued, the absolute frame index in the
  source video corresponding to `time`. Confirmed relationship: `frame_num ≈ seconds(time) * fps`
  where `fps` is per-match (25 or 30, see (e)) — median ratio computed across 20 random rows
  matched the per-match fps to within ~0.03 frames/sec, consistent with simple rounding.
- Time is NOT in seconds as a raw float anywhere in this dataset — it's the `H:MM:SS` string, and
  the frame-accurate signal is `frame_num`. Use `frame_num` for anything that needs to be
  precise; only use `time` for eyeballing/debugging.

Example: `time = "0:07:39"` (459 s), `frame_num = 11496.0` -> 11496/459 = 25.04 fps, matching a
25fps match. Another row from a different (30fps) match: `time = "00:06:57"` (417 s),
`frame_num = 12510.0` -> 12510/417 = 30.0 fps exactly.

## (d) Match -> video file/URL mapping

- There is no video file shipped with the dataset — only labels. Video source is a YouTube URL
  in `match.csv.url`, one row per match, keyed by `match.csv.id` and `match.csv.video`
  (which equals the folder name under `set/`).
- All 44 `set/` subfolder names match `match.csv.video` values exactly (verified: zero folders
  missing from match.csv, zero match.csv videos missing a folder).
- **One match (id=12, loser="SHI Yuqi") has a NULL url** — no video source recorded. Task 7/12
  (video fetching) needs to handle this match as "labels only, no video available" rather than
  erroring on a missing URL.
- Recommended join for later tasks: read `match.csv`, for each row build
  `{match_id, video_folder, youtube_url, winner_name, loser_name, num_sets}`, then within
  `set/<video_folder>/set{N}.csv` treat stroke rows' `player == 'A'` as `winner_name` and
  `player == 'B'` as `loser_name` for that match only (A/B assignment is match-scoped, not global).
- Full list of (match_id, video_folder, url) for all 44 matches is in
  `training/data/raw/shuttleset/set/match.csv` itself (not duplicated here to avoid drift with
  the source-of-truth file). Sample of first 4 rows and the null-URL row is quoted in (a) above.

## (e) fps assumptions

**fps is NOT a single global constant for the dataset — it is per-match, either 25 or 30 fps.**
Measured by computing `frame_num / seconds(time)` (median per file, excluding `time == 0`) for
all 104 set CSVs:
```
Counter({30: 58, 25: 46})
```
i.e. 58 of the 104 set-files are ~30fps broadcasts, 46 are ~25fps. This split is **consistent
within a match** (all sets of the same match share the same fps — spot-checked across all sets
of several 3-set matches) but differs *between* matches (presumably reflecting different original
broadcast frame rates / regional TV feeds). **Task 6's adapter must derive fps per match from the
data itself (e.g. by fitting `frame_num` vs `seconds(time)` for that match's rows), not assume a
fixed 25 or 30 fps for the whole dataset.** Do not hardcode either value.

Example measurements (median frame_num/seconds per file, real output):
```
.../An_Se_Young_Pornpawee_Chochuwong_TOYOTA_THAILAND_OPEN_2021_QuarterFinals/set1.csv  30.016
.../An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals/set1.csv      30.013
.../Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals/set1.csv     25.010
.../Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final/set1.csv     25.007
```

## (f) Player identity per stroke + rally winner encoding

- **Per-stroke player identity**: the `player` column holds `'A'` or `'B'` only — never a real
  name. Per the dataset README, `'A'` is a match-scoped alias for whichever real player *won that
  match overall* (all sets), and `'B'` for the loser — **this mapping is fixed for the whole
  match, all sets/rallies within it use the same A/B<->name assignment.** To recover the real
  name for a given stroke: join the stroke row's file path (which encodes the match folder) to
  `match.csv.video`, then map `player=='A' -> match.csv.winner`, `player=='B' -> match.csv.loser`.
  There is no per-rally or per-set re-assignment of A/B (not verified by re-simulating full
  matches against known results, but the README is explicit and internally consistent — `player`
  value_counts across the whole dataset are nearly balanced: A=18283, B=18201, as expected for two
  players alternating strokes across many matches of varying length).
- **Rally winner**: `getpoint_player` column, `'A'`/`'B'`, populated only on the rally's final
  stroke row (see (c) for the 174-rally exception where it's missing entirely, and the 1-row
  quirk where it appears mid-rally). Use the same A-vs-B, real-name resolution rule as for
  `player`. `getpoint_player` value_counts: A=1968, B=1542 (NaN=32974, i.e. one non-null per
  rally roughly, matching the ~3509 rallies that got an outcome recorded).
- There is no separate "server" identity column that reliably names who served (the `server`
  int column is a weak rally-position marker, not a player identity — see (a)). To know who
  served a given rally, look at `player` on the `ball_round==1` row of that rally.

## Miscellaneous / gotchas for Task 6

- `ball_round`, `roundscore_A`, `roundscore_B` etc. load as `float64` even though they're
  integer-valued (pandas default on columns with any float-typed sibling values in the CSV
  reader's type inference) — cast explicitly (`.astype(int)`) rather than assuming int dtype
  from `pd.read_csv` defaults.
- `hit_height`/`landing_height`/`aroundhead`/`backhand`/`flaw` all use "NaN means false/absent,
  1.0 means true" for the boolean-ish columns, and 1/2 for the height columns — there is no
  explicit 0 anywhere. Do not `.fillna(0)` blindly if you need to distinguish "not annotated"
  from "definitely false" for `aroundhead`/`backhand`/`flaw` (both are legitimately ambiguous in
  this dataset since NaN could mean either "not applicable" or "not recorded").
- Directory/file names contain spaces and non-ASCII characters in a few cases (e.g.
  `"Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals"` has a literal space
  before `Indonesia`) — always use `glob`/`Path` rather than assuming underscore-only names when
  writing loaders.

## What this task did NOT verify (flag for Task 6 / later review)

- `downcourt` column semantics in `match.csv` (0/1, guessed as camera-side flag, unconfirmed).
- `flaw` column semantics beyond "sparse binary flag, undocumented".
- Whether `hit_area`/`landing_area` grid numbering (1-16) is oriented the same way across all
  matches regardless of `downcourt` value, or needs a flip.
- The `CoachAI-Challenge-IJCAI2023/ShuttleSet22/` superset directory in the same repo was noticed
  but not inspected column-by-column — do not assume it matches this schema exactly if it's used
  later.
