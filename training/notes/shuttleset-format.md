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
containing a larger superset (more 2022-era matches, `set/<match_folder>/set{N}.csv` — the same
directory convention as the dataset copied here). We did NOT copy it — out of scope for this
task. **Correction/clarification: whether `ShuttleSet22` has its own top-level `match.csv`/
`homography.csv` (analogous to this dataset's) is UNKNOWN, not confirmed absent** — the only
listing done during this task was a directory-only `find -type d` at shallow depth, which cannot
prove those files don't exist at the top level; we simply never checked for files there. Treat
`ShuttleSet22`'s schema (including whether it has match-level metadata files) as fully
uninspected, not as "known to differ", if it's considered for later use.

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
  rim, 2=above, matching landing_height's convention below). **NaN rule, verified by full groupby
  over all 36,484 rows (not spot checks) — the "NaN ⇔ first shot of rally" hypothesis is only a
  strong correlation, not an exact rule:**
  ```
  col=hit_area:   NaN&first(TP)=3644  NaN-but-not-first(FP)=1059  first-but-not-NaN(FN)=40   not-NaN&not-first(TN)=31741
  col=hit_x:      NaN&first(TP)=3646  NaN-but-not-first(FP)=1103  first-but-not-NaN(FN)=38   not-NaN&not-first(TN)=31697
  col=hit_y:      NaN&first(TP)=3646  NaN-but-not-first(FP)=1103  first-but-not-NaN(FN)=38   not-NaN&not-first(TN)=31697
  col=hit_height: NaN&first(TP)=0     NaN-but-not-first(FP)=42    first-but-not-NaN(FN)=3684 not-NaN&not-first(TN)=32758
  ```
  Reading this: ~99% of the 3,684 first-of-rally shots have NaN `hit_area`/`hit_x`/`hit_y` (no
  preceding shot to describe an incoming position for — makes semantic sense for a serve), but
  1,059–1,103 rows that are NOT the first shot of their rally are also NaN for these columns
  (~3% of all non-first rows, cause unconfirmed — possibly annotation gaps), and 38–40 first-shot
  rows are NOT NaN (unconfirmed, possibly let/re-serve edge cases). **`hit_height` behaves
  oppositely: it is populated (non-NaN) on literally 100% of first-of-rally shots** (its 42 NaN
  occurrences are all on non-first shots) — do not apply the same "NaN on serve" assumption to
  `hit_height` as to the other three `hit_*` columns.
- `landing_height`: 1 = below net height, 2 = above (per README). **The "NaN ⇔ last stroke of
  rally" hypothesis is REFUTED by full groupby over all 3,683 rallies:**
  ```
  total rallies=3683
  rallies where LAST stroke landing_height IS NaN: 398        (10.8% of rallies)
  rallies where LAST stroke landing_height NOT NaN: 3285       (89.2% of rallies)
  non-last rows with landing_height NaN: 5920                  (93.7% of all 6,318 landing_height NaNs)
  rallies with at least one non-last NaN row: 684
  ```
  In other words the *opposite* of the naive hypothesis is true: only 6.3% of `landing_height`
  NaNs occur on the rally-ending stroke; the large majority (93.7%) occur on non-final strokes.
  There is no clean structural rule tying `landing_height` NaN to rally position. Breaking it down
  by shot `type` also shows no single strong correlate — every shot type has a NaN rate somewhere
  in the 10%–30% band (highest: `未知球種` 29.5%, `小平球` 26.5%; lowest: `撲球` 10.4%, `發長球`
  11.0%), consistent with genuine sporadic missing/unannotated data rather than a deterministic
  encoding. **Conclusion: treat `landing_height` NaN as "not annotated for this stroke", full
  stop — do not build rally-boundary or serve-detection logic on top of it.**
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

## Canonical 8-class stroke-type mapping (CONTROLLER RULING)

Task 6 needs a smaller canonical action-class vocabulary than the 19 raw Chinese `type` values.
**The controller mandated the following exact 19→8 mapping** (recorded here as a ruling, not a
task-owner judgment call):

```
殺球 → smash        點扣 → smash        撲球 → smash
長球 → clear
切球 → drop          過度切球 → drop
網前球 → net         放小球 → net        擋小球 → net        勾球 → net
挑球 → lift          防守回挑 → lift
平球 → drive         小平球 → drive       推球 → drive        防守回抽 → drive
後場抽平球 → drive (addendum, see below)
發短球 → serve       發長球 → serve
未知球種 → DROPPED (map to None; log/count, do not silently discard)
```

**Addendum (controller ruling, post-review):** `後場抽平球` ("back-court drive", 473 occurrences,
flagged as unmapped below when this section was first written) has been explicitly ruled by the
controller to map to `drive`, alongside `平球`/`小平球`/`推球`/`防守回抽`. This closes the
19-value coverage gap noted below; the mapping table above and the per-class counts have been
updated accordingly. The "flag, not a guess" caveat immediately below is retained as a historical
record of why this wasn't guessed at initially, not as a live open question.

**Flag, not a guess: `網前球` never appears in the real data.** Cross-checking the controller's
19 mapping keys against the 19 real `type` values found in the CSVs (§(b)) shows a mismatch:
- The controller's list includes `網前球` ("net-front shot", generic), which is **absent from
  the actual 19-value vocabulary** in this dataset.
- The real data's 19th value that the controller's list does **not** cover is `後場抽平球`
  ("back-court drive", 473 occurrences, 1.3% of all strokes) — present in every inspection in
  §(b) but not addressable by any key in the controller's mapping.

At the time this section was first written, `後場抽平球` was left UNMAPPED pending an explicit
controller decision (not silently folded into `drive`, even though "back-court drive" sounds
adjacent to that bucket, since that would have been guessing at a ruling that didn't actually
cover it). **Resolved by the addendum above: `後場抽平球` → `drive`.** The counts below reflect
that resolution (RESOLVED, no longer an open unmapped-type gap).

**Per-canonical-class counts, applying the ruling + addendum (real value_counts, all 104
files, 36,484 rows):**
```
canonical
net      11281
lift      5632
smash     4746
drive     4572
drop      3500
clear     2922
serve     2424
```
Plus:
- `未知球種` (DROPPED per ruling): **1,407** rows -> mapped to `None`, logged, excluded from the
  8-class counts above.
- Sum check: 11281+5632+4746+4572+3500+2922+2424 (mapped, total 35077) + 1407 (dropped) =
  36,484 = total row count. Every row accounted for, none double-counted; the 19->8 mapping now
  has full coverage of the real vocabulary (`未知球種` dropped by design, all other 18 mapped).

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
  where `fps` is per-match (25, 29.97, or 30 — see (e) for the corrected, bias-free derivation
  method; a naive per-row ratio like the example below is biased high by `time`'s 1-second
  truncation and should not be used to resolve fps — use the whole-match regression in (e) instead).
- Time is NOT in seconds as a raw float anywhere in this dataset — it's the `H:MM:SS` string, and
  the frame-accurate signal is `frame_num`. Use `frame_num` for anything that needs to be
  precise; only use `time` for eyeballing/debugging.

Example (illustrative only — do not use single-row ratios to resolve fps, see (e)):
`time = "0:07:39"` (459 s), `frame_num = 11496.0` -> naive ratio 11496/459 = 25.04, from a match
whose whole-match-regressed fps is 25.0. Another row from a different match: `time = "00:06:57"`
(417 s), `frame_num = 12510.0` -> naive ratio 12510/417 = 30.0, from a match whose regressed fps
is 30.0.

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

**Full 44-row match -> video table, reproduced here in full (not just `match.csv`) because the DVC
remote/push was skipped by controller ruling — `training/data/raw/shuttleset` currently exists
only on this machine's local DVC cache, so this table is the only copy that survives a fresh clone
until someone runs `dvc push` per `NEXT-STEPS.md`.** `resolved_fps` is the per-match fps derived
in §(e) (nominal value; raw regression slope in parentheses) — Task 6/7 should use this column
directly instead of re-deriving it, until video is downloaded and reconciled per the
"Label-to-video alignment" subsection below.

| id | video_folder | youtube_id | url | winner | loser | sets | resolved_fps (raw slope) |
|---|---|---|---|---|---|---|---|
| 1 | `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals` | O669aZhH0LI | https://www.youtube.com/watch?v=O669aZhH0LI | Kento MOMOTA | CHOU Tien Chen | 3 | 25.0 (24.99988) |
| 2 | `CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage` | -aOI9_JxoWc | https://www.youtube.com/watch?v=-aOI9_JxoWc | CHEN Long | CHOU Tien Chen | 2 | 25.0 (24.99973) |
| 3 | `Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final` | eugfCRwSBJo | https://www.youtube.com/watch?v=eugfCRwSBJo | Kento MOMOTA | CHOU Tien Chen | 2 | 25.0 (25.00044) |
| 4 | `CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal` | y6QbtrTV-K0 | https://www.youtube.com/watch?v=y6QbtrTV-K0 | CHEN Long | CHOU Tien Chen | 2 | 25.0 (24.99975) |
| 5 | `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2018_Finals` | xhUi2KpmVkI | https://www.youtube.com/watch?v=xhUi2KpmVkI | Kento MOMOTA | CHOU Tien Chen | 3 | 25.0 (24.99981) |
| 6 | `Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals` | SYyvDrUgClc | https://www.youtube.com/watch?v=SYyvDrUgClc | Kento MOMOTA | CHOU Tien Chen | 3 | 25.0 (24.99989) |
| 7 | `Kento_MOMOTA_CHOU_Tien_Chen_Malaysia_Open_2018_QuarterFinals` | Y0tCJ6DWXKM | https://www.youtube.com/watch?v=Y0tCJ6DWXKM | Kento MOMOTA | CHOU Tien Chen | 2 | 25.0 (24.99983) |
| 8 | `CHOU_Tien_Chen_Anders_ANTONSEN_Fuzhou_Open_2019_Semi-finals` | 32j2Tg64Zbg | https://www.youtube.com/watch?v=32j2Tg64Zbg | CHOU Tien Chen | Anders ANTONSEN | 2 | 25.0 (24.99969) |
| 9 | `CHOU_Tien_Chen_Jonatan_CHRISTIE_Sudirman_Cup_2019_Quarter-finals` | 8E98Gpk-fOM | https://www.youtube.com/watch?v=8E98Gpk-fOM | CHOU Tien Chen | Jonatan CHRISTIE | 2 | 25.0 (25.00006) |
| 10 | `CHOU_Tien_Chen_NG_Ka_Long_Angus_Sudirman_Cup_2019_Group_Stage` | li1sbr6S34g | https://www.youtube.com/watch?v=li1sbr6S34g | CHOU Tien Chen | NG Ka Long Angus | 2 | 25.0 (24.99936) |
| 11 | `CHOU_Tien_Chen_Jonatan_CHRISTIE_Indonesia_Open_2019_Quarter-finals` | yD6WKVqsAKc | https://www.youtube.com/watch?v=yD6WKVqsAKc | CHOU Tien Chen | Jonatan CHRISTIE | 3 | 25.0 (24.99997) |
| 12 | `NG_Ka_Long_Angus_SHI_Yu_Qi_Thailand_Masters_2020_SemiFinals` | **NULL** | **NULL** | NG Ka Long Angus | SHI Yuqi | 2 | 29.97 (29.97047) |
| 13 | `Kento_MOMOTA_Viktor_AXELSEN_Malaysia_Masters_2020_Finals` | boQC4J4E1ZQ | https://www.youtube.com/watch?v=boQC4J4E1ZQ | Kento MOMOTA | Viktor AXELSEN | 2 | 25.0 (25.00037) |
| 14 | `Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals` | 5W6txLGZ1Rs | https://www.youtube.com/watch?v=5W6txLGZ1Rs | Anders ANTONSEN | Jonatan CHRISTIE | 3 | 25.0 (25.00011) |
| 15 | `Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final` | yu9oyMXRGHY | https://www.youtube.com/watch?v=yu9oyMXRGHY | Anthony Sinisuka GINTING | Anders ANTONSEN | 3 | 25.0 (24.99997) |
| 16 | `Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals` | 5kS_a7vS5xI | https://www.youtube.com/watch?v=5kS_a7vS5xI | Anthony Sinisuka GINTING | Viktor AXELSEN | 2 | 25.0 (24.99947) |
| 17 | `NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals` | G9r400zkkz8 | https://www.youtube.com/watch?v=G9r400zkkz8 | NG Ka Long Angus | Jonatan CHRISTIE | 3 | 25.0 (25.00006) |
| 18 | `Viktor_AXELSEN _SHI_Yu_Qi_All_England_Open_2020_QuarterFinals` | 8lHAsyRhYYQ | https://www.youtube.com/watch?v=8lHAsyRhYYQ | Viktor AXELSEN | SHI Yuqi | 2 | 25.0 (24.99956) |
| 19 | `Viktor_AXELSEN_CHEN_Long_Malaysia_Masters_2020_QuarterFinals` | YmW83aQFADg | https://www.youtube.com/watch?v=YmW83aQFADg | Viktor AXELSEN | CHEN Long | 3 | 25.0 (25.00008) |
| 20 | `Viktor_AXELSEN_NG_Ka_Long_Angus_Malaysia_Masters_2020_SemiFinals` | 4e3JJ4rvT3Q | https://www.youtube.com/watch?v=4e3JJ4rvT3Q | Viktor AXELSEN | NG Ka Long Angus | 2 | 25.0 (25.00025) |
| 21 | `An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals` | gloiZ_gTJaE | https://www.youtube.com/watch?v=gloiZ_gTJaE | An Se Young | Ratchanok INTANON | 2 | 30.0 (29.99993) |
| 22 | `Mia_Blichfeldt_Busanan_Ongbamrungphan_YONEX_Thailand_Open_2021_QuarterFinals` | 8u_UHCnYSkk | https://www.youtube.com/watch?v=8u_UHCnYSkk | Mia BLICHFELDT | Busanan ONGBAMRUNGPHAN | 2 | 30.0 (29.99927) |
| 23 | `Ng_Ka_Long_Angus_Lee_Cheuk_Yiu_YONEX_Thailand_Open_2021_QuarterFinals` | rSK9Qx8LapE | https://www.youtube.com/watch?v=rSK9Qx8LapE | NG Ka Long Angus | LEE Cheuk Yiu | 2 | 30.0 (29.99910) |
| 24 | `Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals` | NlDrJyQUTSI | https://www.youtube.com/watch?v=NlDrJyQUTSI | Anthony Sinisuka GINTING | Rasmus GEMKE | 3 | 30.0 (29.99971) |
| 25 | `Carolina_Marin_Supanida_Katethong_YONEX_Thailand_Open_2021_QuarterFinals` | FZPrpoGdyHI | https://www.youtube.com/watch?v=FZPrpoGdyHI | Carolina MARIN | Supanida KATETHONG | 2 | 30.0 (29.99990) |
| 26 | `Viktor_Axelsen_Jonatan_Christie_YONEX_Thailand_Open_2021_QuarterFinals` | RRI_k2KZgOM | https://www.youtube.com/watch?v=RRI_k2KZgOM | Viktor AXELSEN | Jonatan CHRISTIE | 2 | 30.0 (29.99990) |
| 27 | `Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals` | HTBf9wFL0mk | https://www.youtube.com/watch?v=HTBf9wFL0mk | Viktor AXELSEN | Anthony Sinisuka GINTING | 3 | 30.0 (30.00040) |
| 28 | `Viktor_Axelsen_Ng_Ka_Long_Angus_YONEX_Thailand_Open_2021_Finals` | IuXmsimDOW8 | https://www.youtube.com/watch?v=IuXmsimDOW8 | Viktor AXELSEN | NG Ka Long Angus | 2 | 30.0 (30.00049) |
| 29 | `An_Se_Young_Pornpawee_Chochuwong_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | TXT-qlniM90 | https://www.youtube.com/watch?v=TXT-qlniM90 | An Se Young | Pornpawee CHOCHUWONG | 2 | 30.0 (29.99953) |
| 30 | `Anders_Antonsen_Sameer_Verma_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | YP8YlZkrQq8 | https://www.youtube.com/watch?v=YP8YlZkrQq8 | Anders ANTONSEN | Sameer VERMA | 3 | 30.0 (30.00006) |
| 31 | `Carolina_Marin_Neslihan_Yigit_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | gJ_KHu0EC6I | https://www.youtube.com/watch?v=gJ_KHu0EC6I | Carolina MARIN | Neslihan YIGIT | 2 | 30.0 (29.99977) |
| 32 | `Hans-Kristian_Solberg_Vittinghus_Lee_Cheuk_Yu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | ROAnTfC_8zA | https://www.youtube.com/watch?v=ROAnTfC_8zA | Hans-Kristian Solberg VITTINGHUS | LEE Cheuk Yiu | 3 | 30.0 (29.99998) |
| 33 | `Viktor_Axelsen_Liew_Daren_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | OzRtd3D0hEo | https://www.youtube.com/watch?v=OzRtd3D0hEo | Viktor AXELSEN | LIEW Daren | 2 | 30.0 (29.99975) |
| 34 | `Ratchanok_Intanon_Pusarla_V._Sindhu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | o51ingUOU20 | https://www.youtube.com/watch?v=o51ingUOU20 | Ratchanok INTANON | PUSARLA V. Sindhu | 2 | 30.0 (30.00116) |
| 35 | `Carolina_Marin_An_Se_Young_TOYOTA_THAILAND_OPEN_2021_SemiFinals` | XmJ-OdVFQtk | https://www.youtube.com/watch?v=XmJ-OdVFQtk | Carolina MARIN | An Se Young | 2 | 30.0 (29.99975) |
| 36 | `Hans-Kristian_Solberg_Vittinghus_Anders_Antonsen_TOYOTA_THAILAND_OPEN_2021_SemiFinals` | D27aAZvuRTw | https://www.youtube.com/watch?v=D27aAZvuRTw | Hans-Kristian Solberg VITTINGHUS | Anders ANTONSEN | 2 | 30.0 (29.99980) |
| 37 | `Viktor_Axelsen_Hans-Kristian_Solberg_VIittinghus_TOYOTA_THAILAND_OPEN_2021_Finals` | 4rQUHv9oGpI | https://www.youtube.com/watch?v=4rQUHv9oGpI | Viktor AXELSEN | Hans-Kristian Solberg VITTINGHUS | 2 | 30.0 (30.00017) |
| 38 | `Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | OH6dnTXhZy4 | https://www.youtube.com/watch?v=OH6dnTXhZy4 | An Se Young | Carolina MARIN | 3 | 30.0 (30.00072) |
| 39 | `Anthony_Sinisuka_Ginting_Lee_Zii_Jia_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | XYY6YXv6Nss | https://www.youtube.com/watch?v=XYY6YXv6Nss | Anthony Sinisuka GINTING | LEE Zii Jia | 3 | 30.0 (30.00033) |
| 40 | `Evgeniya_Kosetskaya_Michelle_Li_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | IDSr0z5f52k | https://www.youtube.com/watch?v=IDSr0z5f52k | Evgeniya KOSETSKAYA | Michelle LI | 2 | 30.0 (30.00004) |
| 41 | `Ng_Ka_Long_Angus_Kidambi_Srikanth_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | yr2JQTdzNjY | https://www.youtube.com/watch?v=yr2JQTdzNjY | NG Ka Long Angus | KIDAMBI Srikanth | 3 | 30.0 (29.99977) |
| 42 | `Pusarla_V._Sindhu_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | Mawo3l3Hb9E | https://www.youtube.com/watch?v=Mawo3l3Hb9E | PUSARLA V. Sindhu | Pornpawee CHOCHUWONG | 2 | 30.0 (30.00012) |
| 43 | `Carolina_Marin_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_SemiFinals` | vfzkc3lFTdM | https://www.youtube.com/watch?v=vfzkc3lFTdM | Carolina MARIN | Pornpawee CHOCHUWONG | 2 | 30.0 (30.00000) |
| 44 | `Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals` | j7_cjmJDYNU | https://www.youtube.com/watch?v=j7_cjmJDYNU | Anders ANTONSEN | Viktor AXELSEN | 3 | 30.0 (30.00017) |

Generated by joining `match.csv` with a per-match robust fps regression (see §(e) for method);
regenerable by re-running the same query against `training/data/raw/shuttleset/set/match.csv` and
`set/<video_folder>/set*.csv`.

## (e) fps assumptions

**fps is NOT a single global constant for the dataset — it is per-match**, and the naive
per-file-median estimate used in the first pass of this task was **biased HIGH**, because `time`
is floor-truncated to whole seconds (an `H:MM:SS` string has no sub-second resolution) while
`frame_num` is exact — dividing an exact numerator by a truncated-down denominator systematically
inflates the ratio. The earlier estimates (30.016, 30.013, 25.010, 25.007, ...) were all a few
hundredths of a frame/sec too high because of this truncation bias, not because of file-to-file
fps drift.

**Corrected method: robust slope, not per-row ratio.** For each match, pool every stroke row
across all its `set*.csv` files, compute `seconds(time)` (x) and `frame_num` (y), and fit an
ordinary least-squares line `y = slope * x + intercept` across the whole match (not per-file, not
per-row-ratio). The floor-truncation noise on individual `time` values averages out across many
widely-separated (x, y) pairs, so the fitted `slope` is a much tighter estimate of the true fps
than any single row's `frame_num/seconds(time)`. **Then snap `slope` to the nearest of the three
nominal broadcast frame rates `{25, 29.97, 30}`** (30.000 exact and 29.97 = 30000/1001 NTSC are
both plausible for BWF broadcast footage and are within ~0.03 of each other — indistinguishable
from a coarse per-row ratio, which is exactly why the robust whole-match slope matters).

Result across all 44 matches (slope regressed over every stroke in the match, all sets pooled):
```
{25.0: 19, 29.97: 1, 30.0: 24}
```
19 matches resolve cleanly to 25.0 (slopes 24.9994–25.0004), 24 matches resolve cleanly to 30.0
(slopes 29.9991–30.0012), and **exactly one match (id=12, `NG_Ka_Long_Angus_SHI_Yu_Qi_...`, the
same match with the NULL video URL) resolves clearly to 29.97** (slope 29.97047, ~0.03 away from
30.0 — not noise, a real 3rd cluster). The full per-match resolved fps is in the table in §(d).

**Important caveat — 29.97 vs 30 cannot be fully trusted from labels alone.** A slope of 29.97047
is closer to 29.97 than to 30.0, but the two nominal rates differ by only 0.1%, and our estimate
has its own residual noise from `time`'s 1-second quantization even after the whole-match
regression. **Task 6/7 must reconcile the label-derived fps against `ffprobe`'s reported fps on
the actual downloaded video stream before trusting frame-level alignment** — see "Label-to-video
alignment" below. Do not hardcode 25 or 30 (or 29.97) anywhere; always resolve fps per match and
then re-verify against the real video file once downloaded.

### Label-to-video alignment

`frame_num` indexes **the annotators' original copy of the broadcast video**, not necessarily the
exact byte-for-byte file a `yt-dlp` download of the same YouTube URL will produce today. Sources
of potential drift between the label's frame numbering and a freshly downloaded video:
- **Re-uploads / re-encodes**: if the URL points to a re-upload rather than the original capture,
  frame rate, container, or even total frame count can differ (e.g. re-encoded to 30fps from a
  25fps original, or vice versa).
- **Lead-in / intro trimming**: broadcast footage sometimes has intro/replay segments before the
  live coverage begins; if the annotators' copy and the currently-hosted YouTube copy have
  different lead-in lengths, every `frame_num` in the labels will be offset by a constant amount
  from the corresponding frame in the freshly downloaded video.
- **Variable frame rate (VFR) sources**: YouTube re-encodes can occasionally produce VFR outputs
  even from CFR (constant frame rate) sources, which would break a simple `frame_num / fps =
  seconds` mapping entirely for parts of the video.

**Required before any match's video enters training (Task 7/12):**
1. Run `ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,avg_frame_rate
   -of default=noprint_wrappers=1 <downloaded_file>` on the downloaded stream and compare its
   reported fps to the label-derived `resolved_fps` for that match (§(d) table). A mismatch (e.g.
   labels say 25.0 but ffprobe says 29.97) means `frame_num` cannot be used directly as a frame
   index into the downloaded file — a fps-ratio rescaling (and possibly an offset) is needed.
2. Do a **per-match visual spot-check**: pick a `type == '殺球'` (smash) row from that match's
   labels, seek the downloaded video to `frame_num` (adjusted for fps ratio/offset per step 1 if
   needed), and visually confirm a smash is actually happening at or near that frame. If it isn't
   (wrong frame entirely, or off by a large constant), do not trust that match's frame alignment
   for training — flag it and fall back to `time`-based seeking (coarser, but not subject to a
   frame-index/offset mismatch) or exclude the match.
3. Only after both checks pass for a given match should its `frame_num` values be treated as
   directly indexable frame offsets into that match's downloaded video file.

## (f) Player identity per stroke + rally winner encoding

- **Per-stroke player identity**: the `player` column holds `'A'` or `'B'` only — never a real
  name. Per the dataset README, `'A'` is a match-scoped alias for whichever real player *won that
  match overall* (all sets), and `'B'` for the loser — **this mapping is fixed for the whole
  match, all sets/rallies within it use the same A/B<->name assignment.** To recover the real
  name for a given stroke: join the stroke row's file path (which encodes the match folder) to
  `match.csv.video`, then map `player=='A' -> match.csv.winner`, `player=='B' -> match.csv.loser`.

  **This is no longer just an assumption from the README — it has been independently verified**
  with a real test that does not rely on the README's claim or on name-string matching:
  for each of the 44 matches, tally `getpoint_player` counts (`'A'` vs `'B'`) within each
  `set*.csv` file to determine that set's winner (falling back to comparing the final row's
  `roundscore_A` vs `roundscore_B` on the rare set where the tally is exactly tied — this fallback
  fired for 2/104 sets, match ids 13 and 27); then take the letter that won the majority of sets
  in the match (2 of 2, 2 of 3, or 3 of 3) as the match's derived winner-letter. **If `'A'` truly
  always denotes the match winner, this derived letter must equal `'A'` for every one of the 44
  matches — and it does: 44/44.** So the A=winner / B=loser convention is confirmed at 100% and
  can be relied upon in Task 6 without a controller flag. Compressed evidence (the full
  per-match breakdown lived in a now-gitignored scratchpad path, so it's summarized here
  instead): 44/44; per-set letters reproducible via the method above run against
  `training/data/raw/shuttleset/set/*/set*.csv`.
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
- ~~`後場抽平球` (473 rows, 1.3%) has no canonical-class mapping in the controller's ruling~~ —
  RESOLVED: controller addendum maps it to `drive` (see the canonical mapping section above).
- fps resolution for the one 29.97 match (id=12) is based on label-derived regression only; it has
  not been cross-checked against `ffprobe` of an actual downloaded video (that match also has no
  URL, so it cannot be downloaded at all — its fps label should be treated as unconfirmed).
