# ShuttleSense Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deployed badminton match-analysis app: upload a singles clip → rally segmentation + stroke classification by self-trained models → interactive match report (annotated video player, momentum, rally explorer), with pre-analyzed sample matches as the landing page.

**Architecture:** Monorepo with a shared `core/` Python package (feature extraction, smoothing, homography — imported identically by training and serving), a `training/` area (DVC-versioned data, W&B-tracked training CLIs, eval suite, ONNX export), a FastAPI `backend/` (async job worker, ONNX Runtime CPU inference), and a React `frontend/` (canvas overlays driven by JSON tracks). Single-container deploy on Render.

**Tech Stack:** Python 3.11, PyTorch (training only), ONNX Runtime (serving), rtmlib (pose), OpenCV, pandas/pyarrow, DVC + Google Drive, Weights & Biases, FastAPI + SQLite, React + Vite, Docker, Render.

**Spec:** `docs/superpowers/specs/2026-08-26-shuttlesense-design.md`

**Subagent model assignments (per project CLAUDE.md):** implementation subagents → **Sonnet**; review subagents → **Opus**; substantive testing subagents → **Opus**.

## Global Constraints

- Singles matches only; two players per video.
- Stroke classes (canonical, exactly these strings): `clear, smash, drop, net, lift, drive, serve` + background class `none` (8 classes total for the classifier).
- Serving inference is ONNX Runtime **CPU only**; PyTorch appears nowhere in `backend/`.
- Feature code lives ONLY in `core/`; training and serving must import the same functions (train/serve consistency test enforces this).
- All training runs: fixed seed from config; logged to W&B (public project `shuttlesense`); dataset splits are **by match**, never random-by-sample.
- Upload caps: ≤ 95 seconds duration, ≤ 100 MB, extensions `.mp4 .mov .mkv`.
- Landing page must show a fully analyzed sample match with zero user action.
- Git commits: NO AI attribution / co-author trailers (project CLAUDE.md).
- Free tiers only: Colab/Kaggle (GPU), W&B, Google Drive (DVC remote), Render.
- **Spec deviation (documented):** court-relative position features (spec §5) are implemented behind a feature flag but OFF in Phase 1 training — automatic court detection lands in Phase 2. Phase 1 features are pose-space only. Rationale: spec §10 risk table prioritizes an unblocked Phase 1; `core/homography.py` math still ships and is tested now.
- **Momentum data contract:** `RallyInterval.winner` is optional. Sample matches (from ShuttleSet labels) have winners → score-race chart renders. User uploads have `winner=null` → the UI renders the control ribbon (attacking-stroke share) only. No score OCR.

## Repository layout (target state after this plan)

```
shuttlesense/
├── core/
│   ├── pyproject.toml
│   └── shuttlesense_core/
│       ├── __init__.py
│       ├── schemas.py        # Task 1
│       ├── features.py       # Task 2
│       ├── smoothing.py      # Task 3
│       └── homography.py     # Task 4
├── training/
│   ├── requirements.txt
│   ├── dvc.yaml              # Task 8
│   ├── configs/{stroke_tcn.yaml, rally_gru.yaml}
│   ├── data/                 # DVC-tracked; raw/ and processed/
│   ├── notes/shuttleset-format.md   # Task 5 output
│   ├── prepare_shuttleset.py # Task 6
│   ├── extract_poses.py      # Task 7
│   ├── build_windows.py      # Task 8
│   ├── models.py             # Task 9
│   ├── train_stroke.py       # Task 10
│   ├── train_rally.py        # Task 11
│   ├── evaluate.py           # Task 12
│   ├── export_onnx.py        # Task 13
│   └── reports/              # eval markdown reports (committed)
├── backend/
│   ├── requirements.txt
│   ├── app/{config.py, db.py, main.py, routes.py, worker.py, pipeline.py}
│   ├── models/               # pinned ONNX files (downloaded, gitignored)
│   ├── samples/              # pre-analyzed matches (Task 19)
│   └── tests/
├── frontend/                 # Vite + React (Tasks 17–18)
├── scripts/build_samples.py  # Task 19
├── tests/                    # cross-cutting: consistency test (Task 13)
├── Dockerfile                # Task 20
└── render.yaml               # Task 20
```

---

# Part A — Core library

### Task 1: Repo scaffolding + `core` package with schemas

**Files:**
- Create: `core/pyproject.toml`, `core/shuttlesense_core/__init__.py`, `core/shuttlesense_core/schemas.py`
- Create: `requirements-dev.txt`, `.gitignore` (extend), `pytest.ini`
- Test: `core/tests/test_schemas.py`

**Interfaces:**
- Produces: `STROKE_CLASSES: list[str]`, `ALL_CLASSES: list[str]` (strokes + `"none"`), dataclasses `RallyInterval(start_frame:int, end_frame:int, winner:int|None)`, `StrokeEvent(frame:int, player:int, stroke:str, confidence:float)`, `MatchReport(fps:float, width:int, height:int, n_frames:int, rallies:list[RallyInterval], strokes:list[StrokeEvent])` with `to_dict()` / `from_dict()` round-trip. Every later task imports these.

- [ ] **Step 1: Scaffold environment**

```bash
cd /Users/banik/Desktop/Projects/shuttlesense
python3.11 -m venv .venv && source .venv/bin/activate
printf 'pytest>=8\nnumpy>=1.26\nopencv-python-headless>=4.9\n' > requirements-dev.txt
pip install -r requirements-dev.txt
printf '.venv/\n__pycache__/\n*.pyc\nbackend/models/\ntraining/data/\nwandb/\n.dvc/cache\n' >> .gitignore
printf '[pytest]\ntestpaths = core/tests backend/tests tests\n' > pytest.ini
```

- [ ] **Step 2: Write the failing test** — `core/tests/test_schemas.py`

```python
from shuttlesense_core.schemas import (
    ALL_CLASSES, STROKE_CLASSES, MatchReport, RallyInterval, StrokeEvent,
)

def test_classes():
    assert STROKE_CLASSES == ["clear", "smash", "drop", "net", "lift", "drive", "serve"]
    assert ALL_CLASSES == STROKE_CLASSES + ["none"]

def test_report_roundtrip():
    r = MatchReport(
        fps=15.0, width=1280, height=720, n_frames=900,
        rallies=[RallyInterval(30, 300, winner=0), RallyInterval(360, 700, winner=None)],
        strokes=[StrokeEvent(frame=45, player=0, stroke="serve", confidence=0.91)],
    )
    d = r.to_dict()
    assert d["rallies"][1]["winner"] is None
    r2 = MatchReport.from_dict(d)
    assert r2 == r
```

- [ ] **Step 3: Run to verify failure** — `pip install -e ./core` will fail (no pyproject yet); that is the failure. Then create the package:

`core/pyproject.toml`:

```toml
[project]
name = "shuttlesense-core"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["shuttlesense_core*"]
```

`core/shuttlesense_core/schemas.py`:

```python
from __future__ import annotations
from dataclasses import asdict, dataclass, field

STROKE_CLASSES = ["clear", "smash", "drop", "net", "lift", "drive", "serve"]
NONE_CLASS = "none"
ALL_CLASSES = STROKE_CLASSES + [NONE_CLASS]

@dataclass
class RallyInterval:
    start_frame: int
    end_frame: int
    winner: int | None = None

@dataclass
class StrokeEvent:
    frame: int
    player: int
    stroke: str
    confidence: float

@dataclass
class MatchReport:
    fps: float
    width: int
    height: int
    n_frames: int
    rallies: list[RallyInterval] = field(default_factory=list)
    strokes: list[StrokeEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchReport":
        return cls(
            fps=d["fps"], width=d["width"], height=d["height"], n_frames=d["n_frames"],
            rallies=[RallyInterval(**r) for r in d["rallies"]],
            strokes=[StrokeEvent(**s) for s in d["strokes"]],
        )
```

Create empty `core/shuttlesense_core/__init__.py` and `core/tests/__init__.py`.

- [ ] **Step 4: Install and run tests to verify pass**

```bash
pip install -e ./core
pytest core/tests/test_schemas.py -v   # expect 2 passed
```

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: repo scaffolding and core schemas"`

---

### Task 2: `core.features` — pose normalization and windowing

**Files:**
- Create: `core/shuttlesense_core/features.py`
- Test: `core/tests/test_features.py`

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces (exact signatures, used by Tasks 8 and 15):
  - `normalize_pose(kpts: np.ndarray) -> np.ndarray` — `(17,2)` → `(17,2)`, hip-centered, torso-scaled. COCO-17 keypoint order; hips = indices 11,12; shoulders = 5,6.
  - `stroke_window(kpts_seq: np.ndarray, center: int, w: int = 30) -> np.ndarray` — `(T,17,2)` → `(w, 68)` float32: per-frame flattened normalized positions (34) + frame-to-frame velocity (34). Out-of-range indices clamp to clip edges.
  - `rally_frame_features(kpts_all: np.ndarray, scores_all: np.ndarray) -> np.ndarray` — `(T,2,17,2)` and `(T,2,17)` → `(T,4)` float32: per-player motion energy (mean |Δ| of normalized keypoints, 0 for frame 0) and mean keypoint confidence.
  - Constants: `WINDOW = 30`, `FEAT_DIM = 68`, `COCO_EDGES: list[tuple[int,int]]` (for frontend overlay export).

- [ ] **Step 1: Write the failing tests** — `core/tests/test_features.py`

```python
import numpy as np
from shuttlesense_core.features import (
    FEAT_DIM, WINDOW, normalize_pose, rally_frame_features, stroke_window,
)

def _pose(cx=100.0, cy=200.0, torso=50.0):
    k = np.zeros((17, 2), dtype=np.float32)
    k[11] = [cx - 10, cy]; k[12] = [cx + 10, cy]          # hips
    k[5] = [cx - 12, cy - torso]; k[6] = [cx + 12, cy - torso]  # shoulders
    k[9] = [cx - 30, cy - torso - 20]                      # a wrist
    return k

def test_normalize_translation_and_scale_invariant():
    a = normalize_pose(_pose(100, 200, 50))
    b = normalize_pose(_pose(500, 900, 50) )
    c = normalize_pose(_pose(100, 200, 100) * 1.0)
    np.testing.assert_allclose(a, b, atol=1e-5)          # translation invariant
    hip_mid = (a[11] + a[12]) / 2
    np.testing.assert_allclose(hip_mid, [0, 0], atol=1e-5)  # centered

def test_stroke_window_shape_and_edge_clamp():
    seq = np.stack([_pose(100 + t, 200) for t in range(40)])  # (40,17,2)
    w = stroke_window(seq, center=2)                          # near left edge
    assert w.shape == (WINDOW, FEAT_DIM)
    assert w.dtype == np.float32

def test_rally_features_motion_vs_still():
    still = np.stack([_pose()] * 20)
    moving = np.stack([_pose(100 + 5 * t, 200) for t in range(20)])
    kpts = np.stack([moving, still], axis=1)              # (20,2,17,2)
    scores = np.ones((20, 2, 17), dtype=np.float32)
    f = rally_frame_features(kpts, scores)
    assert f.shape == (20, 4)
    assert f[5:, 0].mean() > f[5:, 1].mean()              # mover has more energy
```

- [ ] **Step 2: Run to verify failure** — `pytest core/tests/test_features.py -v` → `ModuleNotFoundError`/`ImportError`.

- [ ] **Step 3: Implement** — `core/shuttlesense_core/features.py`

```python
from __future__ import annotations
import numpy as np

WINDOW = 30
FEAT_DIM = 68
HIP_L, HIP_R, SH_L, SH_R = 11, 12, 5, 6
COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

def normalize_pose(kpts: np.ndarray) -> np.ndarray:
    center = (kpts[HIP_L] + kpts[HIP_R]) / 2.0
    shoulder_mid = (kpts[SH_L] + kpts[SH_R]) / 2.0
    torso = float(np.linalg.norm(shoulder_mid - center))
    scale = torso if torso > 1e-6 else 1.0
    return ((kpts - center) / scale).astype(np.float32)

def stroke_window(kpts_seq: np.ndarray, center: int, w: int = WINDOW) -> np.ndarray:
    T = kpts_seq.shape[0]
    idx = np.clip(np.arange(center - w // 2, center + w - w // 2), 0, T - 1)
    norm = np.stack([normalize_pose(kpts_seq[i]) for i in idx])     # (w,17,2)
    pos = norm.reshape(w, -1)                                       # (w,34)
    vel = np.diff(pos, axis=0, prepend=pos[:1])                     # (w,34)
    return np.concatenate([pos, vel], axis=1).astype(np.float32)    # (w,68)

def rally_frame_features(kpts_all: np.ndarray, scores_all: np.ndarray) -> np.ndarray:
    T, P = kpts_all.shape[0], kpts_all.shape[1]
    out = np.zeros((T, 2 * P), dtype=np.float32)
    for p in range(P):
        norm = np.stack([normalize_pose(kpts_all[t, p]) for t in range(T)])
        disp = np.abs(np.diff(norm, axis=0)).mean(axis=(1, 2))      # (T-1,)
        out[1:, p] = disp
        out[:, P + p] = scores_all[:, p].mean(axis=1)
    return out
```

- [ ] **Step 4: Run tests to verify pass** — `pytest core/tests/test_features.py -v` → 3 passed.
- [ ] **Step 5: Commit** — `git commit -am "feat: core pose features (normalization, stroke windows, rally features)"`

---

### Task 3: `core.smoothing` — intervals from probabilities + event NMS

**Files:**
- Create: `core/shuttlesense_core/smoothing.py`
- Test: `core/tests/test_smoothing.py`

**Interfaces:**
- Produces (used by Tasks 8, 11, 15):
  - `probs_to_intervals(p: np.ndarray, threshold: float = 0.5, min_len: int = 30, merge_gap: int = 15) -> list[tuple[int, int]]` — per-frame play probabilities → merged `(start, end)` frame intervals (end exclusive).
  - `suppress_events(events: list[dict], min_gap: int) -> list[dict]` — events have keys `frame:int, confidence:float`; greedy NMS keeping highest confidence, dropping any event strictly closer than `min_gap` frames to a kept one (distance exactly `min_gap` survives); ties broken by `(-confidence, frame)` so results are input-order-independent; returns sorted by frame.

- [ ] **Step 1: Write the failing tests** — `core/tests/test_smoothing.py`

```python
import numpy as np
from shuttlesense_core.smoothing import probs_to_intervals, suppress_events

def test_intervals_merge_and_minlen():
    p = np.zeros(200)
    p[10:60] = 0.9        # run 1
    p[70:130] = 0.9       # run 2, gap of 10 (< merge_gap) -> merged with run 1
    p[150:160] = 0.9      # run 3, len 10 (< min_len) -> dropped
    out = probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15)
    assert out == [(10, 130)]

def test_nms_keeps_best_and_orders():
    ev = [
        {"frame": 100, "confidence": 0.7},
        {"frame": 104, "confidence": 0.9},   # wins over 100
        {"frame": 130, "confidence": 0.6},
    ]
    out = suppress_events(ev, min_gap=8)
    assert [e["frame"] for e in out] == [104, 130]
```

- [ ] **Step 2: Run to verify failure** — `pytest core/tests/test_smoothing.py -v` → import error.

- [ ] **Step 3: Implement** — `core/shuttlesense_core/smoothing.py`

```python
from __future__ import annotations
import numpy as np

def probs_to_intervals(p, threshold=0.5, min_len=30, merge_gap=15):
    mask = np.asarray(p) >= threshold
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(mask)))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]

def suppress_events(events, min_gap):
    kept = []
    for ev in sorted(events, key=lambda e: -e["confidence"]):
        if all(abs(ev["frame"] - k["frame"]) >= min_gap for k in kept):
            kept.append(ev)
    return sorted(kept, key=lambda e: e["frame"])
```

- [ ] **Step 4: Run tests to verify pass**, then **Step 5: Commit** — `git commit -am "feat: rally interval smoothing and stroke-event NMS"`

---

### Task 4: `core.homography` — court mapping math (Phase 2 foundation)

**Files:**
- Create: `core/shuttlesense_core/homography.py`
- Test: `core/tests/test_homography.py`
- Modify: `core/pyproject.toml` (add `opencv-python-headless>=4.9` to dependencies)

**Interfaces:**
- Produces: `COURT_W = 6.1`, `COURT_L = 13.4` (meters, doubles-width court incl. lines); `fit_homography(img_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray` — ≥4 correspondences → `(3,3)` float64 H mapping image px → court meters; `to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray` — `(N,2)` px → `(N,2)` meters.

- [ ] **Step 1: Write the failing test** — `core/tests/test_homography.py`

```python
import numpy as np
from shuttlesense_core.homography import COURT_L, COURT_W, fit_homography, to_court

def test_roundtrip_known_projection():
    court = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float64)
    img = np.array([[300, 600], [980, 600], [1180, 100], [100, 100]], dtype=np.float64)
    H = fit_homography(img, court)
    mapped = to_court(H, img)
    np.testing.assert_allclose(mapped, court, atol=1e-6)
    center = to_court(H, np.array([[640.0, 350.0]]))
    assert 0 < center[0, 0] < COURT_W and 0 < center[0, 1] < COURT_L
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — `core/shuttlesense_core/homography.py`

```python
from __future__ import annotations
import cv2
import numpy as np

COURT_W = 6.1
COURT_L = 13.4

def fit_homography(img_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(
        np.asarray(img_pts, dtype=np.float64), np.asarray(court_pts, dtype=np.float64)
    )
    if H is None:
        raise ValueError("homography fit failed")
    return H

def to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)
```

- [ ] **Step 4: `pip install -e ./core && pytest core/tests -v`** → all core tests pass.
- [ ] **Step 5: Commit** — `git commit -am "feat: court homography math"`

---

# Part B — Data & training

### Task 5: DVC setup + ShuttleSet acquisition and format notes

**Files:**
- Create: `training/requirements.txt`, `training/notes/shuttleset-format.md`, `training/data/` (DVC-tracked)
- Modify: `.gitignore` (DVC handles `training/data`)

**Interfaces:**
- Produces: `training/data/raw/shuttleset/` containing the ShuttleSet label CSVs; `training/notes/shuttleset-format.md` documenting the ACTUAL columns, stroke-type vocabulary with counts, and video-source URLs. Task 6 is written against this note file.

This is the week-1 de-risking spike from spec §10. ShuttleSet ships stroke-level CSVs but NOT videos — videos are broadcast matches referenced by URL and fetched separately with `yt-dlp` for research use.

- [ ] **Step 1: Install tooling**

```bash
printf 'dvc[gdrive]>=3.48\npandas>=2.2\npyarrow>=15\nyt-dlp>=2025.1.1\n' > training/requirements.txt
pip install -r training/requirements.txt
dvc init && git commit -m "chore: init DVC"
```

- [ ] **Step 2: Configure the DVC remote** — create a Google Drive folder named `shuttlesense-dvc` in the user's Drive, copy its folder ID from the URL, then:

```bash
dvc remote add -d gdrive gdrive://<FOLDER_ID_FROM_URL>
git commit -am "chore: DVC gdrive remote"
```

(First `dvc push` triggers a browser OAuth flow — the user must complete it; pause and ask if running unattended.)

- [ ] **Step 3: Fetch ShuttleSet labels**

```bash
mkdir -p training/data/raw
git clone --depth 1 https://github.com/wywyWang/CoachAI-Projects /tmp/coachai
# Locate the ShuttleSet dataset directory inside the clone (search for stroke-level CSVs):
find /tmp/coachai -iname '*.csv' | head -20
cp -r <located_shuttleset_dir> training/data/raw/shuttleset
```

If the CSVs are not in that repo (layout changes), search GitHub for `ShuttleSet` (author: wywyWang) and use the dedicated dataset repo. This step is verified by Step 4 producing a non-empty inventory.

- [ ] **Step 4: Inspect and document the format** — run and paste results into `training/notes/shuttleset-format.md`:

```bash
python - <<'EOF'
import pandas as pd, glob
files = glob.glob("training/data/raw/shuttleset/**/*.csv", recursive=True)
print(f"{len(files)} csv files")
df = pd.read_csv(files[0])
print(df.columns.tolist())
print(df.head(10).to_string())
all_types = pd.concat([pd.read_csv(f) for f in files[:50]])
col = [c for c in all_types.columns if c.lower() in ("type", "stroke", "ball_type")][0]
print(all_types[col].value_counts())
EOF
```

The note file must record: (a) exact column names, (b) full stroke-type vocabulary with counts, (c) how rally boundaries and hit frames/times are encoded, (d) how matches map to video files/URLs, (e) fps assumptions. **Do not proceed to Task 6 until this file answers all five.**

- [ ] **Step 5: Track with DVC and commit**

```bash
dvc add training/data/raw/shuttleset
git add training/data/raw/shuttleset.dvc training/notes/shuttleset-format.md training/requirements.txt
git commit -m "data: ShuttleSet labels (DVC) + format notes"
dvc push
```

---

### Task 6: `prepare_shuttleset.py` — adapter to the canonical label schema

**Files:**
- Create: `training/prepare_shuttleset.py`
- Test: `training/tests/test_prepare.py` (+ empty `training/tests/__init__.py`)

**Interfaces:**
- Consumes: raw ShuttleSet CSVs; `training/notes/shuttleset-format.md`.
- Produces: `training/data/processed/labels.parquet` with EXACTLY these columns (the contract for Tasks 7, 8, 12, 19): `match_id:str, video_file:str, fps:float, rally_id:int, hit_frame:int, player:int (0|1), stroke:str (one of STROKE_CLASSES), rally_start_frame:int, rally_end_frame:int, rally_winner:int (0|1)`. Also exposes `convert(raw_df: pd.DataFrame, match_id: str, fps: float) -> pd.DataFrame` for tests, and `STROKE_MAP: dict[str, str]`.

- [ ] **Step 1: Write the failing test with a synthetic raw CSV** — `training/tests/test_prepare.py`. The synthetic fixture mimics the REAL columns documented in Task 5's notes (adjust field names there once, to match reality):

```python
import pandas as pd
from training.prepare_shuttleset import convert
from shuttlesense_core.schemas import STROKE_CLASSES

def _raw():
    return pd.DataFrame({
        "rally": [1, 1, 1, 2, 2],
        "frame_num": [100, 130, 170, 400, 430],
        "player": ["A", "B", "A", "B", "A"],
        "type": ["short service", "lob", "smash", "long service", "net shot"],
        "getpoint_player": ["A", "A", "A", "B", "B"],
    })

def test_convert_schema_and_mapping():
    out = convert(_raw(), match_id="m01", fps=30.0)
    assert set(out.columns) == {
        "match_id", "video_file", "fps", "rally_id", "hit_frame", "player",
        "stroke", "rally_start_frame", "rally_end_frame", "rally_winner",
    }
    assert out["stroke"].isin(STROKE_CLASSES).all()
    r1 = out[out.rally_id == 1]
    assert r1["rally_start_frame"].iloc[0] <= 100 and r1["rally_end_frame"].iloc[0] >= 170
    assert (r1["rally_winner"] == 0).all()     # player A -> 0

def test_unknown_stroke_raises():
    raw = _raw(); raw.loc[0, "type"] = "???"
    try:
        convert(raw, "m01", 30.0)
        assert False, "should raise"
    except KeyError:
        pass
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — `training/prepare_shuttleset.py`. The `STROKE_MAP` below covers common ShuttleSet vocabulary; extend it from the ACTUAL value counts in the notes file until a full run over the real CSVs raises no `KeyError` (unmappable rare types may map to `None` and be dropped with a logged count):

```python
from __future__ import annotations
import argparse, glob, sys
import pandas as pd

STROKE_MAP = {
    "clear": "clear", "smash": "smash", "drop": "drop", "drop shot": "drop",
    "net shot": "net", "net kill": "smash", "net lift": "lift",
    "lob": "lift", "defensive lob": "lift", "push": "drive", "drive": "drive",
    "short service": "serve", "long service": "serve", "service": "serve",
    "wrist smash": "smash", "back-court drive": "drive", "cross-court net shot": "net",
    "return net": "net", "rush": "drive", "defensive return drive": "drive",
    "defensive return lob": "lift",
}
PAD_BEFORE, PAD_AFTER = 15, 30  # frames of context around first/last hit

def convert(raw: pd.DataFrame, match_id: str, fps: float) -> pd.DataFrame:
    players = sorted(raw["player"].unique().tolist())
    pmap = {players[0]: 0, players[1] if len(players) > 1 else players[0]: 1}
    rows = []
    for rally_id, g in raw.groupby("rally"):
        start = int(g["frame_num"].min()) - PAD_BEFORE
        end = int(g["frame_num"].max()) + PAD_AFTER
        winner = pmap[g["getpoint_player"].iloc[0]]
        for _, r in g.iterrows():
            stroke = STROKE_MAP[r["type"].strip().lower()]  # KeyError on unknown = by design
            if stroke is None:
                continue
            rows.append(dict(
                match_id=match_id, video_file=f"{match_id}.mp4", fps=fps,
                rally_id=int(rally_id), hit_frame=int(r["frame_num"]),
                player=pmap[r["player"]], stroke=stroke,
                rally_start_frame=max(start, 0), rally_end_frame=end,
                rally_winner=winner,
            ))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="training/data/raw/shuttleset")
    ap.add_argument("--out", default="training/data/processed/labels.parquet")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    frames = []
    for f in sorted(glob.glob(f"{args.raw_dir}/**/*.csv", recursive=True)):
        match_id = f.split("/")[-1].removesuffix(".csv")
        frames.append(convert(pd.read_csv(f), match_id, args.fps))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(args.out)
    print(f"{len(out)} strokes, {out.match_id.nunique()} matches -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
```

**Adapt column names (`rally`, `frame_num`, `player`, `type`, `getpoint_player`) to the real ones from the notes file — change them in BOTH test fixture and implementation.** If hit positions are timestamps rather than frames, convert via fps here so the canonical schema stays frame-based.

- [ ] **Step 4: Tests pass; then run for real**: `mkdir -p training/data/processed && python training/prepare_shuttleset.py`. Fix `STROKE_MAP` until it completes; record dropped-type counts in the notes file. `dvc add training/data/processed/labels.parquet`.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: ShuttleSet -> canonical labels adapter"` and `dvc push`.

---

### Task 7: `extract_poses.py` — video → keypoint tracks

**Files:**
- Create: `training/extract_poses.py`
- Modify: `training/requirements.txt` (add `rtmlib>=0.0.13`, `onnxruntime>=1.17`, `opencv-python-headless`)

**Interfaces:**
- Consumes: a video file; target fps.
- Produces: per-video `<out_dir>/<match_id>.npz` with arrays — `kpts: (T,2,17,2) float32` (player 0 = nearer/bottom, player 1 = farther/top), `scores: (T,2,17) float32`, `meta: json string` (`fps_sampled, orig_fps, width, height, n_frames`). Exposes `extract(video_path: str, target_fps: float = 15.0) -> tuple[np.ndarray, np.ndarray, dict]` — imported by backend pipeline (Task 15).

No unit test (depends on model weights + real video); verification is an integration check on a real clip. TDD exemption is deliberate; the deterministic parts it uses (person→player assignment) get a unit test.

- [ ] **Step 1: Unit-testable helper first** — add to `training/extract_poses.py` and test in `training/tests/test_assign.py`:

```python
# in extract_poses.py
import numpy as np

def assign_players(kpts_people: np.ndarray, scores_people: np.ndarray):
    """(N,17,2),(N,17) for N detected people -> ((2,17,2),(2,17)).
    Picks the 2 highest-confidence people; player 0 = larger mean y (nearer camera).
    Pads with zeros if fewer than 2 people."""
    out_k = np.zeros((2, 17, 2), dtype=np.float32)
    out_s = np.zeros((2, 17), dtype=np.float32)
    if len(kpts_people) == 0:
        return out_k, out_s
    order = np.argsort(-scores_people.mean(axis=1))[:2]
    chosen = sorted(order, key=lambda i: -kpts_people[i, :, 1].mean())  # bottom first
    for slot, i in enumerate(chosen):
        out_k[slot], out_s[slot] = kpts_people[i], scores_people[i]
    return out_k, out_s
```

```python
# training/tests/test_assign.py
import numpy as np
from training.extract_poses import assign_players

def test_assign_two_best_by_conf_then_depth_order():
    k = np.zeros((3, 17, 2), dtype=np.float32)
    k[0, :, 1] = 600   # bottom person
    k[1, :, 1] = 100   # top person
    k[2, :, 1] = 350   # mid person, low conf -> excluded
    s = np.ones((3, 17), dtype=np.float32); s[2] *= 0.1
    ok, osc = assign_players(k, s)
    assert ok[0, 0, 1] == 600 and ok[1, 0, 1] == 100

def test_assign_handles_empty():
    ok, osc = assign_players(np.zeros((0, 17, 2)), np.zeros((0, 17)))
    assert ok.shape == (2, 17, 2) and (osc == 0).all()
```

Run failing → implement → passing.

- [ ] **Step 2: Implement extraction CLI** (same file):

```python
import argparse, json, cv2
from rtmlib import Body

def extract(video_path: str, target_fps: float = 15.0):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(orig_fps / target_fps)), 1)
    body = Body(mode="balanced", backend="onnxruntime", device="cpu")
    kpts_l, scores_l, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            people_k, people_s = body(frame)
            k2, s2 = assign_players(np.asarray(people_k), np.asarray(people_s))
            kpts_l.append(k2); scores_l.append(s2)
        i += 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    meta = dict(fps_sampled=orig_fps / step, orig_fps=orig_fps, width=w, height=h,
                n_frames=len(kpts_l))
    return np.stack(kpts_l), np.stack(scores_l), meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--out-dir", default="training/data/processed/poses")
    ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()
    import os; os.makedirs(a.out_dir, exist_ok=True)
    k, s, m = extract(a.video, a.fps)
    mid = os.path.basename(a.video).rsplit(".", 1)[0]
    np.savez_compressed(f"{a.out_dir}/{mid}.npz", kpts=k, scores=s, meta=json.dumps(m))
    print(f"{mid}: {k.shape} @ {m['fps_sampled']:.1f}fps")

if __name__ == "__main__":
    main()
```

**Verify rtmlib's actual return convention first** (`python -c "from rtmlib import Body; help(Body.__call__)"`); if `body(frame)` returns differently shaped arrays, adapt inside `extract` only — `assign_players` and the npz contract must not change.

- [ ] **Step 3: Integration verification on one real clip** — download ~60s of one ShuttleSet-referenced match (URL from notes file): `yt-dlp -f 'bv*[height<=720]' --download-sections '*10:00-11:00' -o training/data/raw/videos/m01.mp4 '<URL>'`, run `python training/extract_poses.py training/data/raw/videos/m01.mp4`, then check: shapes printed; spot-check by rendering 5 frames with keypoints drawn (`cv2.circle`) into `/tmp/pose_check/*.jpg` and viewing them — both players covered, player 0 is the bottom player.
- [ ] **Step 4: Commit** — `git add training/extract_poses.py training/tests/test_assign.py training/requirements.txt && git commit -m "feat: pose extraction CLI with player assignment"`

---

### Task 8: `build_windows.py` — training tensors + match-level splits + dvc.yaml

**Files:**
- Create: `training/build_windows.py`, `training/dvc.yaml`
- Test: `training/tests/test_build_windows.py`

**Interfaces:**
- Consumes: `labels.parquet` (Task 6 schema), pose npz dir (Task 7 contract), `shuttlesense_core.features.stroke_window/rally_frame_features`, `schemas.ALL_CLASSES`.
- Produces:
  - `training/data/processed/stroke_windows.npz`: `X:(N,30,68) float32`, `y:(N,) int64` (index into `ALL_CLASSES`), `match:(N,) <U32>`. Negatives (`none`) sampled 1:1 with positives at frames ≥ `fps` frames away from every hit.
  - `training/data/processed/rally_frames.npz`: per-match concatenated `X:(M,4) float32`, `y:(M,) float32` (1.0 inside any rally interval), `match:(M,)`.
  - `training/data/processed/splits.json`: `{"train": [match_ids], "val": [...], "test": [...]}` — 70/15/15 **by match**, seed 13, sorted then shuffled deterministically.
  - Function `build_stroke_samples(labels: pd.DataFrame, kpts: np.ndarray, fps_scale: float, rng) -> tuple[np.ndarray, np.ndarray]` for tests. `fps_scale = fps_sampled / labels.fps` maps label hit frames into pose-frame indices.

- [ ] **Step 1: Failing test** — `training/tests/test_build_windows.py`:

```python
import numpy as np, pandas as pd
from training.build_windows import build_stroke_samples, make_splits
from shuttlesense_core.schemas import ALL_CLASSES

def test_stroke_samples_positive_and_negative():
    rng = np.random.default_rng(0)
    T = 600
    kpts = rng.normal(size=(T, 2, 17, 2)).astype(np.float32)
    labels = pd.DataFrame({
        "hit_frame": [100, 300], "player": [0, 1], "stroke": ["smash", "net"],
        "fps": [30.0, 30.0],
    })
    X, y = build_stroke_samples(labels, kpts, fps_scale=0.5, rng=rng)
    assert X.shape[1:] == (30, 68)
    assert (y == ALL_CLASSES.index("smash")).sum() == 1
    assert (y == ALL_CLASSES.index("none")).sum() == 2   # 1:1 with positives

def test_splits_disjoint_and_deterministic():
    ids = [f"m{i:02d}" for i in range(20)]
    s1, s2 = make_splits(ids), make_splits(ids)
    assert s1 == s2
    assert not (set(s1["train"]) & set(s1["test"]))
    assert set(s1["train"]) | set(s1["val"]) | set(s1["test"]) == set(ids)
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — `training/build_windows.py`:

```python
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pandas as pd
from shuttlesense_core.features import rally_frame_features, stroke_window
from shuttlesense_core.schemas import ALL_CLASSES, NONE_CLASS

def build_stroke_samples(labels, kpts, fps_scale, rng):
    T = kpts.shape[0]
    Xs, ys, hit_idx = [], [], []
    for _, r in labels.iterrows():
        f = int(round(r["hit_frame"] * fps_scale))
        if not (0 <= f < T):
            continue
        Xs.append(stroke_window(kpts[:, int(r["player"])], f))
        ys.append(ALL_CLASSES.index(r["stroke"]))
        hit_idx.append(f)
    n_neg, guard = len(Xs), int(round(15))  # >= ~1s at 15fps sampled
    tries = 0
    while n_neg > 0 and tries < 10000:
        f = int(rng.integers(0, T)); tries += 1
        if all(abs(f - h) > guard for h in hit_idx):
            p = int(rng.integers(0, 2))
            Xs.append(stroke_window(kpts[:, p], f))
            ys.append(ALL_CLASSES.index(NONE_CLASS))
            n_neg -= 1
    return np.stack(Xs), np.asarray(ys, dtype=np.int64)

def make_splits(match_ids, seed=13):
    ids = sorted(match_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids); a, b = int(n * 0.7), int(n * 0.85)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="training/data/processed/labels.parquet")
    ap.add_argument("--poses", default="training/data/processed/poses")
    ap.add_argument("--out-dir", default="training/data/processed")
    a = ap.parse_args()
    labels = pd.read_parquet(a.labels)
    rng = np.random.default_rng(13)
    SX, Sy, Sm, RX, Ry, Rm = [], [], [], [], [], []
    for npz_path in sorted(glob.glob(f"{a.poses}/*.npz")):
        mid = os.path.basename(npz_path).removesuffix(".npz")
        ml = labels[labels.match_id == mid]
        if ml.empty:
            continue
        z = np.load(npz_path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        scale = meta["fps_sampled"] / float(ml["fps"].iloc[0])
        X, y = build_stroke_samples(ml, z["kpts"], scale, rng)
        SX.append(X); Sy.append(y); Sm.append(np.full(len(y), mid, dtype="<U32"))
        rf = rally_frame_features(z["kpts"], z["scores"])
        ry = np.zeros(len(rf), dtype=np.float32)
        for _, r in ml.drop_duplicates("rally_id").iterrows():
            s = int(r["rally_start_frame"] * scale); e = int(r["rally_end_frame"] * scale)
            ry[max(s, 0):e] = 1.0
        RX.append(rf); Ry.append(ry); Rm.append(np.full(len(ry), mid, dtype="<U32"))
    np.savez_compressed(f"{a.out_dir}/stroke_windows.npz",
                        X=np.concatenate(SX), y=np.concatenate(Sy), match=np.concatenate(Sm))
    np.savez_compressed(f"{a.out_dir}/rally_frames.npz",
                        X=np.concatenate(RX), y=np.concatenate(Ry), match=np.concatenate(Rm))
    with open(f"{a.out_dir}/splits.json", "w") as f:
        json.dump(make_splits(labels.match_id.unique().tolist()), f, indent=2)

if __name__ == "__main__":
    main()
```

`training/dvc.yaml` (pipeline reproducibility):

```yaml
stages:
  prepare:
    cmd: python training/prepare_shuttleset.py
    deps: [training/prepare_shuttleset.py, training/data/raw/shuttleset]
    outs: [training/data/processed/labels.parquet]
  windows:
    cmd: python training/build_windows.py
    deps: [training/build_windows.py, training/data/processed/labels.parquet,
           training/data/processed/poses]
    outs: [training/data/processed/stroke_windows.npz,
           training/data/processed/rally_frames.npz,
           training/data/processed/splits.json]
```

(Pose extraction stays a manual per-video step — GPU/YouTube dependent — its outputs directory is `dvc add`-ed.)

- [ ] **Step 4: Tests pass** (`pytest training/tests/test_build_windows.py -v`).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: training tensor builder with match-level splits and dvc pipeline"`

---

### Task 9: `models.py` — StrokeTCN + RallyGRU

**Files:**
- Create: `training/models.py`
- Test: `training/tests/test_models.py`
- Modify: `training/requirements.txt` (add `torch>=2.2`, `wandb>=0.17`, `scikit-learn>=1.4`, `pyyaml>=6`)

**Interfaces:**
- Produces: `StrokeTCN(in_dim=68, n_classes=8)` — forward `(B,30,68) -> (B,8)` logits; `RallyGRU(in_dim=4, hidden=32)` — forward `(B,T,4) -> (B,T)` per-frame play logits. Both pure `torch.nn.Module`, ONNX-exportable (no dynamic control flow).

- [ ] **Step 1: Failing test** — `training/tests/test_models.py`:

```python
import torch
from training.models import RallyGRU, StrokeTCN

def test_stroke_tcn_shapes():
    m = StrokeTCN()
    out = m(torch.randn(4, 30, 68))
    assert out.shape == (4, 8)

def test_rally_gru_shapes():
    m = RallyGRU()
    out = m(torch.randn(2, 100, 4))
    assert out.shape == (2, 100)
```

- [ ] **Step 2: Run to verify failure** (after `pip install -r training/requirements.txt`).

- [ ] **Step 3: Implement** — `training/models.py`:

```python
import torch
import torch.nn as nn

class StrokeTCN(nn.Module):
    def __init__(self, in_dim=68, n_classes=8, channels=(64, 128, 128), k=5):
        super().__init__()
        layers, c_in = [], in_dim
        for i, c in enumerate(channels):
            d = 2 ** i
            layers += [nn.Conv1d(c_in, c, k, dilation=d, padding=d * (k - 1) // 2),
                       nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(0.2)]
            c_in = c
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x):                    # (B,T,F)
        h = self.tcn(x.transpose(1, 2))      # (B,C,T)
        return self.head(h.mean(dim=2))      # (B,n_classes)

class RallyGRU(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):                    # (B,T,F)
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)      # (B,T) logits
```

- [ ] **Step 4: Tests pass.** **Step 5: Commit** — `git commit -am "feat: StrokeTCN and RallyGRU model definitions"`

---

### Task 10: `train_stroke.py` — training CLI with W&B

**Files:**
- Create: `training/train_stroke.py`, `training/configs/stroke_tcn.yaml`
- Test: `training/tests/test_train_stroke_smoke.py`

**Interfaces:**
- Consumes: `stroke_windows.npz`, `splits.json`, `models.StrokeTCN`.
- Produces: checkpoint `training/checkpoints/stroke_tcn/best.pt` = `{"state_dict", "config", "val_macro_f1", "classes"}`; W&B run when `--wandb` passed (default OFF so tests/smoke runs are offline). CLI: `python training/train_stroke.py --config training/configs/stroke_tcn.yaml [--data ...] [--splits ...] [--out-dir ...] [--wandb]`.

- [ ] **Step 1: Config** — `training/configs/stroke_tcn.yaml`:

```yaml
seed: 13
epochs: 40
batch_size: 256
lr: 0.001
weight_decay: 0.0001
channels: [64, 128, 128]
kernel: 5
```

- [ ] **Step 2: Failing smoke test** — `training/tests/test_train_stroke_smoke.py` (tiny synthetic npz, 2 epochs, asserts checkpoint + metric key):

```python
import json, subprocess, sys
import numpy as np

def test_train_smoke(tmp_path):
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, 30, 68)).astype(np.float32)
    y = rng.integers(0, 8, size=n).astype(np.int64)
    match = np.array(["m01"] * 32 + ["m02"] * 32, dtype="<U32")
    np.savez(tmp_path / "sw.npz", X=X, y=y, match=match)
    (tmp_path / "splits.json").write_text(json.dumps(
        {"train": ["m01"], "val": ["m02"], "test": []}))
    r = subprocess.run([sys.executable, "training/train_stroke.py",
                        "--config", "training/configs/stroke_tcn.yaml",
                        "--data", str(tmp_path / "sw.npz"),
                        "--splits", str(tmp_path / "splits.json"),
                        "--out-dir", str(tmp_path / "ckpt"), "--epochs", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import torch
    ck = torch.load(tmp_path / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    assert "state_dict" in ck and "val_macro_f1" in ck and len(ck["classes"]) == 8
```

- [ ] **Step 3: Run to verify failure**, then implement `training/train_stroke.py`:

```python
from __future__ import annotations
import argparse, json, os, random
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from training.models import StrokeTCN
from shuttlesense_core.schemas import ALL_CLASSES

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_split(data_path, splits_path):
    z = np.load(data_path)
    splits = json.loads(open(splits_path).read())
    out = {}
    for name in ("train", "val", "test"):
        m = np.isin(z["match"], splits[name])
        out[name] = (torch.from_numpy(z["X"][m]), torch.from_numpy(z["y"][m]))
    return out

def run_epoch(model, X, y, bs, opt=None, loss_fn=None):
    idx = torch.randperm(len(X)) if opt is not None else torch.arange(len(X))
    losses, preds, ys = [], [], []
    for i in range(0, len(X), bs):
        b = idx[i:i + bs]
        logits = model(X[b])
        if opt is not None:
            loss = loss_fn(logits, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        preds.append(logits.argmax(1)); ys.append(y[b])
    p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return (np.mean(losses) if losses else 0.0,
            f1_score(t, p, average="macro", zero_division=0), (t, p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="training/data/processed/stroke_windows.npz")
    ap.add_argument("--splits", default="training/data/processed/splits.json")
    ap.add_argument("--out-dir", default="training/checkpoints/stroke_tcn")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.epochs:
        cfg["epochs"] = a.epochs
    set_seed(cfg["seed"])
    data = load_split(a.data, a.splits)
    model = StrokeTCN(channels=tuple(cfg["channels"]), k=cfg["kernel"])
    counts = np.bincount(data["train"][1].numpy(), minlength=len(ALL_CLASSES)).astype(np.float32)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project="shuttlesense", job_type="train-stroke", config=cfg)
    os.makedirs(a.out_dir, exist_ok=True)
    best = -1.0
    for ep in range(cfg["epochs"]):
        model.train()
        tr_loss, tr_f1, _ = run_epoch(model, *data["train"], cfg["batch_size"], opt, loss_fn)
        model.eval()
        with torch.no_grad():
            _, va_f1, (t, p) = run_epoch(model, *data["val"], cfg["batch_size"])
        if wb:
            wb.log({"epoch": ep, "train_loss": tr_loss, "train_f1": tr_f1, "val_f1": va_f1})
        print(f"ep{ep} loss={tr_loss:.3f} train_f1={tr_f1:.3f} val_f1={va_f1:.3f}")
        if va_f1 > best:
            best = va_f1
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "val_macro_f1": best, "classes": ALL_CLASSES,
                        "confusion": confusion_matrix(t, p, labels=range(len(ALL_CLASSES))).tolist()},
                       f"{a.out_dir}/best.pt")
    if wb:
        wb.summary["best_val_macro_f1"] = best
        wb.finish()

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke test passes** (`pytest training/tests/test_train_stroke_smoke.py -v`).
- [ ] **Step 5: Commit** — `git commit -am "feat: stroke classifier training CLI (W&B, seeded, class-weighted)"`

---

### Task 11: `train_rally.py` — rally segmenter training CLI

**Files:**
- Create: `training/train_rally.py`, `training/configs/rally_gru.yaml`
- Test: `training/tests/test_train_rally_smoke.py`

**Interfaces:**
- Consumes: `rally_frames.npz` (per-frame `(M,4)` features, `(M,)` binary labels, match ids), `models.RallyGRU`, `smoothing.probs_to_intervals`.
- Produces: `training/checkpoints/rally_gru/best.pt` = `{"state_dict", "config", "val_frame_f1"}`. Same CLI flags as Task 10. Sequences are built by chunking each match's frames into windows of 512 with stride 512 (train) — chunking helper `chunk(X: np.ndarray, y: np.ndarray, size: int = 512) -> tuple[torch.Tensor, torch.Tensor]` exposed for the smoke test.

- [ ] **Step 1: Config** — `training/configs/rally_gru.yaml`:

```yaml
seed: 13
epochs: 30
batch_size: 16
lr: 0.001
weight_decay: 0.0001
hidden: 32
chunk: 512
```

- [ ] **Step 2: Failing smoke test** — mirror of Task 10's: synthetic `rally_frames.npz` with `X:(2000,4)`, `y` alternating blocks of 0/1, `match` half `m01` half `m02`; run CLI with `--epochs 2`; assert `best.pt` exists with `val_frame_f1` key. Use BCEWithLogitsLoss; frame-F1 = `f1_score(y_true, probs > 0.5)`.

```python
import json, subprocess, sys
import numpy as np

def test_train_rally_smoke(tmp_path):
    X = np.random.default_rng(0).normal(size=(2000, 4)).astype(np.float32)
    y = (np.arange(2000) // 100 % 2).astype(np.float32)
    match = np.array(["m01"] * 1000 + ["m02"] * 1000, dtype="<U32")
    np.savez(tmp_path / "rf.npz", X=X, y=y, match=match)
    (tmp_path / "splits.json").write_text(json.dumps(
        {"train": ["m01"], "val": ["m02"], "test": []}))
    r = subprocess.run([sys.executable, "training/train_rally.py",
                        "--config", "training/configs/rally_gru.yaml",
                        "--data", str(tmp_path / "rf.npz"),
                        "--splits", str(tmp_path / "splits.json"),
                        "--out-dir", str(tmp_path / "ckpt"), "--epochs", "2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import torch
    ck = torch.load(tmp_path / "ckpt" / "best.pt", map_location="cpu", weights_only=False)
    assert "val_frame_f1" in ck
```

- [ ] **Step 3: Implement `training/train_rally.py`** — structure identical to `train_stroke.py` (argparse, seed, yaml config, optional W&B, best-checkpoint save) with these differences:

```python
# chunking helper (module level, imported by test if needed)
def chunk(X, y, size=512):
    n = (len(X) // size) * size
    if n == 0:
        pad = size - len(X)
        X = np.pad(X, ((0, pad), (0, 0))); y = np.pad(y, (0, pad))
        n = size
    return (torch.from_numpy(X[:n].reshape(-1, size, X.shape[1])),
            torch.from_numpy(y[:n].reshape(-1, size)))
# training loop core:
#   model = RallyGRU(hidden=cfg["hidden"])
#   loss_fn = nn.BCEWithLogitsLoss()
#   per epoch: logits = model(Xb); loss = loss_fn(logits, yb)
#   val metric: f1_score(y_val.ravel(), (torch.sigmoid(logits).numpy().ravel() > 0.5))
#   save {"state_dict", "config": cfg, "val_frame_f1": best}
```

Write it out fully following Task 10's skeleton — same `load_split` pattern applied per-match then chunked, same best-checkpoint logic with key `val_frame_f1`.

- [ ] **Step 4: Smoke test passes.** **Step 5: Commit** — `git commit -am "feat: rally segmenter training CLI"`

---

### Task 12: Real data runs + `evaluate.py` + eval report

**Files:**
- Create: `training/evaluate.py`, `training/reports/` (first report), `training/notes/colab.md`
- Test: covered by evaluate's own run on real checkpoints (no unit test; deterministic logic is reused from Tasks 2–3 which are tested)

**Interfaces:**
- Consumes: both checkpoints, `stroke_windows.npz`/`rally_frames.npz`, `splits.json`.
- Produces: `training/reports/YYYY-MM-DD-eval.md` — committed markdown with: dataset sizes per split, stroke macro-F1 + per-class F1 + confusion matrix (test split), rally frame-F1 + temporal IoU of `probs_to_intervals` output vs. label intervals, and the **published baseline** table (majority-class and a logistic-regression-on-flattened-window baseline, both computed here). CLI: `python training/evaluate.py --stroke-ckpt ... --rally-ckpt ... --out training/reports/<date>-eval.md`.

- [ ] **Step 1: Extract poses for the full label set.** For each match in `labels.parquet` (start with ≥ 8 matches for meaningful splits): download the referenced video segment (`yt-dlp`, URLs in notes file), run `extract_poses.py`. GPU not required (rtmlib CPU ~2-4× slower than realtime; Colab is fine too — see `training/notes/colab.md` written this step: venv-free `pip install -r training/requirements.txt`, mount Drive, run the same CLIs). Then `python training/build_windows.py`, `dvc add training/data/processed/poses` + `dvc repro` + `dvc push`.
- [ ] **Step 2: Train both models for real** — on Colab GPU or locally: `python training/train_stroke.py --config training/configs/stroke_tcn.yaml --wandb` and `python training/train_rally.py --config training/configs/rally_gru.yaml --wandb`. Record W&B run URLs in the report.
- [ ] **Step 3: Implement `training/evaluate.py`** — loads checkpoints, rebuilds test tensors from npz + splits, computes: stroke macro/per-class F1 + confusion (reuse `run_epoch` pattern from Task 10); baselines: `sklearn.dummy.DummyClassifier(strategy="most_frequent")` and `sklearn.linear_model.LogisticRegression(max_iter=1000)` on `X.reshape(N, -1)`; rally: frame-F1 and mean temporal IoU between predicted intervals (`probs_to_intervals(sigmoid(logits))`) and true intervals per test match. Renders all as a markdown file with tables. (~120 lines; follow Task 10's loading conventions exactly.)
- [ ] **Step 4: Quality gate check** — the TCN must beat the logistic baseline on test macro-F1. If it does not, STOP and debug (superpowers:systematic-debugging) — likely fps_scale bugs or player-assignment noise — before proceeding to Part C.
- [ ] **Step 5: Commit** — `git add training/evaluate.py training/reports training/notes/colab.md && git commit -m "feat: eval suite with baselines; first eval report"`

---

### Task 13: ONNX export + registry + train/serve consistency test

**Files:**
- Create: `training/export_onnx.py`, `tests/test_consistency.py`, `tests/fixtures/` (small npz fixture)
- Modify: `training/requirements.txt` (add `onnx>=1.16`)

**Interfaces:**
- Consumes: both checkpoints.
- Produces: `backend/models/stroke_tcn.onnx` (input `x:(B,30,68)`, output `logits:(B,8)`), `backend/models/rally_gru.onnx` (input `x:(1,T,4)` dynamic T, output `logits:(1,T)`); `backend/models/manifest.json` = `{"stroke": {"file", "val_macro_f1", "git_sha", "classes"}, "rally": {"file", "val_frame_f1", "git_sha"}}`. With `--wandb`, uploads both as W&B artifacts (`stroke-tcn:latest`, `rally-gru:latest`) — that is the model registry; serving pins via the committed manifest.

- [ ] **Step 1: Failing consistency + parity test** — `tests/test_consistency.py`:

```python
import json, subprocess, sys
from pathlib import Path
import numpy as np

def test_train_and_serve_import_same_feature_functions():
    import backend.app.pipeline as sp
    import shuttlesense_core.features as cf
    assert sp.stroke_window is cf.stroke_window
    assert sp.rally_frame_features is cf.rally_frame_features

def test_onnx_matches_torch(tmp_path):
    import onnxruntime as ort
    import torch
    from training.models import StrokeTCN
    m = StrokeTCN(); m.eval()
    ck = {"state_dict": m.state_dict(),
          "config": {"channels": [64, 128, 128], "kernel": 5},
          "val_macro_f1": 0.0, "classes": []}
    torch.save(ck, tmp_path / "best.pt")
    r = subprocess.run([sys.executable, "training/export_onnx.py",
                        "--stroke-ckpt", str(tmp_path / "best.pt"),
                        "--skip-rally", "--out-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    x = np.random.default_rng(0).normal(size=(3, 30, 68)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(tmp_path / "stroke_tcn.onnx"))
    out = sess.run(None, {"x": x})[0]
    np.testing.assert_allclose(out, ref, atol=1e-4)
```

(The first test fails until Task 15 creates `backend/app/pipeline.py`; mark it `@pytest.mark.xfail(reason="pipeline lands in Task 15")` now and REMOVE the marker in Task 15.)

- [ ] **Step 2: Implement `training/export_onnx.py`** — loads checkpoints, rebuilds models from stored config, `torch.onnx.export` with `input_names=["x"], output_names=["logits"]`, `dynamic_axes={"x": {0: "B"}}` (stroke) / `{"x": {1: "T"}}` (rally), writes manifest with `git rev-parse HEAD`, optional `--wandb` artifact upload, `--skip-rally`/`--skip-stroke` flags for partial export.
- [ ] **Step 3: Run tests** — onnx parity passes; consistency xfails as expected.
- [ ] **Step 4: Export the real models** — `python training/export_onnx.py --stroke-ckpt training/checkpoints/stroke_tcn/best.pt --rally-ckpt training/checkpoints/rally_gru/best.pt --wandb`; commit `backend/models/manifest.json` (onnx files stay gitignored; Docker build re-downloads from W&B or copies locally).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: ONNX export, model manifest/registry, consistency tests"`

---

# Part C — Serving & demo

### Task 14: Backend scaffold — config, DB, job queue

**Files:**
- Create: `backend/requirements.txt`, `backend/app/__init__.py`, `backend/app/config.py`, `backend/app/db.py`
- Test: `backend/tests/test_db.py` (+ `backend/tests/__init__.py`)

**Interfaces:**
- Produces:
  - `config.Settings` (pydantic-settings): `data_dir: str = "data"`, `models_dir: str = "backend/models"`, `samples_dir: str = "backend/samples"`, `static_dir: str | None = None`, `max_upload_mb: int = 100`, `max_duration_s: int = 95`, `target_fps: float = 15.0`. `get_settings()` cached accessor.
  - `db.connect(path) -> sqlite3.Connection` (row_factory=Row, creates schema), `db.create_job(conn, filename) -> str` (uuid4 hex id, status `queued`), `db.claim_next(conn) -> Row | None` (oldest `queued` → `processing`, atomic), `db.finish(conn, job_id, status: str, error: str | None = None)` (status `done`/`failed`), `db.get_job(conn, job_id) -> Row | None`. Jobs table: `id TEXT PK, filename TEXT, status TEXT, error TEXT, created_at REAL`.

- [ ] **Step 1: Failing tests** — `backend/tests/test_db.py`:

```python
from backend.app import db

def test_job_lifecycle(tmp_path):
    conn = db.connect(tmp_path / "jobs.sqlite")
    jid = db.create_job(conn, "match.mp4")
    assert db.get_job(conn, jid)["status"] == "queued"
    row = db.claim_next(conn)
    assert row["id"] == jid and db.get_job(conn, jid)["status"] == "processing"
    assert db.claim_next(conn) is None          # nothing left to claim
    db.finish(conn, jid, "done")
    assert db.get_job(conn, jid)["status"] == "done"

def test_finish_with_error(tmp_path):
    conn = db.connect(tmp_path / "j.sqlite")
    jid = db.create_job(conn, "x.mp4")
    db.claim_next(conn)
    db.finish(conn, jid, "failed", error="not a badminton video")
    row = db.get_job(conn, jid)
    assert row["status"] == "failed" and "badminton" in row["error"]
```

- [ ] **Step 2: Verify failure**, then implement. `backend/requirements.txt`: `fastapi>=0.111`, `uvicorn[standard]>=0.29`, `pydantic-settings>=2.2`, `python-multipart>=0.0.9`, `onnxruntime>=1.17`, `numpy`, `opencv-python-headless`, `rtmlib`, `httpx>=0.27` (tests). `db.py` uses plain `sqlite3` with `BEGIN IMMEDIATE` in `claim_next` for atomicity:

```python
def claim_next(conn):
    with conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            return None
        conn.execute("UPDATE jobs SET status='processing' WHERE id=?", (row["id"],))
    return row
```

- [ ] **Step 3: Tests pass.** **Step 4: Commit** — `git commit -am "feat: backend config and sqlite job queue"`

---

### Task 15: `pipeline.py` + `worker.py` — the analysis pipeline

**Files:**
- Create: `backend/app/pipeline.py`, `backend/app/worker.py`
- Test: `backend/tests/test_pipeline.py`
- Modify: `tests/test_consistency.py` (remove the xfail marker from Task 13)

**Interfaces:**
- Consumes: `extract_poses.extract` (re-implemented import-free: pipeline imports `training.extract_poses` is FORBIDDEN in backend — instead `pipeline.py` contains its own thin `extract_poses_onnx` calling rtmlib directly, but **all feature/smoothing math comes from `shuttlesense_core`**), ONNX models via manifest, `schemas.MatchReport`.
- Produces:
  - `pipeline.analyze(video_path: str, models_dir: str, target_fps: float = 15.0, pose_fn=None) -> tuple[MatchReport, dict]` — the dict is the tracks payload: `{"fps": float, "edges": COCO_EDGES, "kpts": [[...]], "scores": [[...]]}` (keypoints rounded to 1 decimal for JSON size). `pose_fn` injection point: `(video_path, target_fps) -> (kpts, scores, meta)` — tests pass a fake; production default runs rtmlib.
  - Inference internals (unit-tested via fake pose_fn): rally probs from RallyGRU onnx → `probs_to_intervals(min_len=int(fps), merge_gap=int(fps/2))`; stroke sliding windows stride 4 within rallies only, both players → drop `none` argmax → `suppress_events(min_gap=int(fps/3))` → `StrokeEvent`s.
  - `worker.run_forever(settings)` — poll loop: claim job → `analyze` → write `data/jobs/<id>/report.json` + `tracks.json` → `finish(done)`; exceptions → `finish(failed, error=str(exc))`. `worker.run_once(settings) -> bool` for tests.

- [ ] **Step 1: Failing test** — `backend/tests/test_pipeline.py` with a fake pose function that injects an obvious "rally then idle" pattern:

```python
import json
import numpy as np
from backend.app.pipeline import analyze

def fake_pose(video_path, target_fps):
    T = 300
    rng = np.random.default_rng(1)
    kpts = np.zeros((T, 2, 17, 2), dtype=np.float32)
    base = np.zeros((17, 2), dtype=np.float32)
    base[11], base[12], base[5], base[6] = [90, 400], [110, 400], [88, 340], [112, 340]
    kpts[:] = base
    kpts[50:200] += rng.normal(scale=15.0, size=(150, 2, 17, 2))  # movement burst
    scores = np.ones((T, 2, 17), dtype=np.float32)
    return kpts, scores, {"fps_sampled": 15.0, "width": 1280, "height": 720, "n_frames": T}

def test_analyze_produces_report_and_tracks(tmp_path):
    report, tracks = analyze("ignored.mp4", models_dir="backend/models",
                             pose_fn=fake_pose)
    assert report.n_frames == 300 and report.fps == 15.0
    assert tracks["fps"] == 15.0 and len(tracks["kpts"]) == 300
    for ev in report.strokes:
        assert 0 <= ev.frame < 300 and ev.stroke != "none"
    json.dumps(report.to_dict())   # serializable
```

(This test needs real ONNX files in `backend/models/` — from Task 13 Step 4. In CI without them, `pytest.mark.skipif(not Path("backend/models/stroke_tcn.onnx").exists(), ...)`.)

- [ ] **Step 2: Verify failure**, implement `pipeline.py`:

```python
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from shuttlesense_core.features import COCO_EDGES, rally_frame_features, stroke_window
from shuttlesense_core.schemas import ALL_CLASSES, NONE_CLASS, MatchReport, RallyInterval, StrokeEvent
from shuttlesense_core.smoothing import probs_to_intervals, suppress_events

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def extract_poses_onnx(video_path, target_fps):
    import cv2
    from rtmlib import Body
    # identical logic to training/extract_poses.extract but self-contained;
    # person->player slotting copied as _assign (same math, unit-covered there)
    ...

def analyze(video_path, models_dir, target_fps=15.0, pose_fn=None):
    pose_fn = pose_fn or extract_poses_onnx
    kpts, scores, meta = pose_fn(video_path, target_fps)
    fps = float(meta["fps_sampled"])
    rally_sess = ort.InferenceSession(str(Path(models_dir) / "rally_gru.onnx"))
    stroke_sess = ort.InferenceSession(str(Path(models_dir) / "stroke_tcn.onnx"))
    rf = rally_frame_features(kpts, scores)[None, ...]              # (1,T,4)
    probs = _sigmoid(rally_sess.run(None, {"x": rf})[0][0])
    intervals = probs_to_intervals(probs, min_len=int(fps), merge_gap=int(fps / 2))
    events = []
    for s, e in intervals:
        for player in (0, 1):
            wins, frames = [], list(range(s, e, 4))
            for f in frames:
                wins.append(stroke_window(kpts[:, player], f))
            logits = stroke_sess.run(None, {"x": np.stack(wins)})[0]
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            for f, pr in zip(frames, p):
                c = int(pr.argmax())
                if ALL_CLASSES[c] != NONE_CLASS:
                    events.append({"frame": f, "player": player,
                                   "stroke": ALL_CLASSES[c], "confidence": float(pr[c])})
    events = suppress_events(events, min_gap=int(fps / 3))
    report = MatchReport(
        fps=fps, width=meta["width"], height=meta["height"], n_frames=meta["n_frames"],
        rallies=[RallyInterval(s, e) for s, e in intervals],
        strokes=[StrokeEvent(**{k: ev[k] for k in ("frame", "player", "stroke", "confidence")})
                 for ev in events],
    )
    tracks = {"fps": fps, "edges": COCO_EDGES,
              "kpts": np.round(kpts, 1).tolist(), "scores": np.round(scores, 2).tolist()}
    return report, tracks
```

`worker.py`: `run_once` claims a job, calls `analyze(settings.data_dir + f"/uploads/{row['id']}/{row['filename']}", settings.models_dir)`, writes both JSONs under `data/jobs/<id>/`, finishes; `run_forever` loops with `time.sleep(1)`; started as a thread from `main.py` (Task 16). If `analyze` raises `ValueError("no rallies detected")` (raise it when `intervals == []`), finish failed with the friendly message `"This video doesn't look like a badminton match we can analyze."`

- [ ] **Step 3: Remove the Task 13 xfail; run** `pytest backend/tests/test_pipeline.py tests/test_consistency.py -v` → pass (with real onnx present).
- [ ] **Step 4: Commit** — `git commit -am "feat: analysis pipeline and job worker"`

---

### Task 16: API routes + app assembly + hardened static serving

**Files:**
- Create: `backend/app/routes.py`, `backend/app/main.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Produces (consumed by frontend Task 17-18):
  - `POST /api/matches` (multipart `file`) → `202 {"job_id"}`; `400` on bad extension/size; `413` over `max_upload_mb`; duration probed with `cv2.VideoCapture` (`frame_count / fps`), `400` if > `max_duration_s`.
  - `GET /api/matches/{job_id}` → `{"status": "queued|processing|done|failed", "error": str|null}`; 404 unknown.
  - `GET /api/matches/{job_id}/report` → report JSON; `GET /api/matches/{job_id}/tracks` → tracks JSON; `GET /api/matches/{job_id}/video` → uploaded file (FileResponse); all 404 until done.
  - `GET /api/samples` → `[{"id", "title"}]`; `GET /api/samples/{id}/report|tracks|video` — served from `samples_dir/<id>/`.
  - SPA static serving when `settings.static_dir` set: **reuse Lift's resolve-and-confine pattern** — resolve requested path against static root, `resolved.is_relative_to(root)` or 404; `/api` prefixed paths never fall through to SPA; fallback to `index.html` for client routes.

- [ ] **Step 1: Failing tests** — `backend/tests/test_api.py` using `TestClient`, monkeypatched settings (tmp dirs) and a stub worker (don't start the thread in tests). Cover: upload happy path returns job_id and job is `queued`; `.txt` upload → 400; unknown job → 404; report of queued job → 404; sample listing reads `samples_dir` subdirs with `meta.json`; path traversal `GET /files/../../etc/passwd` style probes on static mount → 404 (create a tiny static dir fixture with `index.html`).

```python
# representative excerpts — write all listed cases
def test_upload_and_status(client, tmp_dirs):
    r = client.post("/api/matches", files={"file": ("m.mp4", b"0" * 1024, "video/mp4")})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert client.get(f"/api/matches/{jid}").json()["status"] == "queued"

def test_bad_extension_rejected(client, tmp_dirs):
    r = client.post("/api/matches", files={"file": ("x.txt", b"nope", "text/plain")})
    assert r.status_code == 400

def test_traversal_confined(client_with_static):
    for probe in ("/../../etc/passwd", "/%2e%2e/%2e%2e/etc/passwd", "//etc/passwd"):
        assert client_with_static.get(probe).status_code in (200, 404)
        # 200 only if it served index.html fallback — assert body is the SPA shell
```

- [ ] **Step 2: Verify failures, implement `routes.py` + `main.py`.** `main.py`: builds FastAPI app, includes router, starts worker thread on startup (`threading.Thread(target=run_forever, daemon=True)`) unless env `SHUTTLESENSE_NO_WORKER=1` (tests/CI), mounts static per the confine pattern. Duration probe helper in routes:

```python
def video_duration_s(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(n / fps) if fps else 0.0
```

- [ ] **Step 3: All backend tests pass** (`pytest backend/tests -v`).
- [ ] **Step 4: Commit** — `git commit -am "feat: matches API, samples API, hardened SPA serving"`

---

### Task 17: Frontend scaffold + annotated video player

**Files:**
- Create: `frontend/` via Vite (`npm create vite@latest frontend -- --template react`), `frontend/src/api.js`, `frontend/src/components/Player.jsx`, `frontend/src/pages/Report.jsx`, `frontend/src/App.jsx`
- Test: `frontend/src/components/player.test.js` (vitest) for the pure helpers

**Interfaces:**
- Consumes: `/api/samples`, `/api/{samples|matches}/{id}/report|tracks|video`.
- Produces: `<Player report={} tracks={} videoUrl={} onTimeRef={}/>` — `<video>` with absolutely-positioned `<canvas>`; per `requestAnimationFrame`: `frame = Math.round(video.currentTime * tracks.fps)`, draw skeleton edges per player (player 0 green `#63d5a0`, player 1 blue `#6fa8ff`), active stroke chip (stroke label + confidence for events within ±0.4s), rally segments rendered on a custom seek bar div under the video. Pure helpers exported for tests: `frameForTime(t, fps)`, `activeStroke(strokes, frame, fps)`, `scaleKpt(kpt, videoW, videoH, canvasW, canvasH)`.
- Before implementing components, load the **dataviz skill** (chart styling rules) and **frontend-design skill** (visual direction) — the report page should read as a designed product, dark theme per the approved mockups.

- [ ] **Step 1: Scaffold** — `npm create vite@latest frontend -- --template react && cd frontend && npm i && npm i -D vitest`; add `"test": "vitest run"` script; dev proxy in `vite.config.js`: `server: {proxy: {"/api": "http://localhost:8000"}}`.
- [ ] **Step 2: Failing helper tests**:

```js
import { activeStroke, frameForTime } from "./Player.jsx";
import { expect, test } from "vitest";

test("frameForTime rounds", () => {
  expect(frameForTime(2.03, 15)).toBe(30);
});

test("activeStroke finds event within window", () => {
  const strokes = [{ frame: 45, player: 0, stroke: "smash", confidence: 0.9 }];
  expect(activeStroke(strokes, 46, 15)).toMatchObject({ stroke: "smash" });
  expect(activeStroke(strokes, 80, 15)).toBeNull();
});
```

- [ ] **Step 3: Implement Player + Report page.** Report page fetches report/tracks/video for a sample or job id from the route (`/#/sample/:id`, `/#/match/:id` — hash routing, no router dep), renders Player. Canvas sized to the video element with `ResizeObserver`; kpts scaled from `report.width/height` to canvas box.
- [ ] **Step 4: `npm test` green; manual check** — `uvicorn backend.app.main:app` + `npm run dev`, open the sample route once Task 19 provides a sample (until then, verify against a locally analyzed job id via `curl -F file=@clip.mp4 localhost:8000/api/matches`).
- [ ] **Step 5: Commit** — `git add frontend && git commit -m "feat: report page with annotated video player"`

---

### Task 18: Momentum chart, rally explorer, upload flow, landing page

**Files:**
- Create: `frontend/src/components/Momentum.jsx`, `frontend/src/components/RallyList.jsx`, `frontend/src/components/Upload.jsx`, `frontend/src/lib/stats.js`
- Test: `frontend/src/lib/stats.test.js`

**Interfaces:**
- Consumes: `report.rallies` (winner may be null), `report.strokes`, Player's seek ref.
- Produces:
  - `stats.js` pure functions (tested): `scoreRace(rallies) -> [{frame, p0, p1}] | null` (null if any winner is null), `controlRibbon(strokes, nFrames, fps, win=10s) -> [{startFrame, endFrame, leader}]` (leader = player with higher attacking share — attacking strokes: `smash`, `drive`; `leader:null` when no strokes in window), `strokeMix(strokes) -> {0: {clear: n, ...}, 1: {...}}`, `rallySummaries(rallies, strokes) -> [{index, startFrame, endFrame, shots, endedBy}]`.
  - `Momentum` — SVG: score-race polylines when `scoreRace` non-null, else ribbon-only; clicking an x position seeks the video.
  - `RallyList` — rows from `rallySummaries`; click seeks to `startFrame/fps`.
  - `Upload` — file input + POST, poll status every 2s, redirect to `/#/match/:id` when done, show friendly error text on `failed`.
  - Landing (`/`): fetch `/api/samples`, immediately redirect to `/#/sample/<first>` — the zero-action demo requirement.

- [ ] **Step 1: Failing stats tests** (representative — write all four functions' cases):

```js
import { controlRibbon, rallySummaries, scoreRace } from "./stats.js";
import { expect, test } from "vitest";

test("scoreRace null when winners missing", () => {
  expect(scoreRace([{ start_frame: 0, end_frame: 10, winner: null }])).toBeNull();
});

test("scoreRace accumulates", () => {
  const out = scoreRace([
    { start_frame: 0, end_frame: 100, winner: 0 },
    { start_frame: 120, end_frame: 300, winner: 1 },
    { start_frame: 320, end_frame: 400, winner: 1 },
  ]);
  expect(out[2]).toMatchObject({ p0: 1, p1: 2 });
});

test("rallySummaries counts shots inside interval", () => {
  const out = rallySummaries(
    [{ start_frame: 0, end_frame: 100, winner: null }],
    [{ frame: 10, player: 0, stroke: "serve" }, { frame: 60, player: 1, stroke: "smash" }],
  );
  expect(out[0]).toMatchObject({ shots: 2, endedBy: "smash" });
});
```

- [ ] **Step 2: Implement stats.js → tests green.**
- [ ] **Step 3: Implement components + wire Report page** layout per approved mockup: player hero, momentum below, rally list below that; "Analyze your own video" button top-right opens Upload.
- [ ] **Step 4: Manual end-to-end check** with dev servers; then **Step 5: Commit** — `git commit -am "feat: momentum, rally explorer, upload flow, sample landing"`

---

### Task 19: Sample match generation

**Files:**
- Create: `scripts/build_samples.py`, `scripts/samples.yaml`, `backend/samples/` outputs

**Interfaces:**
- Consumes: `pipeline.analyze`, local clip files + optional ShuttleSet-derived winners.
- Produces: for each entry in `scripts/samples.yaml` (`id`, `title`, `clip` path, optional `winners`: list of 0/1 per rally): `backend/samples/<id>/{meta.json, report.json, tracks.json, video.mp4}`; video re-encoded ≤ 720p/2Mbps via ffmpeg for hosting weight. Winners are zipped onto `report.rallies` in order when provided (lengths must match or the script fails loudly).

- [ ] **Step 1: Write `scripts/build_samples.py`** — for each sample: run `analyze`, patch winners, write outputs, `ffmpeg -i clip -vf scale=-2:720 -b:v 2M -movflags +faststart video.mp4`, write `meta.json = {"id", "title"}`.
- [ ] **Step 2: Produce 2 samples** from held-out **test-split** matches (never train matches — the demo must show honest generalization): pick two 60–90s segments with clear rallies; run the script; verify each report has ≥ 3 rallies and sensible stroke labels by eye in the UI.
- [ ] **Step 3: Track with DVC** (`dvc add backend/samples`) — sample videos are too heavy for git; Docker build pulls them via `dvc pull` or a COPY from build context.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: pre-analyzed sample matches for zero-action demo"`

---

### Task 20: Container + Render deploy

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `render.yaml`
- Modify: `README.md` (badges, live URL, architecture, screenshots)

**Interfaces:**
- Consumes: everything.
- Produces: single container — stage 1 `node:20-slim` builds frontend (`npm ci && npm run build`); stage 2 `python:3.11-slim` installs `backend/requirements.txt` + `core/`, copies `backend/`, `frontend/dist` → serves with `STATIC_DIR=/app/static`, `PORT` from env, models + samples copied into image. Health endpoint `GET /api/healthz` → `{"ok": true}` (add in this task to `routes.py` with a one-line test).

- [ ] **Step 1: Dockerfile**:

```dockerfile
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY core/ core/
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt -e ./core
COPY backend/ backend/
COPY --from=ui /ui/dist static/
RUN useradd -m -u 1000 appuser && mkdir -p /app/data && chown -R appuser /app/data
USER appuser
ENV STATIC_DIR=/app/static PYTHONUNBUFFERED=1
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

- [ ] **Step 2: Local container verification** — `docker build -t shuttlesense . && docker run -p 8000:8000 shuttlesense`; check: `curl localhost:8000/api/healthz`, landing page shows sample report, traversal probes 404, upload of a 30s clip completes end-to-end (CPU will be slow — that's expected and documented).
- [ ] **Step 3: `render.yaml`** — mirror Lift's blueprint (web service, docker runtime, free plan, health check `/api/healthz`); push repo to GitHub (public), create Render blueprint deploy; verify live URL end-to-end.
- [ ] **Step 4: README** — hero screenshot + live URL + 3-minute demo script + architecture diagram + honest model-metrics table copied from the eval report + W&B project link + "what I'd do next" (Phase 2). Screenshot the live sample report for the hero image.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: single-container deploy on Render"` and push.

---

## Self-review (completed during plan writing)

1. **Spec coverage:** §3 architecture → Tasks 1–4, 14–16, 20; §4 UI (hero player, momentum, rally explorer, zero-action landing) → Tasks 17–19; §5 datasets/models 1–2 → Tasks 5–12; §6 MLOps (W&B, DVC, registry, eval gate, consistency test) → Tasks 5, 8, 10–13; §7 serving/guardrails → Tasks 14–16, 20; §8 testing → throughout; §9 Phase 1 milestones → all. Phase 2 items (TrackNet, heatmaps, heat overlay, domain-gap eval) intentionally out — separate plan. Court-relative features deferred with a documented deviation in Global Constraints.
2. **Placeholder scan:** the two `...` ellipses (Task 15 `extract_poses_onnx` body, Task 11 loop core) are accompanied by explicit "copy Task 10's/Task 7's structure with these exact differences" instructions and full interface contracts — implementable without invention. STROKE_MAP extension is a defined discovery procedure with a loud failure mode, not a TBD.
3. **Type consistency:** `MatchReport/RallyInterval/StrokeEvent` field names match across Tasks 1, 15, 18 (JSON uses `start_frame/end_frame` via dataclass asdict — frontend tests use the same names). `stroke_window/rally_frame_features/probs_to_intervals/suppress_events` signatures consistent across Tasks 2, 3, 8, 15. npz keys (`kpts/scores/meta`, `X/y/match`) consistent across Tasks 7, 8, 10, 11. Manifest/ONNX io names (`x`/`logits`) consistent across Tasks 13, 15.
