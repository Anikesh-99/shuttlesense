# ShuttleSense Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shuttle tracking + a top-down shot-placement heatmap to the completed Phase 1 app: fine-tune a pretrained TrackNetV3 shuttle tracker, run it on the existing broadcast sample matches, and render a court heatmap of shot placement on the sample reports.

**Architecture:** Extends Phase 1 (Approach A). A fine-tuned TrackNet (ONNX-CPU) becomes a new stage in `backend/app/pipeline.py`, gated on a per-match manual 4-corner calibration; landings are detected at rally-ends and mapped to court meters via the existing `core/shuttlesense_core/homography.py`; `report.json` gains `court_corners`/`landings`; a new React `CourtHeatmap` renders them. Samples are calibrated once at build time so the live demo shows heatmaps with zero interaction. Uploads (no calibration in v1) keep the Phase 1 report + a "coming soon" note.

**Tech Stack:** Python 3.11 (training, PyTorch ≥2.6), Python 3.12 (backend, onnxruntime, no torch), TrackNetV3 (vendored arch + public pretrained weights), OpenCV, DVC + Google Drive, Weights & Biases, FastAPI, React + Vite, Docker, Render.

**Spec:** `docs/superpowers/specs/2026-09-03-shuttlesense-phase2-design.md`

**Subagent model assignments (per project CLAUDE.md):** implementation → **Sonnet**; review → **Opus**; substantive testing → **Opus**. Fix-loop rounds 4-5 escalate the implementer to Opus.

## Global Constraints

- **Builds on Phase 1, merged to `main`.** Follow the established Phase 1 patterns: `training/` CLIs are config-driven (YAML), seeded, W&B-tracked with `WANDB_MODE=offline` fallback when `WANDB_API_KEY` is absent; datasets are DVC-tracked with **by-match** splits (never random-by-frame); `core/` holds the train/serve-shared feature/post-processing code, pinned by a consistency test.
- **Backend stays torch-free.** `backend/app/` imports no `torch` and no `training.*`. TrackNet serves as ONNX via `onnxruntime`. Any frame pre/post-processing the backend needs that also lives in training is either imported from `core/` or duplicated with an anti-drift parity test (Phase 1's `assign_players` pattern).
- **Free Colab/Kaggle GPU only.** TrackNet fine-tunes from pretrained weights at low resolution; if the full dataset overruns sessions, subset it and disclose the subset in the eval report (no silent caps).
- **Court dimensions:** `COURT_W = 6.1`, `COURT_L = 13.4` meters (already in `homography.py`).
- **Canonical corner order (fixed, used everywhere):** `["near_left", "near_right", "far_right", "far_left"]` — clockwise from the corner nearest the camera on the left. Court-meter coordinates for these, in this order: `[(0, 0), (COURT_W, 0), (COURT_W, COURT_L), (0, COURT_L)]`. Any calibration stores the 4 image-pixel points in this order.
- **Quality gate:** the fine-tuned TrackNet must beat the pretrained-as-is baseline on the held-out by-match test split (detection F1 within the dataset's standard pixel tolerance). If it does not, ship pretrained-as-is and report the honest negative result — do not overclaim a fine-tune win.
- **Landing heuristic (v1):** a rally's landing = the shuttle's court position at that rally's end frame (rally intervals come from the Phase 1 segmenter). No arc-fitting in v1.
- **`report.json` back-compat:** `court_corners`/`landings` are OPTIONAL and absent for uploads; Phase 1 reports without them must still load. `MatchReport.from_dict` defaults them.
- Git commits: plain messages, NO AI attribution (project CLAUDE.md).
- Free tiers only: Colab/Kaggle, W&B, Google Drive (DVC remote), Render. Target cost $0.

## Repository layout (Phase 2 additions)

```
training/
  tracknet/                        # vendored TrackNetV3 arch (Task 3)
    __init__.py
    model.py                       # the network (adopted from public repo)
  prepare_tracknet.py              # Task 2: dataset -> canonical index + splits
  train_tracknet.py                # Task 4: fine-tune CLI
  evaluate_tracknet.py             # Task 5: held-out metrics + baseline + gate
  export_onnx.py                   # Task 6: EXTENDED (add --tracknet-ckpt)
  configs/tracknet.yaml            # Task 4
  data/raw/tracknet/               # DVC-tracked (Task 1)
  notes/tracknet-format.md         # Task 1 output
core/shuttlesense_core/
  shuttle.py                       # Task 7: smoothing, landing selection, corner constants
  schemas.py                       # Task 8: + Landing, MatchReport.court_corners/landings
backend/app/
  pipeline.py                      # Task 9: + track_shuttle, shuttle stage in analyze()
  shuttle_infer.py                 # Task 9: torch-free TrackNet frame pre/post (parity-tested)
scripts/
  build_samples.py                 # Task 10: + calibration + shuttle stage
  calibrate_sample.py              # Task 10: one-off corner-capture helper
frontend/src/
  lib/heatmap.js                   # Task 11: landing binning helpers
  components/CourtHeatmap.jsx      # Task 12
  pages/Report.jsx                 # Task 12: + heatmap panel
tests/ , backend/tests/            # consistency + pipeline tests
```

---

# Milestone 1 — Data

### Task 1: TrackNet dataset acquisition + DVC + format notes + splits

**Files:**
- Create: `training/data/raw/tracknet/` (DVC-tracked), `training/notes/tracknet-format.md`
- Modify: `training/requirements.txt` (add nothing yet if torch/cv2 present; add `gdown` only if the dataset needs it)

**Interfaces:**
- Produces: the public TrackNet badminton dataset on disk under `training/data/raw/tracknet/`, and `training/notes/tracknet-format.md` documenting the ACTUAL layout: how frames are stored (per-match folders of JPGs vs videos), the label CSV columns (expect per-frame `frame, visibility, x, y` or similar), the coordinate origin/scale, native resolution and fps, and the match→folder mapping. Task 2 is written against this note.

This is the Milestone-1 de-risking spike (Phase 1 Task 5 pattern). The canonical source is the **TrackNetV2/V3 badminton dataset** (search: "TrackNetV3 badminton dataset", author wywyWang / the TrackNetV3 GitHub release). It ships labeled frames, NOT a fine-tuning harness.

- [ ] **Step 1: Install DVC tooling if not already present** — `./.venv/bin/pip install 'dvc[gdrive]>=3.48' gdown` (DVC already initialized in Phase 1; do not re-init).
- [ ] **Step 2: Fetch the dataset** into the scratchpad, then copy the labeled-frame portion into `training/data/raw/tracknet/`. The dataset is distributed as a download (Google Drive / GitHub release); use `gdown` or `curl` per whatever the TrackNetV3 repo's README specifies. If it is only available as videos + label CSVs (no extracted frames), record that in the notes — Task 2 will extract frames.
- [ ] **Step 3: Inspect and document** — run a short python snippet to print: number of matches/rallies/labeled frames, the label CSV columns and a sample of rows, the `visibility` value distribution, native frame resolution, and how matches map to folders. Paste real output into `training/notes/tracknet-format.md`. The note MUST answer: (a) frame storage format, (b) label columns + coordinate convention (origin top-left? pixel units at native res?), (c) visibility encoding, (d) native resolution/fps, (e) match→folder mapping. **Do not proceed to Task 2 until all five are answered with pasted evidence.**
- [ ] **Step 4: Track with DVC (local only, no push)** — `./.venv/bin/dvc add training/data/raw/tracknet` ; commit the `.dvc` pointer + notes + requirements. (No `dvc push` — the gdrive remote OAuth is a user action, per Phase 1's standing ruling; note it in the report.)
- [ ] **Step 5: Commit** — `git add training/data/raw/tracknet.dvc training/notes/tracknet-format.md training/requirements.txt && git commit -m "data: TrackNet badminton dataset (DVC) + format notes"`

---

### Task 2: `prepare_tracknet.py` — canonical sample index + by-match splits

**Files:**
- Create: `training/prepare_tracknet.py`, `training/tests/test_prepare_tracknet.py`

**Interfaces:**
- Consumes: raw TrackNet data (Task 1 layout, per `tracknet-format.md`).
- Produces:
  - `training/data/processed/tracknet_index.parquet` with columns (the contract for Tasks 4/5): `match_id:str, frame_path:str, frame_idx:int, x:float, y:float, visible:int(0|1), orig_w:int, orig_h:int`. Coordinates are in ORIGINAL-resolution pixels; `x=y=-1, visible=0` for frames where the shuttle is absent/occluded.
  - `training/data/processed/tracknet_splits.json`: `{"train":[match_ids], "val":[...], "test":[...]}` — 70/15/15 **by match**, seed 13, sorted-then-shuffled deterministically (reuse the exact `make_splits` logic from Phase 1 `training/build_windows.py` — import it or copy with attribution).
  - `convert(raw_labels_df, match_id) -> pd.DataFrame` exposed for tests; `make_splits(match_ids, seed=13)` (reused).
- If Task 1 found videos not frames, `prepare_tracknet.py` also extracts frames (ffmpeg) into `training/data/processed/tracknet_frames/<match_id>/` and points `frame_path` there.

- [ ] **Step 1: Write the failing test** — `training/tests/test_prepare_tracknet.py`. Build a synthetic raw-label DataFrame mimicking the REAL columns from the notes (adjust names to match Task 1's findings), and assert:

```python
import pandas as pd
from training.prepare_tracknet import convert, make_splits

def _raw():
    # column names per tracknet-format.md — adjust to reality
    return pd.DataFrame({
        "Frame": [0, 1, 2, 3],
        "Visibility": [1, 1, 0, 1],
        "X": [640.0, 642.0, 0.0, 700.0],
        "Y": [360.0, 355.0, 0.0, 300.0],
    })

def test_convert_schema_and_absent_encoding():
    out = convert(_raw(), match_id="m01")
    assert set(out.columns) == {
        "match_id","frame_path","frame_idx","x","y","visible","orig_w","orig_h",
    }
    assert out["visible"].isin([0, 1]).all()
    absent = out[out.visible == 0].iloc[0]
    assert absent["x"] == -1 and absent["y"] == -1   # absent -> sentinel, not raw 0

def test_splits_disjoint_and_deterministic():
    ids = [f"m{i:02d}" for i in range(20)]
    s1, s2 = make_splits(ids), make_splits(ids)
    assert s1 == s2
    assert not (set(s1["train"]) & set(s1["test"]))
    assert set(s1["train"]) | set(s1["val"]) | set(s1["test"]) == set(ids)
```

- [ ] **Step 2: Run to verify failure** — `./.venv/bin/python -m pytest training/tests/test_prepare_tracknet.py -v` → ImportError.
- [ ] **Step 3: Implement `convert` + `make_splits` + `main`.** `convert` maps the raw columns (per notes) to the canonical schema, converting absent/occluded frames (visibility 0) to `x=y=-1`. `make_splits` is Phase 1's verbatim logic. `main` walks the dataset, concatenates, writes `tracknet_index.parquet` + `tracknet_splits.json`, and (if needed) extracts frames.
- [ ] **Step 4: Tests pass;** then run for real: `./.venv/bin/python training/prepare_tracknet.py`; `./.venv/bin/dvc add training/data/processed/tracknet_index.parquet` (frames dir too if extracted). Report row/match counts + visible-fraction.
- [ ] **Step 5: Commit** — `git commit -am "feat: TrackNet dataset -> canonical index + by-match splits"`

---

# Milestone 2 — Model

### Task 3: Vendor TrackNetV3 architecture + load pretrained weights

**Files:**
- Create: `training/tracknet/__init__.py`, `training/tracknet/model.py`, `training/tests/test_tracknet_model.py`
- Modify: `training/requirements.txt` (ensure `torch>=2.6`, present from Phase 1)

**Interfaces:**
- Produces: `TrackNet(in_frames:int=3)` — a `torch.nn.Module`; forward `(B, 3*in_frames, H, W) -> (B, in_frames, H, W)` heatmap logits (one heatmap per input frame), matching the public TrackNetV3 I/O. Constants: `IN_FRAMES = 3`, `IN_H = 288`, `IN_W = 512` (TrackNet native). `load_pretrained(path) -> TrackNet` loads the public checkpoint. Exposes nothing that isn't ONNX-exportable (no dynamic control flow).

Adopt the architecture from the public TrackNetV3 repo rather than hand-writing it (same principle as Phase 1 using rtmlib off-the-shelf) — vendoring a known-correct net is more reliable than reconstructing it.

- [ ] **Step 1: Vendor the model** — copy the TrackNetV3 model definition file from the public repo into `training/tracknet/model.py`, adjusting imports to be self-contained. Keep a one-line provenance comment at the top (repo URL + commit). Ensure the class is importable as `from training.tracknet.model import TrackNet`.
- [ ] **Step 2: Write the failing test** — `training/tests/test_tracknet_model.py`:

```python
import torch
from training.tracknet.model import TrackNet, IN_FRAMES, IN_H, IN_W

def test_forward_shape():
    m = TrackNet(in_frames=IN_FRAMES).eval()
    x = torch.randn(2, 3 * IN_FRAMES, IN_H, IN_W)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (2, IN_FRAMES, IN_H, IN_W)

def test_determinism_eval():
    m = TrackNet().eval()
    x = torch.randn(1, 3 * IN_FRAMES, IN_H, IN_W)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b)
```

(If the vendored net's exact output channel convention differs — e.g. a single fused heatmap rather than per-frame — adjust `IN_FRAMES`/the assertion to the real arch and note it; the CONTRACT that matters downstream is "input a frame stack, output a per-pixel heatmap whose argmax is the shuttle pixel.")
- [ ] **Step 3: Run failing → implement/adjust → passing.**
- [ ] **Step 4: Obtain pretrained weights** — download the public TrackNetV3 pretrained checkpoint into `training/tracknet/` (gitignored; DVC-add it or note it as a fetch step in `notes/tracknet-format.md`). Implement `load_pretrained(path)` and verify it loads without shape errors (a test that skipifs when the weights file is absent, mirroring Phase 1's ONNX skipif).
- [ ] **Step 5: Commit** — `git add training/tracknet training/tests/test_tracknet_model.py && git commit -m "feat: vendor TrackNetV3 architecture + pretrained loader"`

---

### Task 4: `train_tracknet.py` — fine-tune CLI

**Files:**
- Create: `training/train_tracknet.py`, `training/configs/tracknet.yaml`, `training/tests/test_train_tracknet_smoke.py`

**Interfaces:**
- Consumes: `tracknet_index.parquet`, `tracknet_splits.json`, `TrackNet`/`load_pretrained`, and the shared frame pre-processing from `core` (Task 7 — but Task 7 lands later; for Task 4 put the preprocessing in `core/shuttlesense_core/shuttle.py` as its FIRST function so Task 7 extends the same module. Define here: `frames_to_input(frames: np.ndarray) -> np.ndarray` — stack of `IN_FRAMES` RGB frames `(IN_FRAMES,H,W,3)` uint8 → `(3*IN_FRAMES, IN_H, IN_W)` float32 normalized [0,1], resized. And `heatmap_target(x, y, visible, H, W, sigma=3.0) -> np.ndarray` Gaussian target.)
- Produces: checkpoint `training/checkpoints/tracknet/best.pt` = `{"state_dict","config","val_f1","val_pixel_err"}`. CLI: `python training/train_tracknet.py --config training/configs/tracknet.yaml [--data ...] [--splits ...] [--pretrained ...] [--out-dir ...] [--epochs N] [--wandb]`. Follows Phase 1 `train_stroke.py` conventions verbatim: `set_seed`/`resolve_splits` from `training/common.py`, device handling (cuda-if-available), W&B offline fallback, `--epochs is not None` semantics + `epochs>0` validation, empty-train raise, split-disjointness enforcement.

- [ ] **Step 1: Config** — `training/configs/tracknet.yaml`: `seed: 13`, `epochs: 20`, `batch_size: 8`, `lr: 0.001`, `weight_decay: 0.0001`, `in_frames: 3`, `input_h: 288`, `input_w: 512`, `heatmap_sigma: 3.0`, `subset_matches: null` (int to cap matches for free-GPU sessions; null = all).
- [ ] **Step 2: Failing smoke test** — `training/tests/test_train_tracknet_smoke.py`: synthesize a tiny index (a handful of solid-color frames written to a tmp dir + a few rows), run the CLI with `--epochs 1 --subset ...` against it, assert `best.pt` exists with keys `{state_dict, config, val_f1, val_pixel_err}`. Loss = per-pixel BCE (or MSE) between predicted heatmap and Gaussian target; val metrics computed by argmax-vs-label within a pixel tolerance.

```python
import json, subprocess, sys
def test_train_smoke(tmp_path):
    # build a tiny frames dir + index parquet + splits.json (2 matches)
    ...
    r = subprocess.run([sys.executable, "training/train_tracknet.py",
        "--config","training/configs/tracknet.yaml","--data",str(idx),
        "--splits",str(splits),"--out-dir",str(tmp_path/"ck"),"--epochs","1"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import torch
    ck = torch.load(tmp_path/"ck"/"best.pt", map_location="cpu", weights_only=False)
    assert {"state_dict","config","val_f1","val_pixel_err"} <= set(ck)
```

- [ ] **Step 3: Run failing → implement.** Dataset loads `IN_FRAMES`-frame stacks centered on each labeled frame via `frames_to_input`; targets via `heatmap_target`; fine-tunes from `--pretrained`; best-by-val-F1 checkpoint; W&B logs loss/F1/pixel-err.
- [ ] **Step 4: Smoke test passes.**
- [ ] **Step 5: Commit** — `git commit -am "feat: TrackNet fine-tune CLI (W&B, seeded, from pretrained)"`

---

### Task 5: `evaluate_tracknet.py` + real fine-tune run + quality gate

**Files:**
- Create: `training/evaluate_tracknet.py`, `training/reports/` (a dated tracknet eval report), `training/notes/colab-tracknet.md`

**Interfaces:**
- Consumes: the fine-tuned checkpoint, the pretrained weights, `tracknet_index.parquet` + `tracknet_splits.json`.
- Produces: `training/reports/YYYY-MM-DD-tracknet-eval.md` (committed) with: dataset sizes per split (+ match counts), **fine-tuned vs pretrained-baseline** detection precision/recall/F1 (within the dataset-standard pixel tolerance — state the tolerance) and mean pixel error, on the TEST split (never val). CLI: `python training/evaluate_tracknet.py --ckpt ... --pretrained ... --out training/reports/<date>-tracknet-eval.md`.

- [ ] **Step 1: Colab notes** — `training/notes/colab-tracknet.md`: how to fine-tune on Colab (pip install, mount Drive, run the CLI with `--wandb`), plus the `subset_matches` guidance for session limits.
- [ ] **Step 2: Real fine-tune** — run `train_tracknet.py --config ... --wandb` on Colab GPU (or locally if feasible). Record W&B run URL + wall time. If sessions overrun, set `subset_matches` and note it.
- [ ] **Step 3: Implement `evaluate_tracknet.py`** — load both models (fine-tuned + pretrained-as-is), run each over the test split, compute F1@tolerance + mean pixel error, render the markdown report with both columns side by side. Reuse `resolve_splits` from `training/common.py` for the split loading + disjointness check.
- [ ] **Step 4: Quality gate** — fine-tuned test F1 must exceed pretrained-baseline test F1. If it does NOT: STOP, report the negative result in the eval report, and flag to the controller — per the spec, v1 then ships pretrained-as-is (Task 6 exports whichever won; record which in the manifest).
- [ ] **Step 5: Commit** — `git add training/evaluate_tracknet.py training/reports training/notes/colab-tracknet.md && git commit -m "feat: TrackNet eval vs pretrained baseline + gate; first eval report"`

---

### Task 6: ONNX export + manifest entry + parity/consistency tests

**Files:**
- Modify: `training/export_onnx.py` (add `--tracknet-ckpt`), `backend/models/manifest.json` (committed), `tests/test_consistency.py`
- Create: (in Task 9 the backend consumer; here just the export + parity)

**Interfaces:**
- Produces: `backend/models/tracknet.onnx` (input `x:(B, 3*IN_FRAMES, IN_H, IN_W)` float32, dynamic batch; output `logits:(B, IN_FRAMES, IN_H, IN_W)`); a `"tracknet"` entry in `manifest.json` = `{"file":"tracknet.onnx","val_f1","val_pixel_err","f1_tolerance_px","git_sha","in_frames","input_h","input_w","source":"finetuned"|"pretrained"}`. Uses the legacy exporter (`dynamo=False`, per Phase 1 Task 13 — torch ≥2.6).

- [ ] **Step 1: Failing parity test** — extend `tests/test_consistency.py`:

```python
def test_tracknet_onnx_matches_torch(tmp_path):
    import onnxruntime as ort, torch, numpy as np, subprocess, sys
    from training.tracknet.model import TrackNet, IN_FRAMES, IN_H, IN_W
    m = TrackNet(in_frames=IN_FRAMES); m.eval()
    ck = {"state_dict": m.state_dict(), "config": {"in_frames": IN_FRAMES},
          "val_f1": 0.0, "val_pixel_err": 0.0}
    torch.save(ck, tmp_path/"tn.pt")
    r = subprocess.run([sys.executable, "training/export_onnx.py",
        "--tracknet-ckpt", str(tmp_path/"tn.pt"), "--skip-stroke", "--skip-rally",
        "--out-dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    x = np.random.default_rng(0).normal(size=(2, 3*IN_FRAMES, IN_H, IN_W)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    out = ort.InferenceSession(str(tmp_path/"tracknet.onnx")).run(None, {"x": x})[0]
    np.testing.assert_allclose(out, ref, atol=1e-4)
```

- [ ] **Step 2: Implement** — add `export_tracknet(ckpt, out_dir)` to `export_onnx.py` (rebuild `TrackNet` from stored config, `eval()`, `torch.onnx.export` with `dynamic_axes={"x":{0:"B"},"logits":{0:"B"}}`, `dynamo=False`); add the manifest entry (merge-before-write so `--skip-*` doesn't clobber siblings); `--tracknet-ckpt`/`--skip-tracknet` flags. Also add the train/serve consistency assertion that `backend.app.shuttle_infer` imports the shared `frames_to_input`/heatmap-argmax post-proc from `core` (this half `xfail`s until Task 9 creates `shuttle_infer.py` — mark `xfail(strict=True)`, remove in Task 9).
- [ ] **Step 3: Run tests** — parity passes with a real ONNX; consistency-import xfails.
- [ ] **Step 4: Export the real model** — `python training/export_onnx.py --tracknet-ckpt training/checkpoints/tracknet/best.pt --skip-stroke --skip-rally --wandb`; commit `manifest.json` (onnx stays gitignored until Task 13 decides commit-vs-LFS).
- [ ] **Step 5: Commit** — `git add training/export_onnx.py backend/models/manifest.json tests/test_consistency.py && git commit -m "feat: TrackNet ONNX export + manifest entry + parity test"`

---

# Milestone 3 — Serving

### Task 7: `core/shuttlesense_core/shuttle.py` — smoothing, landing, corner constants

**Files:**
- Modify: `core/shuttlesense_core/shuttle.py` (created in Task 4 with `frames_to_input`/`heatmap_target`; extend it)
- Test: `core/tests/test_shuttle.py`

**Interfaces:**
- Produces (pure numpy; shared by training + serving):
  - `CORNER_ORDER = ["near_left","near_right","far_right","far_left"]` and `COURT_CORNERS_M = np.array([(0,0),(COURT_W,0),(COURT_W,COURT_L),(0,COURT_L)], float64)` (import `COURT_W/COURT_L` from `.homography`).
  - `heatmap_to_xy(heatmap: np.ndarray, orig_w:int, orig_h:int) -> tuple[float,float,float]` — a single `(IN_H,IN_W)` heatmap → `(x, y, confidence)` in ORIGINAL-resolution pixels (argmax location rescaled from input res to orig res; confidence = peak value). Sub-input-resolution rounding documented.
  - `smooth_track(track: np.ndarray, min_conf: float = 0.5, max_gap: int = 5) -> np.ndarray` — `(T,3)` array of `(x,y,conf)` per frame → cleaned `(T,3)`: frames below `min_conf` are marked missing (`conf=0`), gaps ≤ `max_gap` linearly interpolated, longer gaps left missing. Never invents positions across long gaps.
  - `rally_end_landing(track: np.ndarray, start: int, end: int) -> tuple[float,float] | None` — the shuttle `(x,y)` at the last detected (conf>0) frame within `[start, end)`; `None` if the shuttle is missing throughout the interval.

- [ ] **Step 1: Failing tests** — `core/tests/test_shuttle.py`:

```python
import numpy as np
from shuttlesense_core.shuttle import (
    smooth_track, rally_end_landing, heatmap_to_xy, COURT_CORNERS_M, CORNER_ORDER,
)

def test_heatmap_to_xy_rescales_argmax():
    hm = np.zeros((288, 512), np.float32); hm[144, 256] = 1.0
    x, y, c = heatmap_to_xy(hm, orig_w=1280, orig_h=720)
    assert abs(x - 640) < 3 and abs(y - 360) < 3 and c == 1.0

def test_smooth_interpolates_short_gap_only():
    t = np.array([[100,100,0.9],[0,0,0.0],[104,104,0.9],
                  [0,0,0.0],[0,0,0.0],[0,0,0.0],[0,0,0.0],[0,0,0.0],[0,0,0.0]], float)
    out = smooth_track(t, min_conf=0.5, max_gap=1)
    assert out[1,2] > 0 and abs(out[1,0]-102) < 1e-6   # 1-frame gap filled
    assert out[4,2] == 0                                # long gap left missing

def test_rally_end_landing_last_detected():
    t = np.array([[10,10,0.9],[12,12,0.9],[0,0,0.0]], float)
    assert rally_end_landing(t, 0, 3) == (12.0, 12.0)   # last conf>0 frame
    assert rally_end_landing(np.zeros((3,3)), 0, 3) is None

def test_corner_constants():
    assert CORNER_ORDER[0] == "near_left"
    assert COURT_CORNERS_M.shape == (4, 2)
    assert tuple(COURT_CORNERS_M[0]) == (0.0, 0.0)
```

- [ ] **Step 2: Run failing → implement → passing.**
- [ ] **Step 3: Commit** — `git add core/shuttlesense_core/shuttle.py core/tests/test_shuttle.py && git commit -m "feat: core shuttle smoothing, landing selection, court-corner constants"`

---

### Task 8: Schema extension — `Landing` + `MatchReport.court_corners/landings`

**Files:**
- Modify: `core/shuttlesense_core/schemas.py`
- Test: `core/tests/test_schemas.py` (extend)

**Interfaces:**
- Produces: `@dataclass Landing(rally_id:int, frame:int, court_x:float, court_y:float, winner:int|None=None)`; `MatchReport` gains `court_corners: list[list[float]] | None = None` (4 `[x,y]` image-pixel points in `CORNER_ORDER`) and `landings: list[Landing] = field(default_factory=list)`. `to_dict`/`from_dict` round-trip including the new fields; `from_dict` DEFAULTS them so Phase 1 reports (no such keys) still load.

- [ ] **Step 1: Failing test** — extend `core/tests/test_schemas.py`:

```python
def test_report_landings_roundtrip_and_backcompat():
    from shuttlesense_core.schemas import MatchReport, Landing, RallyInterval
    r = MatchReport(fps=15.0, width=1280, height=720, n_frames=300,
                    rallies=[RallyInterval(0,100,winner=1)],
                    court_corners=[[300,600],[980,600],[1180,100],[100,100]],
                    landings=[Landing(rally_id=0, frame=99, court_x=3.0, court_y=11.0, winner=1)])
    d = r.to_dict()
    assert d["court_corners"][0] == [300,600]
    assert d["landings"][0]["court_x"] == 3.0
    assert MatchReport.from_dict(d) == r
    # back-compat: a Phase 1 dict with no court_corners/landings still loads
    old = {"fps":15.0,"width":1280,"height":720,"n_frames":300,"rallies":[],"strokes":[]}
    m = MatchReport.from_dict(old)
    assert m.court_corners is None and m.landings == []
```

- [ ] **Step 2: Run failing → implement → passing** (add fields + `Landing`; `from_dict` uses `d.get("court_corners")` / `[Landing(**x) for x in d.get("landings", [])]`).
- [ ] **Step 3: Commit** — `git commit -am "feat: MatchReport court_corners + landings (back-compatible)"`

---

### Task 9: Backend shuttle stage — `shuttle_infer.py` + `analyze(court_corners=...)`

**Files:**
- Create: `backend/app/shuttle_infer.py`
- Modify: `backend/app/pipeline.py`, `tests/test_consistency.py` (remove the Task 6 xfail)
- Test: `backend/tests/test_shuttle_pipeline.py`

**Interfaces:**
- `shuttle_infer.py` (torch-free): `extract_frames_onnx(video_path, target_fps) -> np.ndarray (T,H,W,3) uint8` (thin cv2 decode at the SAME sampled fps as pose extraction so shuttle-frame indices align with `report` frame indices); `track_shuttle(video_path, models_dir, target_fps, frames=None) -> np.ndarray (T,3)` — runs `tracknet.onnx` over sliding `IN_FRAMES` stacks (uses `core.shuttle.frames_to_input`), converts each heatmap via `core.shuttle.heatmap_to_xy`, returns the per-frame `(x,y,conf)` track. `frames` injectable for tests.
- `pipeline.analyze` gains params `court_corners: list | None = None`, `track_fn=None`. When `court_corners` is not None AND a `tracknet.onnx` exists: after building `report`, call `track_fn or track_shuttle`, `smooth_track`, then for each rally `rally_end_landing` → `homography.fit_homography(np.array(court_corners), COURT_CORNERS_M)` once → `to_court` the landing px → append `Landing(rally_id=i, frame=<end-1 or last-detected>, court_x, court_y, winner=rally.winner)`. Set `report.court_corners = court_corners`. When `court_corners` is None: unchanged Phase 1 behavior (no landings). Returns `(report, tracks)` as before (landings live on `report`).

- [ ] **Step 1: Failing pipeline test** — `backend/tests/test_shuttle_pipeline.py`, injecting BOTH a fake `pose_fn` (Phase 1 pattern, produces ≥1 rally) and a fake `track_fn` (synthetic shuttle at a known pixel that maps to a known court coordinate):

```python
import numpy as np
from backend.app.pipeline import analyze

def fake_track(video_path, models_dir, target_fps, frames=None):
    T = 300
    tr = np.zeros((T,3), float)
    tr[:,0] = 640; tr[:,1] = 360; tr[:,2] = 0.9    # shuttle centered, always visible
    return tr

def test_landings_mapped_to_court(tmp_path, monkeypatch, fake_pose_with_one_rally):
    corners = [[300,600],[980,600],[1180,100],[100,100]]  # CORNER_ORDER
    report, tracks = analyze("x.mp4", models_dir="backend/models",
                             pose_fn=fake_pose_with_one_rally, track_fn=fake_track,
                             court_corners=corners)
    assert report.court_corners == corners
    assert len(report.landings) >= 1
    ld = report.landings[0]
    assert 0 <= ld.court_x <= 6.1 and 0 <= ld.court_y <= 13.4   # inside the court
    # no corners -> no landings (back-compat path)
    r2, _ = analyze("x.mp4", models_dir="backend/models",
                    pose_fn=fake_pose_with_one_rally)
    assert r2.landings == [] and r2.court_corners is None
```

(Gate the tracknet-onnx-requiring parts with skipif when `backend/models/tracknet.onnx` is absent, like Phase 1's ONNX tests; the injected-`track_fn` path needs no onnx and must run always.)
- [ ] **Step 2: Run failing → implement `shuttle_infer.py` + wire `analyze`.** Remove the Task 6 xfail on the consistency import test; make `shuttle_infer` import `frames_to_input`/`heatmap_to_xy` from `core.shuttlesense_core.shuttle` so the identity holds.
- [ ] **Step 3: Tests pass** (`./.venv/bin/python -m pytest backend/tests/test_shuttle_pipeline.py tests/test_consistency.py -v`).
- [ ] **Step 4: Commit** — `git commit -am "feat: backend shuttle-tracking stage + court-mapped landings in analyze()"`

---

### Task 10: Sample calibration + regenerate sample reports with landings

**Files:**
- Create: `scripts/calibrate_sample.py`
- Modify: `scripts/build_samples.py`, `scripts/samples.yaml`
- (Regenerates: `backend/samples/<id>/report.json`)

**Interfaces:**
- `calibrate_sample.py`: a one-off helper the human runs to capture the 4 court corners for a sample's first frame (opens the frame, click 4 points in `CORNER_ORDER`, or accepts them as CLI args); writes `corners` into that sample's entry in `scripts/samples.yaml`.
- `build_samples.py`: for a sample whose `samples.yaml` entry has `corners`, pass them as `analyze(..., court_corners=corners)`; the resulting `report.json` then carries `court_corners` + `landings`. The re-encode / meta / winner-patching flow is unchanged.

- [ ] **Step 1: Implement `calibrate_sample.py`** — given a sample id, extract frame 0 of its analysis clip, and either (a) render it for the human to click 4 corners (matplotlib ginput / a tiny cv2 window), or (b) accept `--corners "x1,y1;x2,y2;x3,y3;x4,y4"`. Validate exactly 4 points, write them under the sample's `samples.yaml` entry. Document the required `CORNER_ORDER`.
- [ ] **Step 2: Calibrate the two samples** — the human (controller) runs `calibrate_sample.py` for `chou-tien-chen-vs-antonsen-fuzhou-2019` and `vittinghus-vs-antonsen-thailand-2021`, clicking the 4 court corners on each first frame. Commit the corners into `scripts/samples.yaml`.
- [ ] **Step 3: Wire + regenerate** — `build_samples.py` reads `corners`, calls `analyze(court_corners=...)`; re-run `build_samples.py` for both samples; verify each `report.json` now has `court_corners` + a non-empty `landings` list, and spot-check that a couple of landing court-coordinates fall in sensible court regions.
- [ ] **Step 4: Commit** — `git add scripts/calibrate_sample.py scripts/build_samples.py scripts/samples.yaml backend/samples && git commit -m "feat: sample court calibration + regenerated reports with shot landings"`

---

# Milestone 4 — Frontend

### Task 11: Landing-binning helpers (`lib/heatmap.js`)

**Files:**
- Create: `frontend/src/lib/heatmap.js`, `frontend/src/lib/heatmap.test.js`

**Interfaces:**
- Pure functions (vitest-tested with exact values):
  - `binLandings(landings, {cols, rows}) -> {counts: number[][], max: number}` — bins each landing's `(court_x, court_y)` into a `rows×cols` grid over the `6.1×13.4 m` court; returns per-cell counts + the max count (for ramp scaling). Landings exactly on an edge go to the lower cell; out-of-court landings are clamped.
  - `filterLandings(landings, {player, outcome}) -> landings` — `player` filters by `winner`, `outcome` in `{"all","won","lost"}` relative to a given player; `null`/`"all"` = no filter.

- [ ] **Step 1: Failing tests** — `frontend/src/lib/heatmap.test.js`:

```js
import { binLandings, filterLandings } from "./heatmap.js";
import { expect, test } from "vitest";

test("binLandings counts into grid cells with max", () => {
  const lds = [{court_x:0.1,court_y:0.1},{court_x:0.2,court_y:0.2},{court_x:6.0,court_y:13.0}];
  const { counts, max } = binLandings(lds, {cols:2, rows:2});
  expect(counts[0][0]).toBe(2);   // two near-origin
  expect(counts[1][1]).toBe(1);   // one far corner
  expect(max).toBe(2);
});

test("filterLandings by outcome", () => {
  const lds = [{winner:0},{winner:1},{winner:0}];
  expect(filterLandings(lds, {outcome:"all"}).length).toBe(3);
  expect(filterLandings(lds, {player:0, outcome:"won"}).length).toBe(2);
});
```

- [ ] **Step 2: Run failing → implement → passing** (`npm run test`).
- [ ] **Step 3: Commit** — `git add frontend/src/lib/heatmap.js frontend/src/lib/heatmap.test.js && git commit -m "feat: court-heatmap landing binning + filter helpers"`

---

### Task 12: `CourtHeatmap.jsx` + report integration

**Files:**
- Create: `frontend/src/components/CourtHeatmap.jsx`, `frontend/src/components/CourtHeatmap.css`, `frontend/src/components/CourtHeatmap.test.jsx`
- Modify: `frontend/src/pages/Report.jsx`

**Interfaces:**
- `<CourtHeatmap report={} players={} />` — renders only when `report.landings?.length`; otherwise a muted "Shot placement needs court calibration — coming soon" note. Draws a to-scale top-down court (SVG, `6.1×13.4 m` → viewbox with margin), court lines in recessive ink; bins landings via `binLandings` and paints cells with a **sequential single-hue ramp** (light→dark by count/max — a DISTINCT hue from the skeleton green/blue and momentum's neutral race-lines); overlays landing dots; a filter row (player, outcome) above; a legend + per-cell hover tooltip with the count. Before implementing, load the **dataviz skill** guidance for the sequential ramp and run the palette validator on the chosen hue.
- `Report.jsx`: add a `<CourtHeatmap report={report} players={players} />` panel below the rally list.

- [ ] **Step 1: Load dataviz guidance** — invoke the dataviz skill (or, if unavailable to the subagent, the controller relays: sequential single-hue ramp light→dark, one hue not used elsewhere; recessive court lines in ink tokens; legend present; hover tooltip; validate the ramp). Pick the hue; keep it off green/blue.
- [ ] **Step 2: Failing component test** — `CourtHeatmap.test.jsx`: render with a report having 3 landings → assert grid cells render and the "coming soon" note is ABSENT; render with `landings: []` → assert the note is present and no heatmap cells. (Uses the RTL setup added in Phase 1 Task 18.)
- [ ] **Step 3: Run failing → implement Component + CSS + wire into Report → passing.** `npm run test` + `npm run build` + `npm run lint` all green.
- [ ] **Step 4: Manual check** — with the app running (worktree venv + `npm run dev` or built dist), open a calibrated sample and confirm the heatmap renders with landings in plausible court regions; screenshot for the report.
- [ ] **Step 5: Commit** — `git add frontend/src/components/CourtHeatmap.* frontend/src/pages/Report.jsx && git commit -m "feat: top-down court shot-placement heatmap panel"`

---

# Milestone 5 — Deploy

### Task 13: Bake TrackNet ONNX + README/GIF update + final verification

**Files:**
- Modify: `.gitignore` (allow `tracknet.onnx` if committing), `README.md`, `docs/media/` (new heatmap GIF/screenshot), `NEXT-STEPS.md`

**Interfaces:**
- Produces: `backend/models/tracknet.onnx` available to the Docker build (same call as Phase 1's sample videos: commit to git if ≤ ~25 MB; else quantize/LFS and document). README gains a Phase 2 section with the TrackNet metrics (from `training/reports/<date>-tracknet-eval.md`, with the pixel-tolerance + subset caveats) and a heatmap in the demo media. Dockerfile needs NO change if `backend/models/*.onnx` is already copied (it is) — just ensure `tracknet.onnx` isn't excluded.

- [ ] **Step 1: Size call + commit the model** — check `du -h backend/models/tracknet.onnx`. If ≤ ~25 MB, `git add -f` it (add `!backend/models/tracknet.onnx` to `.gitignore` for clarity, mirroring the Phase 1 onnx negation). If larger, quantize (onnxruntime dynamic quantization) or note LFS; record the decision.
- [ ] **Step 2: Verify the built image serves shuttle** — `docker build` (linux/amd64 per Phase 1) and run; confirm a calibrated sample's `report.json` includes `landings` via `curl .../api/samples/<id>/report`, and the SPA renders the heatmap. If docker is unavailable, do the worktree-venv serve-check (Phase 1 pattern) and say so.
- [ ] **Step 3: README + media** — add the "Phase 2: shot placement" section (what it is, the fine-tuned-vs-baseline metric, the manual-calibration + v2 deferrals stated honestly); regenerate/add a heatmap frame to the demo media. Update `NEXT-STEPS.md` (Phase 2 status; amateur domain-gap + upload-calibration + in-video overlay remain v2).
- [ ] **Step 4: Full-suite verification** — `./.venv/bin/python -m pytest -q` (backend+core+training) and `cd frontend && npm run test` both green; paste counts.
- [ ] **Step 5: Commit + push** — `git add -A && git commit -m "feat: bake TrackNet model, README + demo media for shot-placement heatmap"` (push is a controller/user action at finish time).

---

## Self-review (completed during plan writing)

1. **Spec coverage:** §5 data/model → Tasks 1-6; §6 serving pipeline (track→smooth→landings→map, `report.json` fields, samples-only calibration) → Tasks 7-10; §7 frontend heatmap → Tasks 11-12; §8 testing → throughout (consistency test extended in Task 6/9, parity in Task 6, pipeline in Task 9, binning in Task 11); §9 milestones → the 5 milestone groupings; §2 out-of-scope (upload calibration UI, in-video overlay, amateur domain-gap, auto court detection, arc-fitting) → explicitly deferred, no tasks. The manual-calibration linchpin → Task 10 (+ the `court_corners` plumbing in Tasks 8-9).
2. **Placeholder scan:** The two external-artifact steps (Task 1 dataset fetch, Task 3 vendored TrackNet arch + pretrained weights) are concrete "obtain X from the named public source, then verify Y" instructions with loud failure modes and a documented note file — the Phase 1 rtmlib/ShuttleSet pattern, not TBDs. Task 3's forward-shape assertion explicitly allows adjusting to the real arch's channel convention while pinning the load-bearing contract (frame-stack in, heatmap out). No "TODO"/"handle edge cases"/uncoded steps.
3. **Type consistency:** `frames_to_input`/`heatmap_target` introduced in Task 4 and extended in Task 7 live in one module (`core/shuttlesense_core/shuttle.py`); `heatmap_to_xy`/`smooth_track`/`rally_end_landing` signatures are consistent across Tasks 7 and 9; `Landing`/`court_corners`/`landings` names match across Tasks 8, 9, 10, 11, 12; `CORNER_ORDER`/`COURT_CORNERS_M` consistent across Tasks 7, 9, 10; the ONNX io names (`x`/`logits`) and manifest `"tracknet"` shape consistent across Tasks 6 and 9; `analyze(court_corners=, track_fn=)` consistent across Tasks 9 and 10.
