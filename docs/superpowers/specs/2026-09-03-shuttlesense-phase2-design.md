# ShuttleSense Phase 2 — Shuttle Tracking & Shot-Placement Analytics

**Date:** 2026-09-03
**Status:** Approved design, pre-implementation
**Author:** Anikesh (b.anikesh@gmail.com), designed collaboratively with Claude
**Builds on:** Phase 1 (`docs/superpowers/specs/2026-08-26-shuttlesense-design.md`), merged to `main`

## 1. Purpose

Phase 2 adds the shot-placement half of ShuttleSense: track the shuttle through a
rally, detect where each rally's shuttle ends up on the court, and visualize shot
placement as a top-down court heatmap. It extends the completed Phase 1 pipeline
(pose → stroke/rally, ONNX-CPU serving, React report) rather than replacing any of
it, and reuses the court-homography math already shipped in
`core/shuttlesense_core/homography.py`.

**Portfolio angle:** demonstrates transfer learning (fine-tuning a pretrained
TrackNetV3 shuttle tracker) with an honest before/after quality gate, on top of the
Phase 1 from-scratch classifier work.

**Success criteria:**
- The live demo's pre-analyzed sample matches gain a top-down court heatmap of shot
  placement, shown with zero user interaction.
- The README can honestly say: fine-tuned TrackNetV3, evaluated against the
  pretrained baseline on a held-out by-match split, beating it.
- Nothing half-built ships: an uploaded video that lacks calibration shows a clear
  "coming soon" note in the heatmap slot, not a broken panel.

## 2. Scope

**In scope (Phase 2 v1):**
- Fine-tune pretrained TrackNetV3 on the public TrackNet badminton dataset; eval vs
  the pretrained baseline; ONNX export + registry.
- A shuttle-tracking stage in the serving pipeline: track → smooth → detect landings
  → map to court coordinates.
- Manual 4-corner court calibration, applied once at sample-build time (baked into
  the sample reports).
- A top-down court placement heatmap on the report page, rendered for the
  pre-calibrated samples.
- Run the tracker on the existing 12 broadcast matches to produce the sample
  heatmaps.

**Out of scope (v2+), explicitly:**
- Click-to-calibrate calibration UI for uploaded videos (v1 uploads keep the Phase 1
  report with no heatmap + a "coming soon" note).
- The in-video perspective heat overlay (heat painted onto the court inside the
  video frame) — the harder hero visual; top-down heatmap first.
- Amateur / own-footage domain-gap evaluation — gated on the author capturing and
  labeling phone-camera footage, which does not exist yet.
- Automatic court detection (classical or learned) — manual calibration is the v1
  choice; auto-detection is a possible later replacement behind the same interface.
- Shuttle-physics arc-fitting for landing detection — v1 uses the rally-end heuristic
  (§6).

## 3. Key design decisions (from brainstorming)

| Decision | Choice | Why |
|---|---|---|
| Court→image homography | **Manual 4-corner calibration** | Zero training/GPU, robust on any footage; `fit_homography`/`to_court` already exist. Keeps the entire free-GPU budget for TrackNet. |
| Shuttle tracker | **Fine-tune pretrained TrackNetV3** | Fits free Colab/Kaggle; real transfer-learning story + domain-gap eval, without training a frame-level CNN from scratch. |
| v1 eval/target | **Broadcast now, amateur later** | Fine-tune + quantitative eval on public labeled data; run on the existing 12 broadcast matches for the sample heatmaps; defer amateur domain-gap until footage exists. |
| Integration | **Extend the Phase 1 batch pipeline (Approach A)** | Reuses ONNX-CPU serving, the JSON-report contract, the training spine, and the single-container deploy. |
| Calibration scope | **Samples-only in v1** | Samples pre-calibrated at build time → zero-action demo heatmaps; upload calibration UI is v2. |

## 4. Architecture (delta over Phase 1)

```
training/  (offline, free Colab GPU)
  + tracknet dataset prep (DVC)          public TrackNetV2/V3 badminton data
  + train_tracknet.py (fine-tune)        from pretrained weights, W&B, seeded
  + evaluate_tracknet.py                 held-out by-match split + pretrained baseline + gate
  + export_onnx.py (extended)            tracknet.onnx + manifest.json entry

core/shuttlesense_core/
  + shuttle.py                           trajectory smoothing, landing heuristic, heatmap binning
                                         (pure; SHARED by training and serving — consistency-tested)

backend/  (serving, ONNX-CPU, torch-free)
  pipeline.py  + shuttle stage           track -> smooth -> landings -> to_court(H)
  report.json  + court_corners, landings new fields (present only when calibrated)

frontend/  (React)
  + CourtHeatmap.jsx                      top-down court, binned density + landing dots, filters
  Report.jsx  + heatmap panel             rendered iff report.landings present, else "coming soon"

scripts/build_samples.py  + calibration   click 4 corners once per sample; bake corners + landings
```

## 5. Data & model (shuttle tracker)

- **Dataset:** public TrackNetV2/V3 badminton dataset (broadcast rally clips with
  per-frame shuttle `(x, y, visibility)` labels). DVC-tracked like Phase 1's
  ShuttleSet. **By-match** train/val/test split (never random-by-frame — same
  no-leakage discipline as Phase 1).
- **Model:** TrackNetV3 — input a small stack of consecutive frames, output a
  per-pixel heatmap; shuttle position = heatmap argmax. Start from public pretrained
  weights, **fine-tune**.
- **Training (free-GPU discipline):** native low input resolution (~288×512), small
  batch, few epochs, gradient checkpointing, resumable seeded checkpoints, W&B-tracked,
  YAML config — the Phase 1 `training/` CLI conventions.
- **Eval + quality gate:** on the held-out **by-match** test split — detection
  precision/recall/F1 within a pixel tolerance, plus mean positioning error. **Gate:
  the fine-tuned model must beat the pretrained-as-is baseline** on that split.
  Result written to a committed eval report.
- **Compute honesty:** if full-dataset fine-tuning overruns free-Colab sessions,
  subset to fewer matches and state it in the eval report (Phase 1's "no silent caps"
  rule).
- **Export:** ONNX with dynamic axes; an entry in `backend/models/manifest.json`
  alongside the stroke/rally models (with the metric's measurement conditions
  recorded, per Phase 1's threshold-provenance lesson); train/serve consistency test
  on the shared pre/post-processing.

## 6. Serving pipeline (new stage)

Runs in the analysis worker after the Phase 1 stages, all ONNX-CPU (backend stays
torch-free — TrackNet's own thin frame-stack pre/post-processing is duplicated in
`backend/` from `core/` where a torch-free copy is needed, with the same anti-drift
parity test Phase 1 used for pose extraction):

1. **Track** — frame stack → TrackNet ONNX → per-frame `(x, y, confidence)`.
2. **Smooth** — reject low-confidence detections, fill short gaps (`core/shuttlesense_core/shuttle.py`, shared with training).
3. **Detect landings** — v1 heuristic: the shuttle's position at each **rally's end
   frame** (rally intervals come from the Phase 1 segmenter) is that rally's
   landing/terminal point. No arc-fitting in v1; stated as such in the report.
4. **Map** — `to_court(H, landing_px)` using the match's calibrated `court_corners`.
5. **Emit** — `report.json` gains `court_corners` (the 4-corner calibration) and
   `landings: [{rally_id, frame, court_x, court_y, winner}]`. Ship the landings
   **list**, not a pre-binned grid — the frontend bins it (Phase 1 pattern: ship raw,
   render client-side; small payload, tunable binning).

Uploads omit `court_corners`/`landings`; the frontend renders the "coming soon" note.

## 7. Frontend — top-down court heatmap

`CourtHeatmap.jsx`, a panel below the rally list, rendered only when
`report.landings` is present:
- To-scale top-down court in SVG (6.1 × 13.4 m), court lines in recessive ink.
- Landings binned into a density heatmap with a **sequential single-hue ramp**
  (light→dark) — a distinct color channel from the skeleton green/blue and from
  momentum's neutral race-lines; individual landing dots overlaid.
- Filters (one row above): by player, by rally outcome. Legend + hover tooltip with
  bin count. The ramp is run through the dataviz palette validator.
- Pure binning helpers unit-tested with vitest.

## 8. Testing

- Unit (`core/`): trajectory smoothing, landing heuristic, heatmap binning.
- Train/serve consistency test extended to the shuttle pre/post-processing.
- ONNX parity: fine-tuned TrackNet torch vs onnxruntime, `atol`-bounded.
- Pipeline test with an injected fake tracker (synthetic trajectory) → landings map
  to the correct court coordinates.
- Frontend: `CourtHeatmap` binning helpers (vitest).
- Quality gate (eval report, not pytest): fine-tuned > pretrained baseline on the
  held-out by-match split.

## 9. Milestones (v1)

Each independently reviewable, mirroring Phase 1's task cadence:

1. **Data** — TrackNet dataset acquisition (DVC), format notes, by-match splits.
2. **Model** — fine-tune CLI + config (W&B, seeded); `evaluate_tracknet.py` with the
   pretrained baseline + quality gate; ONNX export + manifest entry + parity /
   consistency tests.
3. **Serving** — shared `core/shuttlesense_core/shuttle.py`; the pipeline shuttle
   stage; `landings`/`court_corners` in `report.json`; sample calibration in
   `build_samples.py`; regenerate the sample reports.
4. **Frontend** — `CourtHeatmap` + report integration + binning tests.
5. **Deploy** — bake `tracknet.onnx` into the image (commit if small enough, else the
   size-handling approach used for the sample videos); README update with the new
   metrics + a heatmap in the demo GIF.

**v1/v2 gate:** if anything slips, v1 (samples heatmaps, live) stands alone. v2 =
upload calibration UI + in-video perspective overlay + amateur-footage domain-gap
eval.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| TrackNet fine-tuning overruns free Colab | Subset matches, low res, few epochs, resumable checkpoints; disclose the subset in the eval report |
| Fine-tuned model doesn't beat the pretrained baseline | The gate catches it; if it fails, ship pretrained-as-is + report the honest negative result rather than overclaiming a fine-tune win |
| Landing heuristic (rally-end position) is crude | Scoped and disclosed as v1; arc-fitting is a named v2 refinement; the samples are manually spot-checked before shipping |
| TrackNet on CPU is slow per video | Offline batch (already the product shape); sample heatmaps are precomputed at build time |
| Shuttle tiny/blurred on broadcast → poor detection | Confidence gating + gap-filling in smoothing; qualitative spot-check of the sample heatmaps; honest metrics with the pixel-tolerance stated |
| `tracknet.onnx` too big to commit | Same call as the sample videos: commit if small (~≤25 MB), else re-encode/quantize or LFS |

## 11. Compute & cost

Fine-tuning: free Colab/Kaggle GPU. Tracking data: public TrackNet dataset (DVC,
Google Drive remote). Serving: existing Render free tier. Target total cost: $0.
