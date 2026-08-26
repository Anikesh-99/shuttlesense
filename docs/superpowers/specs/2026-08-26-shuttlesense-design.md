# ShuttleSense — Badminton Match Analytics from Video

**Date:** 2026-08-26
**Status:** Approved design, pre-implementation
**Author:** Anikesh (b.anikesh@gmail.com), designed collaboratively with Claude

## 1. Purpose

Portfolio project targeting **ML Engineer** roles. Fills the identified gaps in the
existing portfolio (github.com/Anikesh-99): no self-trained models, no computer
vision, no MLOps lifecycle. Differentiates from saturated pose-app genres (form
checkers, rep counters) by tackling **temporal sports-video understanding**.

**One-liner:** Upload a badminton match video → get a match report: rallies,
per-player stroke breakdown, momentum chart, and shuttle placement heatmaps —
powered by models trained by the author, behind a reproducible
training/eval/registry pipeline.

**Success criteria:**

- A recruiter opens the URL and is watching an annotated match (skeleton overlays,
  live stroke labels, rally-marked seek bar) within 5 seconds, with zero uploads.
- The README can honestly say: models trained by me, experiment-tracked, versioned
  datasets, eval-gated promotion, deployed inference.
- Stroke classifier beats a published, documented baseline on a fixed test suite.

## 2. Scope

**In scope (Phase 1):** rally segmentation, stroke classification, match report UI
(annotated player, momentum timeline, rally explorer), async processing API,
deployed demo with pre-analyzed sample matches.

**In scope (Phase 2):** shuttle tracking (fine-tuned TrackNet-family CNN), landing
detection, court placement heatmap, in-video heat overlay, amateur-footage
domain-gap evaluation.

**Out of scope (explicitly):** doubles matches (singles only), live/real-time
analysis, mobile app, coaching advice generation, user accounts, video longer than
~90s / one game on the free-tier deployment, 3D pose, multi-camera.

## 3. Architecture

Monorepo, four areas:

```
shuttlesense/
├── core/        # shared package: feature extraction, homography, windowing,
│                # rally smoothing — imported IDENTICALLY by training and serving
├── training/    # dataset prep, training CLIs, configs, eval suite (DVC + W&B)
├── backend/     # FastAPI API + async worker (shared SQLite job table)
└── frontend/    # React SPA match report
```

Runtime data flow (worker, per uploaded video):

1. Frame extraction (ffmpeg, reduced FPS with interpolation — documented
   accuracy/latency trade-off)
2. Court-line detection → homography (classical CV, off-the-shelf)
3. Pose estimation for both players (default: RTMPose via `rtmlib`, chosen for
   robustness on small/blurry broadcast players; MediaPipe is the fallback if
   integration friction eats more than a day)
4. **Rally segmenter** (trained model 1) → rally intervals
5. **Stroke classifier** (trained model 2) → per-hit stroke events
6. *(Phase 2)* **Shuttle tracker** (trained model 3) → trajectory → landings
7. Stats aggregation → match report JSON + artifacts (keypoint tracks, heat grids)

Key decisions:

- **Batch, not real-time.** Upload → job → poll → report. Honest about CPU
  processing time; enables free-tier deployment.
- **ONNX Runtime CPU** for all inference in serving; PyTorch only in training.
- **Train/serve consistency by construction:** feature code lives once, in
  `core/`, plus a test asserting byte-identical features across both paths.
- **Off-the-shelf where training adds no story:** pose and court detection are
  deliberately not trained; README states this reasoning.

## 4. UI / Match report (decided via visual mockups)

Single-page report, **video-hero layout**:

- **Hero: annotated video player.** Client-side `<canvas>` overlay above the
  `<video>` element, driven by JSON tracks — skeleton overlays, stroke labels
  with confidence, rally segments drawn on the seek bar. Overlays toggleable.
  No server-side video re-encoding.
- **Heat overlay (Phase 2, chosen hero visual):** toggle paints shuttle-landing
  heat directly onto the court inside the video frame via the homography.
- **Below hero:** momentum timeline (score race + control ribbon, callouts for
  runs); rally explorer (clickable rows: shot count, ending stroke, jump-to-clip);
  top-down filterable court heatmap (Phase 2) as fallback/exploration view for
  the overlay; stroke mix & effectiveness panel (secondary).
- **Landing page IS a pre-analyzed sample match report.** No empty upload screen.
  2–3 sample matches ship with the image. "Analyze your own video" is the bonus
  path. Requirement: recruiters see the full demo with zero effort.

## 5. Data & models

### Datasets (all public)

| Dataset | Role |
|---|---|
| ShuttleSet / ShuttleSet22 | Primary stroke-level labels (broadcast singles) |
| VideoBadminton (2024) | Secondary source; generalization checks |
| TrackNetV2/V3 badminton data | Phase 2 shuttle-tracking fine-tune |
| Amateur YouTube clips (interim) → author's own court footage (later) | Out-of-domain test set; domain-gap story |

### Model 1 — Rally segmenter (Phase 1)

Binary play/no-play over short windows of pose + motion features; small GRU/TCN;
smoothed into intervals. Metric: temporal IoU vs labeled boundaries.

### Model 2 — Stroke classifier (Phase 1, centerpiece)

- **Input:** ~1–2s window of normalized 2D keypoints for the striking player +
  court-relative position (homography-normalized — pose-space features, never raw
  pixels, for cross-camera transfer).
- **Architecture:** baseline is a temporal CNN (TCN), ~100k–1M params
  (free-Colab friendly); a small transformer encoder is a tracked W&B experiment
  against that baseline, not a design fork.
- **Classes:** clear, smash, drop, net shot, lift, drive, serve.
- **Metrics:** macro-F1, confusion matrix; error-analysis notebook is a
  first-class deliverable.

### Model 3 — Shuttle tracker (Phase 2)

Fine-tuned TrackNetV3 (heatmap-regression CNN). Landing detection from trajectory
+ rally-end logic; homography maps landings to court coordinates.

## 6. MLOps ("solid core" depth)

- **Experiment tracking:** W&B free tier, public project page linked from README.
  Every run logs config, metrics, confusion matrices, sample predictions.
- **Dataset versioning:** DVC, Google Drive remote. Raw video outside git;
  processed features/labels are versioned artifacts; `dvc repro` rebuilds the
  feature pipeline deterministically.
- **Training as code:** one CLI per model (`train_stroke.py --config
  configs/stroke_tcn.yaml`), YAML configs, fixed seeds; Colab notebook is a thin
  wrapper calling the CLI.
- **Model registry:** W&B artifacts holding ONNX exports + metadata (dataset
  version, metrics, git SHA). Serving pins a version in config; promotion is a
  one-line PR.
- **Evaluation gate:** fixed held-out suite (broadcast split + amateur set); every
  candidate runs it pre-promotion; markdown report checked into repo.
- **Train/serve consistency test:** pytest fixture clip → assert serving features
  byte-identical to training features.
- Deliberately excluded at this scale (and documented as such): feature store,
  Kubernetes, automated retraining, monitoring/drift dashboards.

## 7. Serving & deployment

- FastAPI + worker process, shared SQLite job table (no Redis/Celery).
- `POST /api/matches` → queued → worker pipeline → JSON results + artifacts.
- React SPA; overlays rendered client-side from JSON tracks.
- **Deployment:** single-container image on Render free tier (pattern proven by
  the author's Lift project, including static-serving path-traversal hardening).
- **Guardrails:** file-type/size caps (~90s or one game), processing timeout,
  graceful "doesn't look like badminton" rejection when court detection fails.

## 8. Testing

- Unit tests on `core/` (features, homography, smoothing) with small DVC-tracked
  fixture clips.
- Train/serve consistency test (see §6).
- API tests: upload validation, job lifecycle, malformed-video rejection.
- Model quality gates live in the eval suite (not pytest): promoted models must
  beat the published baseline's macro-F1 on the fixed test sets.

## 9. Phases & milestones

**Phase 1 (weeks 1–2) — shippable alone:**

1. Data pipeline: ShuttleSet → pose extraction → windowed features (DVC)
2. Rally segmenter + stroke classifier trained, W&B-tracked, eval report written
3. Inference pipeline + API + report UI (player, momentum, rally list — no heat)
4. Deployed to Render; sample-match landing page live

**Phase 2 (week 3+) — additive:**

5. TrackNetV3 fine-tune → shuttle tracking → landing detection
6. Court heatmap + in-video heat overlay
7. Amateur-footage domain-gap evaluation written into README

Gate between phases is deliberate: if Phase 2 slips, Phase 1 is a complete,
deployed, trained-model project.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| ShuttleSet label format friction / access issues | Week-1 spike: load one match end-to-end before building the full pipeline |
| Pose quality poor on broadcast footage (small players, motion blur) | Reduced-FPS + interpolation; RTMPose over MediaPipe if needed; error analysis separates pose failures from classifier failures |
| Domain gap broadcast → amateur video too large | Homography-normalized pose-space features; measured and reported honestly as part of the story |
| TrackNet underdelivers on amateur footage | Confined to Phase 2; Phase 1 demo complete without it |
| Free Colab session limits | Small models (<1M params), checkpointing, resumable training CLIs |
| Render free-tier CPU too slow for uploads | Strict clip caps; sample matches carry the demo regardless |

## 11. Compute & cost

Training: free Colab/Kaggle GPUs. Tracking: W&B free tier. Data remote: Google
Drive free tier. Hosting: Render free tier. Target total cost: $0.
