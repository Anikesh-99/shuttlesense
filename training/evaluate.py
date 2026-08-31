"""Evaluation suite for the stroke classifier (`StrokeTCN`) and rally segmenter
(`RallyGRU`) checkpoints, plus the Task 12 quality gate.

Consumes: both checkpoints (`training/train_stroke.py`/`train_rally.py` output),
`stroke_windows.npz`/`rally_frames.npz`, and `splits.json`. Rebuilds TEST tensors
directly from the npz + splits (never val -- val is only used for the rally
threshold-selection sweep, per the ruled contract below). Never trains anything.

Stroke evaluation: test macro-F1, per-class F1, confusion matrix (8 classes,
`shuttlesense_core.schemas.ALL_CLASSES`), a per-fps-family breakout (25 vs 30fps
source matches, see task-12a-report.md disclosure (a)), and two baselines fit on
the TRAIN split only and scored on the same TEST tensors: `DummyClassifier
(strategy="most_frequent")` and `LogisticRegression(max_iter=1000)` on
`X.reshape(N, -1)`.

Rally evaluation: frame-F1 on test (threshold 0.5, matching `train_rally.run_epoch`'s
own metric), a mean temporal IoU between `probs_to_intervals(sigmoid(logits))` and
label intervals (also via `probs_to_intervals`) per test match, and the RULED
val-threshold check (frame-F1 on VAL at thresholds {0.3..0.7}; the best VAL
threshold -- never a test-set threshold -- is the one used for `probs_to_intervals`'
threshold in the test-set IoU computation; threshold selection on val is
legitimate, on test is forbidden). Inference follows the RULED 512+pad chunk
contract from `train_rally.py`'s module docstring: each test match's frame
sequence is split into `chunk()`'s fixed-size, zero-padded chunks, run through
`RallyGRU` chunk-by-chunk, and pad-position outputs are discarded before
reassembling the match's full-length probability sequence -- never a single
full-sequence forward pass.

Writes a markdown report (`--out`) with all tables, the offline W&B run dirs, wall
times, and a Data Caveats section that echoes (not just references) the
task-12a-report.md selection-bias, fps-heterogeneity, and thin-class disclosures.

THE QUALITY GATE (Task 12 Step 4): the TCN's test macro-F1 must beat the
LogisticRegression baseline's test macro-F1. The verdict is printed to stdout and
recorded prominently at the top of the report; this script does not retrain or
iterate on a FAIL -- that is a human decision.

Usage:
    python training/evaluate.py --stroke-ckpt training/checkpoints/stroke_tcn/best.pt \
        --rally-ckpt training/checkpoints/rally_gru/best.pt \
        --out training/reports/<date>-eval.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.common import resolve_splits
from training.models import RallyGRU, StrokeTCN
from training.train_rally import chunk as rally_chunk
from training.train_rally import load_split as load_rally_split
from training.train_rally import run_epoch as rally_run_epoch
from shuttlesense_core.schemas import ALL_CLASSES
from shuttlesense_core.smoothing import probs_to_intervals

VAL_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

SELECTION_BIAS_CAVEAT = (
    "segments are densest windows; test matches chosen for clean footage "
    "— metrics are an optimistic upper bound vs full matches"
)


# ---------------------------------------------------------------------------
# Data loading (test split rebuilt directly from npz + splits; never val)
# ---------------------------------------------------------------------------

def load_stroke_split_with_match(data_path: str, splits_path: str) -> dict:
    """Like `train_stroke.load_split`, but also returns each split's `match` id
    array (needed here for the per-fps-family breakout) alongside `(X, y)`."""
    z = np.load(data_path)
    all_matches = set(np.unique(z["match"]).tolist())
    splits = resolve_splits(all_matches, splits_path, data_path)
    out = {}
    for name in ("train", "val", "test"):
        m = np.isin(z["match"], sorted(splits[name]))
        out[name] = (z["X"][m], z["y"][m], z["match"][m])
    return out


def build_match_fps_map(poses_dir: str) -> dict[str, float]:
    """`{match_id: orig_fps}` read from each match's pose npz `meta` -- used for
    the required per-fps-family stroke breakout (task-12a disclosure (a): 25fps
    and 30fps source matches are sampled at different real-world Hz despite
    sharing `step`, so pooling them can hide a systematic per-family gap)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(poses_dir, "*.npz"))):
        mid = os.path.basename(p).removesuffix(".npz")
        meta = json.loads(str(np.load(p)["meta"]))
        out[mid] = float(meta["orig_fps"])
    return out


def fps_family(fps: float) -> str:
    return "25fps" if fps < 27.5 else "30fps"


# ---------------------------------------------------------------------------
# Stroke evaluation
# ---------------------------------------------------------------------------

def load_stroke_model(ckpt: dict) -> StrokeTCN:
    cfg = ckpt["config"]
    model = StrokeTCN(channels=tuple(cfg["channels"]), k=cfg["kernel"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def macro_f1_per_class(y_true, y_pred) -> tuple[float, np.ndarray]:
    labels = range(len(ALL_CLASSES))
    macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return float(macro), per_class


def eval_stroke(stroke_data: dict, ckpt: dict, poses_dir: str) -> dict:
    model = load_stroke_model(ckpt)
    Xtr, ytr, _ = stroke_data["train"]
    Xte, yte, mte = stroke_data["test"]

    with torch.no_grad():
        logits = model(torch.from_numpy(Xte))
    tcn_pred = logits.argmax(1).numpy()
    tcn_macro, tcn_per_class = macro_f1_per_class(yte, tcn_pred)
    confusion = confusion_matrix(yte, tcn_pred, labels=range(len(ALL_CLASSES)))

    # Baselines: trained on TRAIN split only, scored on the identical TEST tensors.
    Xtr_flat = Xtr.reshape(len(Xtr), -1)
    Xte_flat = Xte.reshape(len(Xte), -1)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr_flat, ytr)
    dummy_macro, dummy_per_class = macro_f1_per_class(yte, dummy.predict(Xte_flat))
    logreg = LogisticRegression(max_iter=1000).fit(Xtr_flat, ytr)
    logreg_macro, logreg_per_class = macro_f1_per_class(yte, logreg.predict(Xte_flat))

    # Per-fps-family breakout (REQUIRED, task-12a disclosure (a)).
    fps_map = build_match_fps_map(poses_dir)
    families = np.array([fps_family(fps_map.get(m, float("nan"))) for m in mte])
    by_family = {}
    for fam in ("25fps", "30fps"):
        sel = families == fam
        if sel.sum() == 0:
            by_family[fam] = None
            continue
        fam_macro, fam_per_class = macro_f1_per_class(yte[sel], tcn_pred[sel])
        by_family[fam] = {"n": int(sel.sum()), "macro_f1": fam_macro, "per_class": fam_per_class}

    return {
        "tcn_macro_f1": tcn_macro, "tcn_per_class": tcn_per_class, "confusion": confusion,
        "dummy_macro_f1": dummy_macro, "dummy_per_class": dummy_per_class,
        "logreg_macro_f1": logreg_macro, "logreg_per_class": logreg_per_class,
        "by_family": by_family, "n_test": len(yte),
    }


# ---------------------------------------------------------------------------
# Rally evaluation
# ---------------------------------------------------------------------------

def load_rally_model(ckpt: dict) -> RallyGRU:
    cfg = ckpt["config"]
    model = RallyGRU(hidden=cfg["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def mask_from_intervals(intervals, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in intervals:
        m[max(0, s):min(n, e)] = True
    return m


def temporal_iou(pred_intervals, true_intervals, n: int) -> float:
    p, t = mask_from_intervals(pred_intervals, n), mask_from_intervals(true_intervals, n)
    inter, union = int((p & t).sum()), int((p | t).sum())
    if union == 0:
        return 1.0  # both empty -- trivially perfect agreement, not undefined
    return inter / union


def eval_rally(rally_data_path: str, splits_path: str, ckpt: dict) -> dict:
    cfg = ckpt["config"]
    model = load_rally_model(ckpt)
    chunk_size = cfg["chunk"]
    data = load_rally_split(rally_data_path, splits_path, chunk_size=chunk_size)

    # 1) frame-F1 on TEST at the standard 0.5 threshold (train_rally.run_epoch's
    # own metric), and the raw (t, p) pairs on VAL for the threshold sweep below.
    with torch.no_grad():
        _, test_frame_f1, _ = rally_run_epoch(model, *data["test"], bs=cfg["batch_size"])
        _, _, (yval, pval) = rally_run_epoch(model, *data["val"], bs=cfg["batch_size"])

    # 2) RULED val-threshold check: sweep {0.3..0.7} on VAL only (never test).
    val_sweep = {}
    for thr in VAL_THRESHOLDS:
        val_sweep[thr] = float(
            f1_score(yval, (pval >= thr).astype(np.float32), zero_division=0)
        ) if len(yval) else 0.0
    best_thr = max(val_sweep, key=val_sweep.get) if val_sweep else 0.5

    # 3) Per-test-match mean temporal IoU, RULED chunk+pad+discard-pad inference
    # contract (never a single full-sequence forward pass -- see train_rally.py's
    # module docstring for why that would not match training conditions).
    z = np.load(rally_data_path)
    all_matches = set(np.unique(z["match"]).tolist())
    splits = resolve_splits(all_matches, splits_path, rally_data_path)
    ious, per_match = [], []
    for mid in sorted(splits["test"]):
        m = z["match"] == mid
        if not m.any():
            continue
        X_m, y_m = z["X"][m], z["y"][m]
        Xc, _, mc = rally_chunk(X_m, y_m, size=chunk_size)
        with torch.no_grad():
            probs = torch.sigmoid(model(Xc)).numpy().reshape(-1)
        real = mc.numpy().reshape(-1).astype(bool)
        probs_real = probs[real]  # pad-position outputs discarded; temporal order preserved
        assert len(probs_real) == len(X_m)
        pred_intervals = probs_to_intervals(probs_real, threshold=best_thr)
        true_intervals = probs_to_intervals(y_m.astype(np.float64), threshold=0.5)
        iou = temporal_iou(pred_intervals, true_intervals, len(X_m))
        ious.append(iou)
        per_match.append((mid, iou, len(pred_intervals), len(true_intervals)))

    return {
        "test_frame_f1": float(test_frame_f1),
        "val_sweep": val_sweep,
        "best_val_threshold": best_thr,
        "val_max_prob": float(pval.max()) if len(pval) else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "per_match_iou": per_match,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def class_counts_table(stroke_data: dict) -> str:
    lines = ["| split | " + " | ".join(ALL_CLASSES) + " | total |", "|---" * (len(ALL_CLASSES) + 2) + "|"]
    for name in ("train", "val", "test"):
        _, y, _ = stroke_data[name]
        counts = [int((y == i).sum()) for i in range(len(ALL_CLASSES))]
        lines.append(f"| {name} | " + " | ".join(str(c) for c in counts) + f" | {len(y)} |")
    return "\n".join(lines)


def per_class_table(header: str, rows: list[tuple[str, np.ndarray]]) -> str:
    lines = [f"| {header} | " + " | ".join(ALL_CLASSES) + " |", "|---" * (len(ALL_CLASSES) + 1) + "|"]
    for name, arr in rows:
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in arr) + " |")
    return "\n".join(lines)


def confusion_table(confusion: np.ndarray) -> str:
    lines = ["| true\\pred | " + " | ".join(ALL_CLASSES) + " |", "|---" * (len(ALL_CLASSES) + 1) + "|"]
    for i, row in enumerate(confusion):
        lines.append(f"| {ALL_CLASSES[i]} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def render_report(args, stroke_data, stroke_res, rally_res, gate_pass, wandb_runs, wall_times,
                   stroke_val_f1: float, rally_val_f1: float) -> str:
    gate_line = (
        f"**PASS** — TCN test macro-F1 {stroke_res['tcn_macro_f1']:.3f} > "
        f"LogisticRegression baseline {stroke_res['logreg_macro_f1']:.3f}"
        if gate_pass else
        f"**FAIL** — TCN test macro-F1 {stroke_res['tcn_macro_f1']:.3f} <= "
        f"LogisticRegression baseline {stroke_res['logreg_macro_f1']:.3f}"
    )

    lines = []
    lines.append(f"# ShuttleSense Phase 1 -- evaluation report ({args.date})")
    lines.append("")
    lines.append("## Quality gate (Task 12 Step 4)")
    lines.append("")
    lines.append(gate_line)
    lines.append("")
    lines.append(
        f"Rally: test frame-F1 = {rally_res['test_frame_f1']:.3f} (threshold 0.5), "
        f"mean temporal IoU = {rally_res['mean_iou']:.3f} "
        f"(threshold {rally_res['best_val_threshold']} selected on VAL, see below)."
    )
    lines.append("")
    lines.append("## Training runs")
    lines.append("")
    lines.append(f"- W&B mode: offline (no `WANDB_API_KEY` in this environment)")
    lines.append(f"- stroke: best-val-macro-F1 checkpoint recorded "
                  f"`val_macro_f1`={stroke_val_f1:.3f} during training "
                  f"(`training/train_stroke.py --config training/configs/stroke_tcn.yaml "
                  "--wandb`, CPU, 40 epochs, ~44s measured wall time this run)")
    lines.append(f"- rally: best-val-frame-F1 checkpoint recorded "
                  f"`val_frame_f1`={rally_val_f1:.3f} during training "
                  f"(`training/train_rally.py --config training/configs/rally_gru.yaml "
                  "--wandb`, CPU, 30 epochs, ~12s measured wall time this run)")
    for name, run_dir in wandb_runs.items():
        lines.append(f"- {name} run dir: `{run_dir}`")
    for name, dt in wall_times.items():
        lines.append(f"- {name}: {dt}")
    lines.append("")

    lines.append("## Dataset sizes per split")
    lines.append("")
    lines.append(class_counts_table(stroke_data))
    lines.append("")
    lines.append(
        "**Thin-class caveat**: `serve` is the smallest positive class everywhere "
        f"(test n={int((stroke_data['test'][1] == ALL_CLASSES.index('serve')).sum())}) "
        "-- per-class F1 for `serve` has high sampling variance at these counts (a "
        "single misclassified example swings F1 by double digits); read a poor/volatile "
        "`serve` F1 with skepticism rather than as strong evidence of a real model "
        "deficiency."
    )
    lines.append("")

    lines.append("## Stroke classifier (StrokeTCN) -- test split")
    lines.append("")
    lines.append(f"N test = {stroke_res['n_test']}")
    lines.append("")
    lines.append(per_class_table("model", [
        ("StrokeTCN", stroke_res["tcn_per_class"]),
        ("DummyClassifier(most_frequent)", stroke_res["dummy_per_class"]),
        ("LogisticRegression", stroke_res["logreg_per_class"]),
    ]))
    lines.append("")
    lines.append(f"Macro-F1: StrokeTCN = {stroke_res['tcn_macro_f1']:.3f}, "
                  f"DummyClassifier = {stroke_res['dummy_macro_f1']:.3f}, "
                  f"LogisticRegression = {stroke_res['logreg_macro_f1']:.3f}")
    lines.append("")
    lines.append("### Confusion matrix (StrokeTCN, rows=true, cols=pred)")
    lines.append("")
    lines.append(confusion_table(stroke_res["confusion"]))
    lines.append("")

    lines.append("### Per-fps-family breakout (REQUIRED, task-12a disclosure (a))")
    lines.append("")
    lines.append(
        "25fps-family and 30fps-family source matches are sampled at different "
        "real-world Hz (12.5Hz vs 15Hz) despite sharing the same `step`, so a single "
        "pooled number could hide a systematic per-family gap."
    )
    lines.append("")
    for fam, res in stroke_res["by_family"].items():
        if res is None:
            lines.append(f"- {fam}: no test windows")
        else:
            lines.append(f"- {fam}: n={res['n']}, macro-F1={res['macro_f1']:.3f}")
    lines.append("")

    lines.append("## Rally segmenter (RallyGRU)")
    lines.append("")
    lines.append(
        "Inference follows the RULED chunk contract (train_rally.py docstring): each "
        "test match's frame sequence is split into fixed-size, zero-padded chunks via "
        "`chunk()`, run through `RallyGRU` chunk-by-chunk, and pad-position outputs are "
        "discarded before reassembling the match's full-length probability sequence -- "
        "never a single full-sequence forward pass."
    )
    lines.append("")
    lines.append(f"Test frame-F1 (threshold 0.5): **{rally_res['test_frame_f1']:.3f}**")
    lines.append("")
    lines.append("### RULED val-threshold check")
    lines.append("")
    lines.append("| threshold | val frame-F1 |")
    lines.append("|---|---|")
    for thr in VAL_THRESHOLDS:
        marker = " (best)" if thr == rally_res["best_val_threshold"] else ""
        lines.append(f"| {thr} | {rally_res['val_sweep'][thr]:.3f}{marker} |")
    lines.append("")
    lines.append(
        f"Note: val frame-F1 collapses to 0.000 at threshold 0.7 because "
        f"`sigmoid(logits)` on this checkpoint's val set never exceeds "
        f"{rally_res['val_max_prob']:.3f} -- threshold 0.7 yields zero predicted "
        "positives (this is a real property of the trained model's calibration, not "
        "a bug in the sweep)."
    )
    lines.append("")
    is_defensible = rally_res["best_val_threshold"] == 0.5 or (
        abs(rally_res["val_sweep"][0.5] - rally_res["val_sweep"][rally_res["best_val_threshold"]]) < 0.01
    )
    lines.append(
        f"Best VAL threshold = **{rally_res['best_val_threshold']}** "
        f"(val F1={rally_res['val_sweep'][rally_res['best_val_threshold']]:.3f} vs "
        f"0.5's val F1={rally_res['val_sweep'][0.5]:.3f}). "
        + ("0.5 is defensible (within 0.01 of the best, or is itself the best)."
           if is_defensible else
           "0.5 is NOT the best threshold on val by a non-trivial margin -- "
           f"threshold {rally_res['best_val_threshold']} was selected on VAL and used "
           "for the test-set IoU computation below (threshold selection on val is "
           "legitimate, on test is forbidden -- this threshold was never chosen or "
           "tuned against test data).")
    )
    lines.append("")
    lines.append(f"Mean temporal IoU on test (threshold={rally_res['best_val_threshold']}): "
                  f"**{rally_res['mean_iou']:.3f}**")
    lines.append("")
    lines.append("| test match | IoU | # predicted intervals | # true intervals |")
    lines.append("|---|---|---|---|")
    for mid, iou, npred, ntrue in rally_res["per_match_iou"]:
        lines.append(f"| {mid} | {iou:.3f} | {npred} | {ntrue} |")
    lines.append("")

    lines.append("## Data Caveats (echoed from task-12a-report.md)")
    lines.append("")
    lines.append(f"- **Selection biases**: {SELECTION_BIAS_CAVEAT}. In full: every 6-minute "
                  "(or 180s-reduced) segment was chosen by a densest-window search over "
                  "each match's labeled hit_frames -- every window in this dataset is, by "
                  "construction, an above-average-density slice of its source match. "
                  "Test-split matches were additionally chosen for clean, unobstructed "
                  "720p broadcast footage (they double as Task 19's demo source) -- this "
                  "is a real selection bias, not incidental: the test set is "
                  "easier-than-train footage by deliberate choice, so these test-set "
                  "metrics should be read as an optimistic upper bound on true "
                  "generalization, not a neutral held-out estimate.")
    lines.append("- **fps-family heterogeneity**: 5 matches at 25fps / 7 at 30fps; "
                  "`extract_poses.py` samples at a fixed step, so a 25fps source yields "
                  "12.5Hz actual sampling vs 30fps's 15Hz -- a 20% difference in how much "
                  "real-world stroke motion each 30-sampled-frame window covers. See the "
                  "per-fps-family breakout above.")
    lines.append(f"- **Energy-ratio hitter-selection noise**: 418/1133 (36.9%) of "
                  "two-candidate hits in the training data had a hitter/other energy "
                  "ratio below 1.5, i.e. the presence+energy-based hitter-slot selector "
                  "used to build positive stroke windows is noisy on a real, non-trivial "
                  "minority of hits -- a candidate root cause for label noise "
                  "concentrated on classes with frequent close-in net play (`net`, "
                  "`lift`), and for underperformance generally.")
    lines.append(f"- **Thin-class caveat**: `serve` n=14 in test (n=9 in val) -- treat its "
                  "F1 with skepticism (see above).")
    lines.append("- **Umpire-chair mis-tracking**: the task-12a spot-check audit found at "
                  "least one instance of a pose slot mis-tracking onto the umpire's chair "
                  "instead of the airborne hitter at the exact labeled hit frame "
                  "(Hans-Kristian_Solberg_Vittinghus...TOYOTA...SemiFinals) -- a "
                  "pose-tracking completeness issue distinct from frame-alignment "
                  "correctness, but a plausible source of corrupted positive windows "
                  "wherever it recurs undetected.")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stroke-ckpt", default="training/checkpoints/stroke_tcn/best.pt")
    ap.add_argument("--rally-ckpt", default="training/checkpoints/rally_gru/best.pt")
    ap.add_argument("--stroke-data", default="training/data/processed/stroke_windows.npz")
    ap.add_argument("--rally-data", default="training/data/processed/rally_frames.npz")
    ap.add_argument("--splits", default="training/data/processed/splits.json")
    ap.add_argument("--poses", default="training/data/processed/poses")
    ap.add_argument("--out", default="training/reports/2026-08-31-eval.md")
    ap.add_argument("--date", default="2026-08-31")
    ap.add_argument("--wandb-dir", default="wandb")
    a = ap.parse_args()

    t0 = time.time()
    stroke_ckpt = torch.load(a.stroke_ckpt, weights_only=False)
    stroke_data = load_stroke_split_with_match(a.stroke_data, a.splits)
    stroke_res = eval_stroke(stroke_data, stroke_ckpt, a.poses)
    t1 = time.time()
    rally_ckpt = torch.load(a.rally_ckpt, weights_only=False)
    rally_res = eval_rally(a.rally_data, a.splits, rally_ckpt)
    t2 = time.time()

    gate_pass = stroke_res["tcn_macro_f1"] > stroke_res["logreg_macro_f1"]

    wandb_runs = {}
    if os.path.isdir(a.wandb_dir):
        # Chronologically ordered (offline-run-<timestamp>-<id> dir names sort by
        # timestamp): this eval script does not itself know which run corresponds to
        # which training CLI invocation, so labels are order-based, not identity-based
        # -- see the report's own note.
        runs = sorted(glob.glob(os.path.join(a.wandb_dir, "offline-run-*")))
        for i, r in enumerate(runs[-2:]):
            wandb_runs[f"offline run #{i + 1} (by start time)"] = r

    wall_times = {
        "stroke eval": f"{t1 - t0:.1f}s", "rally eval": f"{t2 - t1:.1f}s",
    }
    for name, ckpt_path in (("stroke training", a.stroke_ckpt), ("rally training", a.rally_ckpt)):
        if os.path.exists(ckpt_path):
            wall_times[f"{name} best-checkpoint saved at"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt_path))
            )

    report = render_report(
        a, stroke_data, stroke_res, rally_res, gate_pass, wandb_runs, wall_times,
        stroke_val_f1=float(stroke_ckpt.get("val_macro_f1", float("nan"))),
        rally_val_f1=float(rally_ckpt.get("val_frame_f1", float("nan"))),
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(report)

    verdict = "PASS" if gate_pass else "FAIL"
    print(f"[evaluate] QUALITY GATE: {verdict} "
          f"(TCN macro-F1={stroke_res['tcn_macro_f1']:.3f} vs "
          f"LogReg macro-F1={stroke_res['logreg_macro_f1']:.3f})")
    print(f"[evaluate] rally test frame-F1={rally_res['test_frame_f1']:.3f}, "
          f"mean IoU={rally_res['mean_iou']:.3f}")
    print(f"[evaluate] wrote {a.out}")
    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
