"""ONNX export + model registry CLI.

Loads the trained `StrokeTCN` / `RallyGRU` checkpoints (produced by
`train_stroke.py` / `train_rally.py`), rebuilds each model from its checkpoint's
stored `config`, exports it to ONNX, and writes `backend/models/manifest.json` --
the committed registry of record that Task 15's serving code pins against (never
hardcode thresholds/paths there; read them from this manifest).

Checkpoint shapes (CONTROLLER RULING, Task 13 carry-over 5):
    stroke: {"state_dict", "config", "val_macro_f1", "classes", "confusion"}
    rally:  {"state_dict", "config", "val_frame_f1"}
Both dicts contain only tensors/primitives, so `torch.load(..., weights_only=False)`
is used (mirrors the training scripts' own weights_only note).

ONNX export mode (CONTROLLER RULING, Task 13 carry-over 1): torch 2.13's default
dynamo exporter requires onnxscript, which is not a project dependency here, so
`dynamo=False` (the legacy TorchScript-tracing exporter) is used -- same
convention as `training/tests/test_models.py` (~L70). `torch>=2.6` is the
requirements floor, so the `dynamo=` kwarg is always available.

Dynamic axes:
    stroke_tcn.onnx: x:(B,30,68) -> logits:(B,8); dynamic on batch only (window
        length 30 is fixed by the training window size).
    rally_gru.onnx:  x:(1,T,4) -> logits:(1,T); dynamic on BOTH batch and time
        (CONTROLLER RULING, Task 13 carry-over 2) because Task 15's inference
        contract (see train_rally.py docstring) runs 512-frame chunks as a
        batch of chunks, i.e. real inputs are (N,512,4), not (1,T,4).

Model registry (W&B artifacts, `--wandb`): uploads `stroke-tcn:latest` /
`rally-gru:latest` W&B artifacts. Offline fallback (CONTROLLER RULING): if
`WANDB_API_KEY` is unset, `WANDB_MODE=offline` is forced so artifact upload
still works (queued locally, synced later via `wandb sync`) instead of blocking
on an interactive login prompt.

Usage:
    python training/export_onnx.py \
        --stroke-ckpt training/checkpoints/stroke_tcn/best.pt \
        --rally-ckpt training/checkpoints/rally_gru/best.pt \
        [--out-dir backend/models] [--wandb] [--skip-stroke] [--skip-rally]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

# Repo root on sys.path so `from training.models import ...` resolves regardless
# of the caller's cwd (mirrors train_stroke.py / train_rally.py convention).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.models import RallyGRU, StrokeTCN  # noqa: E402

RALLY_THRESHOLD = 0.6  # operating threshold (RULED: never hardcode 0.5 downstream;
                        # Task 15 reads this from manifest.json's "rally"."threshold").


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def export_stroke(ckpt_path: str, out_dir: Path) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = StrokeTCN(channels=tuple(cfg["channels"]), k=cfg["kernel"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    out_path = out_dir / "stroke_tcn.onnx"
    x = torch.randn(1, 30, 68)
    # legacy exporter (dynamo=False) -- see module docstring.
    torch.onnx.export(
        model,
        x,
        str(out_path),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "B"}, "logits": {0: "B"}},
        opset_version=17,
        dynamo=False,
    )
    return {
        "file": out_path.name,
        "val_macro_f1": ck["val_macro_f1"],
        "git_sha": git_sha(),
        "classes": ck["classes"],
    }


def export_rally(ckpt_path: str, out_dir: Path) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = RallyGRU(hidden=cfg["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    out_path = out_dir / "rally_gru.onnx"
    x = torch.randn(1, 512, 4)
    # dynamic on BOTH batch and time -- carry-over 2 (Task 15 runs (N,512,4) chunks).
    torch.onnx.export(
        model,
        x,
        str(out_path),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "B", 1: "T"}, "logits": {0: "B", 1: "T"}},
        opset_version=17,
        dynamo=False,
    )
    return {
        "file": out_path.name,
        "val_frame_f1": ck["val_frame_f1"],
        "git_sha": git_sha(),
        "threshold": RALLY_THRESHOLD,
    }


def upload_artifact(name: str, path: Path, metadata: dict) -> str:
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
        print("WANDB_API_KEY not set; forcing WANDB_MODE=offline to avoid an "
              "interactive login prompt blocking export.", file=sys.stderr)
    import wandb

    run = wandb.init(project="shuttlesense", job_type="export-onnx", config=metadata)
    artifact = wandb.Artifact(name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=["latest"])
    run.finish()
    mode = os.environ.get("WANDB_MODE", "online")
    print(f"W&B artifact '{name}:latest' logged (mode={mode}, run={run.id})")
    return run.id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stroke-ckpt", default="training/checkpoints/stroke_tcn/best.pt")
    ap.add_argument("--rally-ckpt", default="training/checkpoints/rally_gru/best.pt")
    ap.add_argument("--out-dir", default="backend/models")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--skip-stroke", action="store_true")
    ap.add_argument("--skip-rally", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if not a.skip_stroke:
        manifest["stroke"] = export_stroke(a.stroke_ckpt, out_dir)
        print(f"exported {out_dir / 'stroke_tcn.onnx'}")
        if a.wandb:
            upload_artifact("stroke-tcn", out_dir / "stroke_tcn.onnx", manifest["stroke"])

    if not a.skip_rally:
        manifest["rally"] = export_rally(a.rally_ckpt, out_dir)
        print(f"exported {out_dir / 'rally_gru.onnx'}")
        if a.wandb:
            upload_artifact("rally-gru", out_dir / "rally_gru.onnx", manifest["rally"])

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
