"""Stroke-classifier training CLI.

Consumes: `stroke_windows.npz` (X, y, match arrays from `training/build_windows.py`)
and `splits.json` (match-level train/val/test partition). Trains `training.models.StrokeTCN`
with a class-frequency-weighted cross-entropy loss, seeded for reproducibility, and saves
the best-val-macro-F1 checkpoint to `<out-dir>/best.pt` as
`{"state_dict", "config", "val_macro_f1", "classes", "confusion"}`.

Usage:
    python training/train_stroke.py --config training/configs/stroke_tcn.yaml \
        [--data training/data/processed/stroke_windows.npz] \
        [--splits training/data/processed/splits.json] \
        [--out-dir training/checkpoints/stroke_tcn] [--epochs N] [--wandb]

Weights_only note (CONTROLLER RULING, Task 10): the saved checkpoint dict contains only
primitives (state_dict of tensors, a plain dict `config` from `yaml.safe_load`, a float
`val_macro_f1`, a `list[str]` `classes`, and a `list[list[int]]` `confusion`) -- no custom
classes or callables -- so it is loadable with both `torch.load(..., weights_only=True)`
and `weights_only=False`. The smoke test uses `weights_only=False` per the brief; this
module's own save path does not depend on either mode to work.
"""
from __future__ import annotations
import argparse, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score

# Bootstrap: this file is normally invoked as `python training/train_stroke.py` (per
# this module's own CLI usage and the smoke test), which puts `training/` -- not the
# repo root -- at sys.path[0]. The self-referential `from training.models import ...`
# below needs the repo root on sys.path (there is no repo-root package install, only
# `shuttlesense_core` is pip-installed editable). Insert it if missing rather than
# relying on the caller's cwd/PYTHONPATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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


def compute_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    """Class-frequency weighting: rarer classes get a larger weight, damped by sqrt
    so the rarest class doesn't dominate the loss. `counts` is clamped to >=1 to avoid
    a divide-by-zero for classes absent from the (tiny, e.g. smoke-test) training split."""
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    return torch.tensor(weights)


def run_epoch(model, X, y, bs, opt=None, loss_fn=None):
    """One pass over (X, y). Guard: an empty split (e.g. a tiny/degenerate val set in
    smoke scenarios) must not crash -- return a 0.0 loss/f1 and empty pred/target arrays
    rather than dividing by zero or calling f1_score on empty input."""
    if len(X) == 0:
        return 0.0, 0.0, (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
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
    assert len(data["train"][0]) > 0, (
        f"train split is empty -- check {a.splits} match names against {a.data}'s "
        "`match` array (no overlap found)"
    )
    model = StrokeTCN(channels=tuple(cfg["channels"]), k=cfg["kernel"])
    weights = compute_class_weights(data["train"][1].numpy(), len(ALL_CLASSES))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    wb = None
    if a.wandb:
        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_MODE"] = "offline"
            print("WANDB_API_KEY not set; forcing WANDB_MODE=offline to avoid an "
                  "interactive login prompt blocking training.", file=sys.stderr)
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
            confusion = (confusion_matrix(t, p, labels=range(len(ALL_CLASSES))).tolist()
                         if len(t) else [[0] * len(ALL_CLASSES) for _ in ALL_CLASSES])
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "val_macro_f1": best, "classes": ALL_CLASSES,
                        "confusion": confusion},
                       f"{a.out_dir}/best.pt")
    if wb:
        wb.summary["best_val_macro_f1"] = best
        wb.finish()


if __name__ == "__main__":
    main()
