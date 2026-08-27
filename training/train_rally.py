"""Rally-segmenter training CLI.

Consumes: `rally_frames.npz` (per-frame `(M,4)` features, `(M,)` binary labels,
and a `match` id array from which per-match sequences are recovered) and
`splits.json` (match-level train/val/test partition). Each match's frames are
chunked into fixed-size windows (`chunk(size=cfg["chunk"])`, default 512) --
never spanning two matches -- and a `training.models.RallyGRU` is trained with
`BCEWithLogitsLoss` to predict a per-frame in-rally probability. Saves the
best-val-frame-F1 checkpoint to `<out-dir>/best.pt` as
`{"state_dict", "config", "val_frame_f1"}`.

Usage:
    python training/train_rally.py --config training/configs/rally_gru.yaml \
        [--data training/data/processed/rally_frames.npz] \
        [--splits training/data/processed/splits.json] \
        [--out-dir training/checkpoints/rally_gru] [--epochs N] [--wandb]

Weights_only note (mirrors Task 10's ruling): the saved checkpoint dict holds
only primitives (a tensor state_dict, a plain dict `config` from
`yaml.safe_load`, and a float `val_frame_f1`) -- no custom classes or
callables -- so it loads under both `torch.load(..., weights_only=True)` and
`weights_only=False`. The smoke test uses `weights_only=False` per the brief.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

# Bootstrap: see train_stroke.py -- this file is invoked as
# `python training/train_rally.py`, which puts `training/` (not the repo
# root) at sys.path[0]. Insert the repo root so `training.*`/`shuttlesense_core`
# absolute imports resolve regardless of the caller's cwd/PYTHONPATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.common import set_seed, resolve_splits
from training.models import RallyGRU


def chunk(X: np.ndarray, y: np.ndarray, size: int = 512):
    """Split one match's (contiguous, non-overlapping) frames into windows of
    `size`. A short final remainder that doesn't fill a whole window is
    dropped UNLESS there isn't even one full window yet (`n == 0`), in which
    case the whole (short) sequence is zero-padded up to exactly one window --
    this guarantees `load_split` never emits a zero-length sequence for a
    match with fewer than `size` frames.

    Never spans two matches: callers must invoke this once per match (see
    `load_split`), not on the whole (multi-match) array.
    """
    n = (len(X) // size) * size
    if n == 0:
        pad = size - len(X)
        X = np.pad(X, ((0, pad), (0, 0)))
        y = np.pad(y, (0, pad))
        n = size
    return (torch.from_numpy(X[:n].reshape(-1, size, X.shape[1])),
            torch.from_numpy(y[:n].reshape(-1, size)))


def load_split(data_path, splits_path, chunk_size: int = 512):
    z = np.load(data_path)
    all_matches = set(np.unique(z["match"]).tolist())
    splits = resolve_splits(all_matches, splits_path)
    n_features = z["X"].shape[1]
    out = {}
    for name in ("train", "val", "test"):
        Xs, ys = [], []
        for mid in splits[name]:
            m = z["match"] == mid
            if not m.any():  # unknown id -- already warned by resolve_splits
                continue
            Xc, yc = chunk(z["X"][m], z["y"][m], size=chunk_size)
            Xs.append(Xc)
            ys.append(yc)
        if Xs:
            out[name] = (torch.cat(Xs), torch.cat(ys))
        else:
            out[name] = (torch.empty((0, chunk_size, n_features), dtype=torch.float32),
                         torch.empty((0, chunk_size), dtype=torch.float32))
    return out


def run_epoch(model, X, y, bs, opt=None, loss_fn=None, device="cpu"):
    """One pass over chunked (X, y). Mirrors train_stroke.py's guard: an empty
    split must not crash -- return 0.0 loss/f1 and empty arrays."""
    if len(X) == 0:
        return 0.0, 0.0, (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    idx = torch.randperm(len(X)) if opt is not None else torch.arange(len(X))
    losses, probs, ys = [], [], []
    for i in range(0, len(X), bs):
        b = idx[i:i + bs]
        xb, yb = X[b].to(device), y[b].to(device)
        logits = model(xb)
        if opt is not None:
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        probs.append(torch.sigmoid(logits).detach().cpu())
        ys.append(yb.cpu())
    p = torch.cat(probs).numpy().ravel()
    t = torch.cat(ys).numpy().ravel()
    f1 = f1_score(t, (p > 0.5).astype(np.float32), zero_division=0)
    return (np.mean(losses) if losses else 0.0, f1, (t, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="training/data/processed/rally_frames.npz")
    ap.add_argument("--splits", default="training/data/processed/splits.json")
    ap.add_argument("--out-dir", default="training/checkpoints/rally_gru")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.epochs is not None:  # 0 is a valid (if useless) explicit override; must not
        cfg["epochs"] = a.epochs  # silently fall back to the config's epochs instead
    if cfg.get("epochs", 0) <= 0:
        raise ValueError(f"cfg['epochs'] must be > 0, got {cfg.get('epochs')!r}")
    set_seed(cfg["seed"])
    data = load_split(a.data, a.splits, chunk_size=cfg["chunk"])
    if len(data["train"][0]) == 0:
        raise ValueError(
            f"train split is empty -- check {a.splits} match names against {a.data}'s "
            "`match` array (no overlap found)"
        )
    if len(data["val"][0]) == 0:
        print("val split empty; model selection disabled; best.pt will hold the "
              "epoch-0 model", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RallyGRU(hidden=cfg["hidden"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    wb = None
    if a.wandb:
        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_MODE"] = "offline"
            print("WANDB_API_KEY not set; forcing WANDB_MODE=offline to avoid an "
                  "interactive login prompt blocking training.", file=sys.stderr)
        import wandb
        wb = wandb.init(project="shuttlesense", job_type="train-rally", config=cfg)
    os.makedirs(a.out_dir, exist_ok=True)
    best = -1.0
    for ep in range(cfg["epochs"]):
        model.train()
        tr_loss, tr_f1, _ = run_epoch(model, *data["train"], cfg["batch_size"], opt, loss_fn,
                                       device=device)
        model.eval()
        with torch.no_grad():
            _, va_f1, _ = run_epoch(model, *data["val"], cfg["batch_size"], device=device)
        if wb:
            wb.log({"epoch": ep, "train_loss": tr_loss, "train_frame_f1": tr_f1,
                     "val_frame_f1": va_f1})
        print(f"ep{ep} loss={tr_loss:.3f} train_f1={tr_f1:.3f} val_f1={va_f1:.3f}")
        if va_f1 > best:
            best = va_f1
            state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state_dict_cpu, "config": cfg,
                        "val_frame_f1": best},
                       f"{a.out_dir}/best.pt")
    if wb:
        wb.summary["best_val_frame_f1"] = best
        wb.finish()


if __name__ == "__main__":
    main()
