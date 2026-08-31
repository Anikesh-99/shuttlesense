# Running the ShuttleSense training pipeline on Colab

Written as the Task 12 Step 1 leftover (`training/notes/colab.md` referenced by
`training/build_windows.py`'s Step 1 brief and `task-12a-report.md`). Nothing here
was actually exercised against a real Colab session for Task 12b (both real training
runs -- Task 12b Part B -- completed locally on CPU in well under a minute each, so
Colab wasn't needed this round); this documents the intended procedure for a future
run that needs GPU (rally/stroke training itself is cheap even on CPU at this
dataset's current size, N=2388/M=45458 -- Colab is more likely to matter for a much
larger `extract_poses.py` pose-extraction run than for training).

## Why Colab at all

`training/extract_poses.py` (rtmlib, CPU-only in this repo's current setup) runs
roughly 2-4x slower than realtime per clip on CPU -- Task 12a's bulk extraction of 12
matches took ~3h on this machine's CPU. A Colab GPU runtime can speed up pose
extraction substantially (rtmlib supports ONNXRuntime GPU execution providers), and
is also a convenient place to run `train_stroke.py`/`train_rally.py` if a future,
much larger dataset makes CPU training slow. Nothing about the training CLIs
themselves requires a GPU -- `device = "cuda" if torch.cuda.is_available() else
"cpu"` in both `train_stroke.py` and `train_rally.py` already picks up a Colab GPU
runtime automatically with no code changes.

## Version requirements (load-bearing, do not skip)

- **Python 3.11+** -- this repo's `pyproject.toml`/venv targets 3.12; Colab's default
  runtime Python may be older. Check with `!python --version` first; if it's below
  3.11, either pick a Colab runtime image that ships a newer Python or install one
  via `!apt-get install python3.11` + a venv, since `torch>=2.6` and several pinned
  deps in `training/requirements.txt` do not reliably support older Pythons.
- **`torch>=2.6`** -- pinned in `training/requirements.txt` per the repo's own
  controller ruling (see `training/models.py`'s history / task reports): older
  `torch` versions on Colab's default image are common and MUST be upgraded, not
  assumed compatible. `pip install -r training/requirements.txt` handles this as
  long as pip is allowed to upgrade `torch` (Colab sometimes ships a system `torch`
  that a plain `pip install` won't touch without `--upgrade`).

## Procedure

1. **No venv needed** (Colab notebooks already run in an isolated-enough runtime;
   creating a venv on top is unnecessary friction). Clone the repo and install
   directly:
   ```
   !git clone <repo-url> shuttlesense
   %cd shuttlesense
   !pip install -r training/requirements.txt
   !pip install -e core/   # shuttlesense_core, editable install (matches local dev)
   ```
2. **Mount Drive** for persistent storage of raw video / pose npz / checkpoints
   across sessions (Colab's local disk is ephemeral):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Point `--videos-dir`, `--out-dir` (poses), and `--out-dir` (checkpoints) at paths
   under `/content/drive/MyDrive/...` rather than the repo's default
   `training/data/...` paths, e.g.:
   ```
   !python training/extract_poses.py <video.mp4> \
       --out-dir /content/drive/MyDrive/shuttlesense/poses --fps 15
   ```
   For the bulk driver (`scripts/bulk_extract.py`), pass the Drive path via
   `--out-dir` and `--videos-dir` the same way; the default `--workers 2` /
   `--timeout-s` guidance from `task-12a-report.md`'s thermal-throttling note is
   CPU-specific and does not necessarily apply to a GPU Colab runtime (re-tune
   `--workers` based on actual observed throughput rather than reusing the
   CPU-tuned defaults blindly).
3. **Run the same CLIs used locally, unmodified** -- this is the whole point of
   `training/requirements.txt` pinning versions identically for both environments:
   ```
   !python training/build_windows.py \
       --labels /content/drive/MyDrive/shuttlesense/labels.parquet \
       --poses /content/drive/MyDrive/shuttlesense/poses \
       --videos-dir /content/drive/MyDrive/shuttlesense/videos \
       --out-dir /content/drive/MyDrive/shuttlesense/processed

   !python training/train_stroke.py --config training/configs/stroke_tcn.yaml \
       --data /content/drive/MyDrive/shuttlesense/processed/stroke_windows.npz \
       --splits /content/drive/MyDrive/shuttlesense/processed/splits.json \
       --out-dir /content/drive/MyDrive/shuttlesense/checkpoints/stroke_tcn --wandb

   !python training/train_rally.py --config training/configs/rally_gru.yaml \
       --data /content/drive/MyDrive/shuttlesense/processed/rally_frames.npz \
       --splits /content/drive/MyDrive/shuttlesense/processed/splits.json \
       --out-dir /content/drive/MyDrive/shuttlesense/checkpoints/rally_gru --wandb
   ```
   `--wandb` behaves identically to local: if `WANDB_API_KEY` isn't set in the Colab
   environment (e.g. via `%env WANDB_API_KEY=...` or a Colab secret), both CLIs
   force `WANDB_MODE=offline` automatically (see their own `main()` -- this is not
   Colab-specific behavior, it's the same fallback used locally) rather than
   blocking on an interactive login prompt. To sync an offline run afterward:
   `!wandb sync <run_dir>` once online with a real API key.
4. **Copy checkpoints back out of Drive** (or evaluate directly against the Drive
   paths) with `training/evaluate.py --stroke-ckpt <drive-path>/best.pt --rally-ckpt
   <drive-path>/best.pt ...` -- it takes the same `--data`/`--splits`/`--poses`
   overrides as the training CLIs, so it works unmodified against Drive-mounted
   paths too.

## What this note does NOT cover

- Colab-specific GPU execution-provider setup for rtmlib/onnxruntime (out of scope
  for Task 12b; `extract_poses.py`'s own docstring documents its CPU-only default
  behavior and would need a real GPU-enabled onnxruntime install --
  `onnxruntime-gpu` instead of the CPU `onnxruntime` pinned in
  `training/requirements.txt` -- to actually exercise a Colab GPU for extraction;
  untested here).
- DVC push/pull to a remote from Colab (this repo's `dvc.yaml`/`.dvc/config` are
  unchanged by this note; DVC remote credentials on Colab are a separate setup step
  not attempted as part of Task 12b).
