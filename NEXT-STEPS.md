# Next steps (manual, requires interactive browser auth)

Task 5 (DVC setup + ShuttleSet acquisition) intentionally stopped short of configuring a remote
and pushing, because that requires a Google OAuth flow only a human can complete in a browser.
DVC has been initialized (`dvc init`) and `training/data/raw/shuttleset` is tracked locally via
`dvc add` (see `training/data/raw/shuttleset.dvc`, committed to git). The actual data lives only
in your local working tree's DVC cache (`.dvc/cache`, gitignored) until you push it somewhere.

To finish remote setup:

1. Create a Google Drive folder named `shuttlesense-dvc` in your Drive, and copy its folder ID
   from the URL (the long string after `/folders/` in the browser address bar).
2. Configure the remote and commit only the DVC config (avoid `-am`, which would stage and
   commit anything else you happen to have modified in the working tree at the time):
   ```bash
   dvc remote add -d gdrive gdrive://<FOLDER_ID_FROM_URL>
   git add .dvc/config
   git commit -m "chore: DVC gdrive remote"
   ```
3. Push the tracked data (this triggers a one-time browser OAuth flow — follow the prompt):
   ```bash
   dvc push
   ```

After this, anyone who clones the repo can run `dvc pull` to fetch `training/data/raw/shuttleset`
without re-downloading it from GitHub.

## Defense-in-depth: nested `.gitignore`

In addition to the root `.gitignore`'s `training/data/**` + negation rules, there is a second,
belt-and-suspenders ignore file at `training/data/raw/.gitignore` containing just `/shuttleset` —
this explicitly excludes the `shuttleset` data directory (but not `shuttleset.dvc`, a different
filename) even if the root pattern's negation trick ever gets simplified or removed by mistake in
a future edit. Keep both in place; they're redundant on purpose.

## Task 20: publish to GitHub + deploy on Render (manual, requires your credentials)

Task 20 built everything needed for a single-container deploy (`Dockerfile`, `.dockerignore`,
`render.yaml`, `README.md`) and verified it locally with a real `docker build` + `docker run`
cycle (see `.superpowers/sdd/2026-08-26-shuttlesense-phase1/task-20-report.md` for the full
verification log, including two real bugs it caught and fixed). What's left needs your own GitHub
account / Render account and can't be automated here:

1. **Push this repo to a public GitHub repo** (Render's free tier needs to pull from a public repo,
   or you'll need to grant Render access to a private one):
   ```bash
   gh repo create shuttlesense --public --source=. --remote=origin
   git push -u origin worktree-phase-1   # or merge to main first, your call
   ```
2. **Create a Render Blueprint deploy** pointing at that GitHub repo -- Render will read
   `render.yaml` from the repo root automatically (docker runtime, free plan, health check
   `/api/healthz`). Via the Render dashboard: New -> Blueprint -> select the repo -> Apply.
3. **Verify the live URL end-to-end** once deployed: `/api/healthz`, the landing page's
   zero-action sample redirect, and at least one sample report loading with its video. Add the
   live URL to `README.md`'s "Live demo" line at the top once confirmed.

**Model/sample-data status (RESOLVED for deploy):** as of the GitHub push, the demo is
self-contained in git and needs no `dvc pull` at build time:
- `backend/models/*.onnx` (~600 KB) are committed directly to git.
- `backend/samples/*/video.mp4` are committed directly to git, re-encoded to 480p (~13 MB total,
  fps and frame count preserved so the skeleton overlay stays aligned); the per-sample DVC
  pointers were removed in favor of plain git tracking.
So a fresh clone / Render build has everything it needs to serve `/api/samples` with video and to
run inference. The W&B `wandb sync` and `dvc push` steps below remain OPTIONAL polish (public
training curves; cloud backup of the training data) -- they are no longer deploy blockers.
