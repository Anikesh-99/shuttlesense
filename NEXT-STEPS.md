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

**Model/sample-data caveat for a fresh clone (read before step 1):** `backend/models/*.onnx` are
gitignored and NOT DVC-tracked (no `.dvc` pointer exists for them -- their registry-of-record is
W&B artifacts, which are only in the local offline `wandb/offline-run-*` dirs right now, not yet
synced anywhere network-reachable -- see the DVC remote setup above and README's "Data / model
provenance" section). `backend/samples/*/video.mp4` ARE DVC-tracked but need `dvc push` (above)
before anyone else can `dvc pull` them. **Render builds from a fresh git clone of whatever you
push** -- if you push to GitHub and create the Render blueprint before doing something about
models/samples recoverability, Render's build will succeed (the Dockerfile doesn't require these
files to exist) but the deployed app will 404 on `/api/samples` and fail real uploads (no
`.onnx` models to run inference with). Before deploying for real, either: (a) sync W&B + push DVC
and add real pull steps to the Dockerfile in a follow-up, or (b) as a stopgap, temporarily commit
the `.onnx`/sample-video files directly to the branch you deploy from (small enough: models total
~600KB, but sample videos are ~70-100MB each -- check GitHub's size limits before doing this).
