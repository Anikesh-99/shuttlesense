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
