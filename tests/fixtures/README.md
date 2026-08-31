# Test fixtures

- `stroke_windows_sample.npz`: the first 10 rows (`X:(10,30,68) float32`,
  `y:(10,) int64`, `match:(10,) str`) of `training/data/processed/stroke_windows.npz`
  as of commit `629ee1ef90d37ef297770e8d2d2e636e51ac709d` (Task 12's real
  ShuttleSet-derived training tensor, produced by `training/build_windows.py`;
  all 10 rows are from the `An_Se_Young_Ratchanok_Intanon_...` match). Real
  pose-derived features, not synthetic -- for Task 15's train/serve
  consistency test to run `shuttlesense_core.features.stroke_window` /
  `backend.app.pipeline.stroke_window` against known-real input and compare
  against the trained `StrokeTCN` / exported ONNX output.
