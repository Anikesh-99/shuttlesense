from shuttlesense_core.schemas import (
    ALL_CLASSES, STROKE_CLASSES, MatchReport, RallyInterval, StrokeEvent,
)

def test_classes():
    assert STROKE_CLASSES == ["clear", "smash", "drop", "net", "lift", "drive", "serve"]
    assert ALL_CLASSES == STROKE_CLASSES + ["none"]

def test_report_roundtrip():
    r = MatchReport(
        fps=15.0, width=1280, height=720, n_frames=900,
        rallies=[RallyInterval(30, 300, winner=0), RallyInterval(360, 700, winner=None)],
        strokes=[StrokeEvent(frame=45, player=0, stroke="serve", confidence=0.91)],
    )
    d = r.to_dict()
    assert d["rallies"][1]["winner"] is None
    r2 = MatchReport.from_dict(d)
    assert r2 == r
