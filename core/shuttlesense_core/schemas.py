from __future__ import annotations
from dataclasses import asdict, dataclass, field

STROKE_CLASSES = ["clear", "smash", "drop", "net", "lift", "drive", "serve"]
NONE_CLASS = "none"
ALL_CLASSES = STROKE_CLASSES + [NONE_CLASS]

@dataclass
class RallyInterval:
    start_frame: int
    # end_frame is EXCLUSIVE (Python-slice convention): the rally spans
    # frames [start_frame, end_frame), i.e. end_frame itself is NOT part of
    # the rally. This is the report-layer contract (matches
    # shuttlesense_core.smoothing.probs_to_intervals's own exclusive-end
    # runs); ShuttleSet's raw label data uses an INCLUSIVE end instead --
    # Task 19's sample-report generation is responsible for converting
    # between the two, not this schema.
    end_frame: int
    winner: int | None = None

@dataclass
class StrokeEvent:
    frame: int
    player: int
    stroke: str
    confidence: float

@dataclass
class MatchReport:
    fps: float
    width: int
    height: int
    n_frames: int
    rallies: list[RallyInterval] = field(default_factory=list)
    strokes: list[StrokeEvent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchReport":
        return cls(
            fps=d["fps"], width=d["width"], height=d["height"], n_frames=d["n_frames"],
            rallies=[RallyInterval(**r) for r in d["rallies"]],
            strokes=[StrokeEvent(**s) for s in d["strokes"]],
        )
