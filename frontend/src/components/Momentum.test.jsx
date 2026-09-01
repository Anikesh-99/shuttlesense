// Fix round 1 regression test: ribbon-only mode (any rally with
// winner: null -- which is EVERY real sample/upload today, see Task 18
// report) must still label which color is which player. Before the fix,
// the head legend was gated on `displaySeries` (the score-race layer)
// and ribbon-only mode rendered bare green/blue swatches with no text at
// all.
import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import Momentum from "./Momentum.jsx";

const RIBBON_ONLY_REPORT = {
  fps: 10,
  n_frames: 100,
  rallies: [{ start_frame: 0, end_frame: 50, winner: null }],
  strokes: [{ frame: 5, player: 0, stroke: "smash", confidence: 0.9 }],
};

const SCORE_RACE_REPORT = {
  fps: 10,
  n_frames: 100,
  rallies: [{ start_frame: 0, end_frame: 50, winner: 0 }],
  strokes: [],
};

test("ribbon-only mode (no decided winners) still shows the player-identity legend", () => {
  render(<Momentum report={RIBBON_ONLY_REPORT} />);
  // The always-on head legend.
  expect(screen.getByText("Player 0")).toBeInTheDocument();
  expect(screen.getByText("Player 1")).toBeInTheDocument();
  // The note explaining WHY it's ribbon-only.
  expect(screen.getByText(/Score race unavailable/)).toBeInTheDocument();
});

test("ribbon-only mode's inline note carries its own P0/P1 key", () => {
  const { container } = render(<Momentum report={RIBBON_ONLY_REPORT} />);
  const keys = container.querySelectorAll(".ss-momentum__inline-key");
  expect(keys).toHaveLength(2);
  expect(keys[0].textContent).toContain("P0");
  expect(keys[1].textContent).toContain("P1");
});

test("score-race mode also shows the head legend and omits the ribbon-only note", () => {
  render(<Momentum report={SCORE_RACE_REPORT} />);
  expect(screen.getByText("Player 0")).toBeInTheDocument();
  expect(screen.getByText("Player 1")).toBeInTheDocument();
  expect(screen.queryByText(/Score race unavailable/)).not.toBeInTheDocument();
});
