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

// Fix round 1 regression tests: the score-race series identity (real
// competitor names, from ShuttleSet ground truth) must never be presented
// as "the same identity as the green/blue skeleton" -- see Momentum.jsx's
// doc comment. Without a `players` prop, everything stays "Player 0"/
// "Player 1" (behavior-identical to before this fix, e.g. every match-job
// upload).

test("score-race mode without a players prop still falls back to Player 0/Player 1", () => {
  const { container } = render(<Momentum report={SCORE_RACE_REPORT} />);
  expect(container.textContent).toContain("Player 0");
  expect(container.textContent).toContain("Player 1");
  expect(container.textContent).not.toMatch(/Chou|Antonsen/);
});

test("score-race mode with a players prop labels the chart with real names, head legend stays Player 0/1", () => {
  const { container } = render(
    <Momentum report={SCORE_RACE_REPORT} players={["Chou Tien Chen", "Anders Antonsen"]} />,
  );
  // Head legend is ALWAYS about the skeleton slot -- unaffected by `players`.
  expect(screen.getByText("Player 0")).toBeInTheDocument();
  expect(screen.getByText("Player 1")).toBeInTheDocument();
  // Score-race end-labels use the real names, not "P0"/"P1".
  expect(container.textContent).toContain("Chou Tien Chen");
  expect(container.textContent).toContain("Anders Antonsen");
  // The decoupling note is shown so the two identity axes read as distinct.
  expect(screen.getByText(/Score-race colors are distinct from the court-side skeleton colors/))
    .toBeInTheDocument();
});

test("a malformed players prop (wrong length) falls back to Player 0/Player 1, not a crash", () => {
  const { container } = render(<Momentum report={SCORE_RACE_REPORT} players={["OnlyOneName"]} />);
  expect(container.textContent).toContain("Player 0");
  expect(container.textContent).toContain("Player 1");
  expect(screen.queryByText(/Score-race colors are distinct/)).not.toBeInTheDocument();
});

// Fix round 2 regression tests: a real competitor NAME on a
// --player-0/--player-1-colored line made the wrong "this line is that
// skeleton" implication MORE confident, not less -- the score-race series
// must render in a palette that's neither --player-0 nor --player-1, no
// matter whether `players` names are shown or it's still "Player 0"/
// "Player 1" text.

test("score-race polylines never use the skeleton palette classes, in either identity mode", () => {
  const withoutNames = render(<Momentum report={SCORE_RACE_REPORT} />);
  const withNames = render(
    <Momentum report={SCORE_RACE_REPORT} players={["Chou Tien Chen", "Anders Antonsen"]} />,
  );
  for (const { container } of [withoutNames, withNames]) {
    // The old --player-0/--player-1-backed classes must never appear on
    // the score-race polylines.
    expect(container.querySelectorAll(".ss-momentum__line--p0, .ss-momentum__line--p1")).toHaveLength(0);
    // The de-paletted neutral-ink classes are what's actually there instead.
    const race0 = container.querySelectorAll("polyline.ss-momentum__race0");
    const race1 = container.querySelectorAll("polyline.ss-momentum__race1");
    expect(race0).toHaveLength(1);
    expect(race1).toHaveLength(1);
  }
});

test("the head legend's swatches still use the player palette (inline style), unaffected by the score-race de-paletting", () => {
  const { container } = render(
    <Momentum report={SCORE_RACE_REPORT} players={["Chou Tien Chen", "Anders Antonsen"]} />,
  );
  const legendDots = container.querySelectorAll(".ss-momentum__legend-item i");
  expect(legendDots).toHaveLength(2);
  expect(legendDots[0].style.background).toBe("rgb(99, 213, 160)"); // --player-0 #63d5a0
  expect(legendDots[1].style.background).toBe("rgb(111, 168, 255)"); // --player-1 #6fa8ff
});
