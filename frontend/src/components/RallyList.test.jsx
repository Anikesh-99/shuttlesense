// Final-review fix regression test: the winner cell must never use the
// PLAYER_COLOR (green/blue) skeleton palette, since `rally.winner` is
// ShuttleSet's match-scoped label identity, not the pipeline's per-frame
// court-side skeleton slot -- see RallyList.jsx's doc comment and
// Momentum.jsx's for the full two-axes rationale.
import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import RallyList from "./RallyList.jsx";

// Must match Player.jsx / Momentum.jsx's skeleton palette exactly -- these
// are the colors the winner cell must NEVER render in.
const PLAYER_COLOR = ["#63d5a0", "#6fa8ff"];
const PLAYER_COLOR_RGB = ["rgb(99, 213, 160)", "rgb(111, 168, 255)"];

const REPORT_WITH_WINNER = {
  fps: 10,
  n_frames: 100,
  rallies: [{ start_frame: 0, end_frame: 50, winner: 0 }],
  strokes: [{ frame: 5, player: 0, stroke: "smash", confidence: 0.9 }],
};

const REPORT_NO_WINNER = {
  fps: 10,
  n_frames: 100,
  rallies: [{ start_frame: 0, end_frame: 50, winner: null }],
  strokes: [],
};

test("winner cell never uses the PLAYER_COLOR skeleton palette (no players prop)", () => {
  const { container } = render(<RallyList report={REPORT_WITH_WINNER} />);
  const winnerCell = container.querySelector(".ss-rallylist__winner");
  expect(winnerCell).not.toBeNull();
  // No inline style at all -- styling comes entirely from the CSS class
  // (neutral ink), never a per-row inline color.
  expect(winnerCell.getAttribute("style")).toBeNull();
  expect(winnerCell.style.color).not.toBe(PLAYER_COLOR[0]);
  expect(winnerCell.style.color).not.toBe(PLAYER_COLOR[1]);
  expect(winnerCell.style.color).not.toBe(PLAYER_COLOR_RGB[0]);
  expect(winnerCell.style.color).not.toBe(PLAYER_COLOR_RGB[1]);
});

test("winner cell falls back to a neutral 'Winner: P{n}' when players is absent (match jobs, older samples)", () => {
  render(<RallyList report={REPORT_WITH_WINNER} />);
  expect(screen.getByText("Winner: P0")).toBeInTheDocument();
});

test("winner cell shows the real competitor name when players is available, still no PLAYER_COLOR", () => {
  const { container } = render(
    <RallyList report={REPORT_WITH_WINNER} players={["Chou Tien Chen", "Anders Antonsen"]} />,
  );
  const winnerCell = container.querySelector(".ss-rallylist__winner");
  expect(winnerCell.textContent).toBe("Chou Tien Chen won");
  expect(winnerCell.getAttribute("style")).toBeNull();
});

test("a malformed players prop (wrong length) falls back to 'Winner: P{n}', not a crash", () => {
  render(<RallyList report={REPORT_WITH_WINNER} players={["OnlyOneName"]} />);
  expect(screen.getByText("Winner: P0")).toBeInTheDocument();
});

test("null winner still renders 'No winner recorded'", () => {
  render(<RallyList report={REPORT_NO_WINNER} />);
  expect(screen.getByText("No winner recorded")).toBeInTheDocument();
});
