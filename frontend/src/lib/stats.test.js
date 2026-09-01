import { expect, test } from "vitest";
import { controlRibbon, rallySummaries, scoreRace, strokeMix } from "./stats.js";

// ---------------------------------------------------------------------------
// scoreRace
// ---------------------------------------------------------------------------

test("scoreRace null when winners missing", () => {
  expect(scoreRace([{ start_frame: 0, end_frame: 10, winner: null }])).toBeNull();
});

test("scoreRace null when only SOME winners are missing", () => {
  expect(
    scoreRace([
      { start_frame: 0, end_frame: 10, winner: 0 },
      { start_frame: 20, end_frame: 30, winner: null },
    ]),
  ).toBeNull();
});

test("scoreRace accumulates", () => {
  const out = scoreRace([
    { start_frame: 0, end_frame: 100, winner: 0 },
    { start_frame: 120, end_frame: 300, winner: 1 },
    { start_frame: 320, end_frame: 400, winner: 1 },
  ]);
  expect(out[2]).toMatchObject({ p0: 1, p1: 2 });
});

test("scoreRace returns exact per-rally entries with frame = end_frame", () => {
  const out = scoreRace([
    { start_frame: 0, end_frame: 50, winner: 1 },
    { start_frame: 60, end_frame: 90, winner: 0 },
  ]);
  expect(out).toEqual([
    { frame: 50, p0: 0, p1: 1 },
    { frame: 90, p0: 1, p1: 1 },
  ]);
});

test("scoreRace on empty rallies returns empty array (not null)", () => {
  expect(scoreRace([])).toEqual([]);
});

// ---------------------------------------------------------------------------
// controlRibbon
// ---------------------------------------------------------------------------

test("controlRibbon buckets into fixed windows and picks the higher attacking count", () => {
  // fps=10, win=1s -> 10-frame windows. nFrames=25 -> windows [0,10) [10,20) [20,25).
  const strokes = [
    { frame: 1, player: 0, stroke: "smash" },
    { frame: 2, player: 0, stroke: "drive" },
    { frame: 3, player: 1, stroke: "smash" },
    // window 2 [10,20): player 1 has 2 attacking, player 0 has 0
    { frame: 12, player: 1, stroke: "smash" },
    { frame: 15, player: 1, stroke: "drive" },
    { frame: 16, player: 0, stroke: "clear" }, // not attacking -- must not count
    // window 3 [20,25): no strokes at all -> leader null
  ];
  const out = controlRibbon(strokes, 25, 10, 1);
  expect(out).toEqual([
    { startFrame: 0, endFrame: 10, leader: 0 },
    { startFrame: 10, endFrame: 20, leader: 1 },
    { startFrame: 20, endFrame: 25, leader: null },
  ]);
});

test("controlRibbon leader is null on a tie (including 0-0)", () => {
  const strokes = [
    { frame: 1, player: 0, stroke: "smash" },
    { frame: 2, player: 1, stroke: "drive" },
  ];
  const out = controlRibbon(strokes, 10, 10, 1);
  expect(out).toEqual([{ startFrame: 0, endFrame: 10, leader: null }]);
});

test("controlRibbon default 10s window at given fps", () => {
  // fps=15 -> 150-frame windows; nFrames=200 -> two windows [0,150) [150,200).
  const strokes = [{ frame: 5, player: 0, stroke: "smash" }];
  const out = controlRibbon(strokes, 200, 15);
  expect(out).toEqual([
    { startFrame: 0, endFrame: 150, leader: 0 },
    { startFrame: 150, endFrame: 200, leader: null },
  ]);
});

test("controlRibbon on zero frames returns no windows", () => {
  expect(controlRibbon([], 0, 30)).toEqual([]);
});

// ---------------------------------------------------------------------------
// strokeMix
// ---------------------------------------------------------------------------

test("strokeMix tallies stroke types per player", () => {
  const strokes = [
    { frame: 1, player: 0, stroke: "clear" },
    { frame: 2, player: 0, stroke: "clear" },
    { frame: 3, player: 0, stroke: "smash" },
    { frame: 4, player: 1, stroke: "drop" },
  ];
  expect(strokeMix(strokes)).toEqual({
    0: { clear: 2, smash: 1 },
    1: { drop: 1 },
  });
});

test("strokeMix on empty strokes returns empty buckets for both players", () => {
  expect(strokeMix([])).toEqual({ 0: {}, 1: {} });
});

// ---------------------------------------------------------------------------
// rallySummaries
// ---------------------------------------------------------------------------

test("rallySummaries counts shots inside interval", () => {
  const out = rallySummaries(
    [{ start_frame: 0, end_frame: 100, winner: null }],
    [
      { frame: 10, player: 0, stroke: "serve" },
      { frame: 60, player: 1, stroke: "smash" },
    ],
  );
  expect(out[0]).toMatchObject({ shots: 2, endedBy: "smash" });
});

test("rallySummaries excludes strokes at or past end_frame (end-exclusive)", () => {
  const out = rallySummaries(
    [{ start_frame: 0, end_frame: 20, winner: 0 }],
    [
      { frame: 19, player: 0, stroke: "clear" },
      { frame: 20, player: 1, stroke: "smash" }, // exactly at end_frame -- excluded
    ],
  );
  expect(out[0]).toEqual({ index: 0, startFrame: 0, endFrame: 20, shots: 1, endedBy: "clear" });
});

test("rallySummaries endedBy is null with no strokes in interval", () => {
  const out = rallySummaries([{ start_frame: 0, end_frame: 10, winner: 1 }], []);
  expect(out[0]).toEqual({ index: 0, startFrame: 0, endFrame: 10, shots: 0, endedBy: null });
});

test("rallySummaries picks the LAST stroke by frame order as endedBy, not array order", () => {
  const out = rallySummaries(
    [{ start_frame: 0, end_frame: 100, winner: 0 }],
    [
      { frame: 80, player: 0, stroke: "smash" }, // appears first in array...
      { frame: 20, player: 1, stroke: "clear" }, // ...but this is earlier in time
    ],
  );
  expect(out[0]).toMatchObject({ shots: 2, endedBy: "smash" });
});

test("rallySummaries indexes rows in input order across multiple rallies", () => {
  const out = rallySummaries(
    [
      { start_frame: 0, end_frame: 10, winner: 0 },
      { start_frame: 20, end_frame: 30, winner: 1 },
    ],
    [],
  );
  expect(out.map((r) => r.index)).toEqual([0, 1]);
});
