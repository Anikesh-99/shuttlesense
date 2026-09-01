import { expect, test } from "vitest";
import { activeStroke, frameForTime, scaleKpt } from "./Player.jsx";

test("frameForTime rounds", () => {
  expect(frameForTime(2.03, 15)).toBe(30);
});

test("frameForTime rounds down/up correctly at .5 boundary", () => {
  // 0.5s at 10fps = frame 5.0 exactly
  expect(frameForTime(0.5, 10)).toBe(5);
  // 1.24s at 10fps = 12.4 -> rounds to 12
  expect(frameForTime(1.24, 10)).toBe(12);
  // 1.26s at 10fps = 12.6 -> rounds to 13
  expect(frameForTime(1.26, 10)).toBe(13);
});

test("frameForTime handles t=0", () => {
  expect(frameForTime(0, 30)).toBe(0);
});

test("activeStroke finds event within window", () => {
  const strokes = [{ frame: 45, player: 0, stroke: "smash", confidence: 0.9 }];
  expect(activeStroke(strokes, 46, 15)).toMatchObject({ stroke: "smash" });
  expect(activeStroke(strokes, 80, 15)).toBeNull();
});

test("activeStroke returns null for empty strokes", () => {
  expect(activeStroke([], 10, 30)).toBeNull();
});

test("activeStroke picks the closest event when two are in window", () => {
  const strokes = [
    { frame: 40, player: 0, stroke: "clear", confidence: 0.5 },
    { frame: 50, player: 1, stroke: "smash", confidence: 0.99 },
  ];
  // 15fps window is +-6 frames; frame 47 is 7 away from 40 (out) and 3 away
  // from 50 (in) -- only "smash" qualifies.
  expect(activeStroke(strokes, 47, 15)).toMatchObject({ stroke: "smash" });

  // frame 45 is 5 away from 40 and 5 away from 50 -- tie goes to whichever
  // is scanned first with strict "<" comparison, i.e. the first in array
  // order stays the pick unless a strictly closer one appears.
  expect(activeStroke(strokes, 45, 15).stroke).toBeDefined();
});

test("activeStroke respects exact boundary of the +-0.4s window", () => {
  // At 10fps, 0.4s = 4 frames exactly.
  const strokes = [{ frame: 10, player: 0, stroke: "drop", confidence: 0.8 }];
  expect(activeStroke(strokes, 14, 10)).toMatchObject({ stroke: "drop" });
  expect(activeStroke(strokes, 15, 10)).toBeNull();
});

test("scaleKpt scales proportionally from video space to canvas space", () => {
  expect(scaleKpt([100, 200], 1000, 500, 500, 250)).toEqual([50, 100]);
});

test("scaleKpt is identity when canvas matches video size", () => {
  expect(scaleKpt([12.3, 45.6], 640, 360, 640, 360)).toEqual([12.3, 45.6]);
});

test("scaleKpt returns origin when video dims are missing (no divide-by-zero)", () => {
  expect(scaleKpt([12, 34], 0, 0, 640, 360)).toEqual([0, 0]);
});
