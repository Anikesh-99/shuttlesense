// Pure, side-effect-free report analytics. None of these touch the DOM or
// React state -- every function takes the exact data it needs as an
// argument (rallies/strokes/nFrames/fps), so they're trivially unit-tested
// and never "look ahead" implicitly (a caller decides what data is in
// scope, e.g. a whole finished report vs. some as-of-now subset).
//
// Frame-interval convention (binding, matches backend/app): a rally's
// [start_frame, end_frame) is END-EXCLUSIVE, i.e. end_frame is the first
// frame NOT in the rally. controlRibbon's windows follow the same
// convention for consistency.

/** Attacking strokes for controlRibbon's "who's on the front foot" signal. */
const ATTACKING_STROKES = new Set(["smash", "drive"]);

/**
 * Cumulative points-won race, one entry per rally in input order.
 *
 * `rallies`: [{start_frame, end_frame, winner}] -- winner is 0, 1, or null
 * (undecided/ambiguous rally). Returns `null` (the whole thing, not a
 * per-entry null) if ANY rally has a null winner, since a partial score
 * race would be misleading rather than merely incomplete -- the caller
 * (Momentum) falls back to ribbon-only rendering in that case.
 *
 * Each output entry is the score EXACTLY as of the end of that rally,
 * with `frame` set to that rally's `end_frame` (the frame the score
 * changed at). There is no synthetic frame-0/"start" entry here -- that's
 * a display concern the chart component may add, not part of this pure
 * data contract.
 */
export function scoreRace(rallies) {
  if (rallies.some((r) => r.winner == null)) return null;
  let p0 = 0;
  let p1 = 0;
  return rallies.map((r) => {
    if (r.winner === 0) p0 += 1;
    else if (r.winner === 1) p1 += 1;
    return { frame: r.end_frame, p0, p1 };
  });
}

/**
 * Bucket `strokes` into fixed-size, non-overlapping windows of `win`
 * seconds (default 10s) across [0, nFrames), and for each window report
 * which player had the higher count of ATTACKING_STROKES (smash, drive)
 * in that window -- `leader: null` when there are no attacking strokes in
 * the window at all, OR when both players are tied (including 0-0).
 *
 * Windows are END-EXCLUSIVE and cover the whole timeline; the final
 * window may be shorter than `win` seconds if nFrames isn't an exact
 * multiple of the window size.
 */
export function controlRibbon(strokes, nFrames, fps, win = 10) {
  const winFrames = Math.max(1, Math.round(win * fps));
  const windows = [];
  for (let start = 0; start < nFrames; start += winFrames) {
    const end = Math.min(start + winFrames, nFrames);
    const counts = [0, 0];
    for (const s of strokes) {
      if (s.frame >= start && s.frame < end && ATTACKING_STROKES.has(s.stroke)) {
        if (s.player === 0 || s.player === 1) counts[s.player] += 1;
      }
    }
    const leader = counts[0] === counts[1] ? null : counts[0] > counts[1] ? 0 : 1;
    windows.push({ startFrame: start, endFrame: end, leader });
  }
  return windows;
}

/**
 * Per-player stroke-type tallies across the WHOLE match (no windowing).
 * `strokes`: [{frame, player, stroke, ...}]. Returns
 * `{0: {clear: n, smash: n, ...}, 1: {...}}` -- only stroke types that
 * actually occur for that player appear as keys.
 */
export function strokeMix(strokes) {
  const mix = { 0: {}, 1: {} };
  for (const s of strokes) {
    const bucket = mix[s.player];
    if (!bucket) continue; // defensive: ignore any player index other than 0/1
    bucket[s.stroke] = (bucket[s.stroke] || 0) + 1;
  }
  return mix;
}

/**
 * One summary row per rally, for RallyList. `shots` is the count of
 * strokes whose `frame` falls inside the rally's END-EXCLUSIVE
 * [start_frame, end_frame) interval; `endedBy` is the stroke TYPE of the
 * last such stroke by frame order (the shot that closed the rally), or
 * `null` if no strokes fall inside the interval.
 */
export function rallySummaries(rallies, strokes) {
  return rallies.map((r, index) => {
    const inRally = strokes
      .filter((s) => s.frame >= r.start_frame && s.frame < r.end_frame)
      .sort((a, b) => a.frame - b.frame);
    const endedBy = inRally.length ? inRally[inRally.length - 1].stroke : null;
    return {
      index,
      startFrame: r.start_frame,
      endFrame: r.end_frame,
      shots: inRally.length,
      endedBy,
    };
  });
}
