import { useEffect, useMemo, useRef, useState } from "react";
import { controlRibbon, scoreRace } from "../lib/stats.js";
import "./Momentum.css";

// ---------------------------------------------------------------------------
// Layout constants (viewBox coordinate space -- the <svg> scales to its
// container width via CSS, this is just the internal aspect ratio).
// ---------------------------------------------------------------------------
const VB_W = 960;
const CHART_H = 170; // score-race plot area
const CHART_PAD_TOP = 14;
const CHART_PAD_BOTTOM = 10;
const RIBBON_GAP = 14; // vertical gap between the chart and the ribbon band
const RIBBON_H = 22;
const VB_H = CHART_H + RIBBON_GAP + RIBBON_H;
const PAD_X = 4;

const PLAYER_COLOR = ["#63d5a0", "#6fa8ff"]; // must match Player.jsx / index.css --player-0/1

/**
 * Momentum: the report-level "what happened" chart. Two stacked layers on
 * ONE shared x-axis (frame -> pixel):
 *   1. Score-race polylines (p0 vs p1, cumulative points) on a SHARED
 *      y-axis -- both players' scores are the same unit, so a dual-axis
 *      chart would be actively misleading here. Rendered only when
 *      `scoreRace` returns non-null (every rally has a decided winner);
 *      otherwise this layer is omitted entirely (never a fake/zero line).
 *   2. The control ribbon: a separate horizontal band below the chart,
 *      NOT a second y-scale -- it's a categorical "who's attacking more"
 *      strip, one color swatch per time window.
 *
 * The player-identity legend (colored dot + "Player 0"/"Player 1") in the
 * head is ALWAYS shown, not gated on the score-race layer -- ribbon-only
 * mode still needs the reader to know which color is which player (Fix
 * round 1), and the ribbon-only note additionally repeats a compact inline
 * key next to itself.
 *
 * Interaction: hovering the chart shows a crosshair + tooltip (frame,
 * score) snapped to the nearest data point; clicking anywhere on the
 * chart or ribbon seeks the video to that x position via `onSeek(frame)`.
 *
 * Read-side of the Task 18 ref contract (see Player.jsx's doc comment):
 * `onTimeRef` is the same mutated ref Player writes `{time, frame}` into
 * every animation frame WITHOUT re-rendering Player. Momentum runs its
 * OWN light rAF loop reading that ref and only calls setState when the
 * rendered cursor frame actually changes (comparing against the last
 * frame that would move the cursor by at least ~1px), so most rAF ticks
 * are a no-op read with zero re-renders -- this avoids lifting
 * `currentFrame` into Report's React state (which would re-render the
 * whole report tree ~60x/sec) while still keeping the cursor live.
 */
export default function Momentum({ report, onTimeRef, onSeek }) {
  const svgRef = useRef(null);
  const [cursorFrame, setCursorFrame] = useState(0);
  const [hover, setHover] = useState(null); // {frame, x} | null

  const fps = report?.fps || 30;
  const nFrames = report?.n_frames || 0;
  const rallies = report?.rallies || [];
  const strokes = report?.strokes || [];

  const race = useMemo(() => scoreRace(rallies), [rallies]);
  const ribbon = useMemo(() => controlRibbon(strokes, nFrames, fps), [strokes, nFrames, fps]);

  const chartW = VB_W - PAD_X * 2;

  const xForFrame = (frame) => (nFrames ? PAD_X + (frame / nFrames) * chartW : PAD_X);
  const frameForX = (x) => {
    if (!nFrames) return 0;
    const ratio = Math.min(Math.max((x - PAD_X) / chartW, 0), 1);
    return Math.round(ratio * nFrames);
  };

  // Display series: prepend a synthetic frame-0 origin point and extend the
  // last known score flat to the end of the video, so the line always
  // spans the full chart width -- purely a rendering concern, NOT part of
  // stats.js's pure contract (scoreRace returns exactly one entry per
  // rally, no synthetic points).
  const displaySeries = useMemo(() => {
    if (!race) return null;
    const points = [{ frame: 0, p0: 0, p1: 0 }, ...race];
    const last = race[race.length - 1];
    if (last && last.frame < nFrames) {
      points.push({ frame: nFrames, p0: last.p0, p1: last.p1 });
    }
    return points;
  }, [race, nFrames]);

  const maxScore = useMemo(() => {
    if (!displaySeries) return 1;
    return Math.max(1, ...displaySeries.map((p) => Math.max(p.p0, p.p1)));
  }, [displaySeries]);

  const plotTop = CHART_PAD_TOP;
  const plotBottom = CHART_H - CHART_PAD_BOTTOM;
  const yForScore = (score) => plotBottom - (score / maxScore) * (plotBottom - plotTop);

  const pathFor = (key) =>
    displaySeries
      ? displaySeries.map((p) => `${xForFrame(p.frame)},${yForScore(p[key])}`).join(" ")
      : "";

  // Nearest race point <= hovered frame, for the tooltip's score readout.
  const nearestScoreAt = (frame) => {
    if (!displaySeries) return null;
    let best = displaySeries[0];
    for (const p of displaySeries) {
      if (p.frame <= frame) best = p;
      else break;
    }
    return best;
  };

  // ---- live playhead cursor: own rAF loop reading onTimeRef -----------
  useEffect(() => {
    if (!onTimeRef) return undefined;
    let raf;
    let lastFrame = -1;
    const tick = () => {
      const f = onTimeRef.current?.frame ?? 0;
      // Only trigger a re-render when the cursor would actually move by a
      // visible amount (finer than 1 chart pixel is wasted work).
      if (nFrames && Math.abs(f - lastFrame) * (chartW / nFrames) >= 1) {
        lastFrame = f;
        setCursorFrame(f);
      } else if (lastFrame === -1) {
        lastFrame = f;
        setCursorFrame(f);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [onTimeRef, nFrames, chartW]);

  const clientXToFrame = (clientX) => {
    const svg = svgRef.current;
    if (!svg) return 0;
    const rect = svg.getBoundingClientRect();
    const ratio = rect.width ? (clientX - rect.left) / rect.width : 0;
    return frameForX(ratio * VB_W);
  };

  const handleClick = (e) => {
    if (!onSeek || !nFrames) return;
    onSeek(clientXToFrame(e.clientX));
  };

  const handleMouseMove = (e) => {
    const frame = clientXToFrame(e.clientX);
    setHover({ frame, x: xForFrame(frame) });
  };

  const handleMouseLeave = () => setHover(null);

  const hoverScore = hover ? nearestScoreAt(hover.frame) : null;
  const cursorX = xForFrame(cursorFrame);

  return (
    <div className="ss-momentum">
      <div className="ss-momentum__head">
        <h2>Momentum</h2>
        <div className="ss-momentum__legend">
          <span className="ss-momentum__legend-item">
            <i style={{ background: PLAYER_COLOR[0] }} /> Player 0
          </span>
          <span className="ss-momentum__legend-item">
            <i style={{ background: PLAYER_COLOR[1] }} /> Player 1
          </span>
        </div>
      </div>

      {/* Fix round 1: ribbon-only mode (every real sample/upload today,
          since winner assignment isn't implemented anywhere in the
          pipeline yet -- see Task 18 report) must NOT ship with an
          unlabeled green/blue swatch strip. The identity key above
          already covers this (it's no longer gated on `displaySeries`),
          but repeat it inline, next to the "attacking control" label
          itself, so it reads correctly even if the chart is scrolled or
          the head legend is out of view. Ink-token text, color only on
          the swatch -- never the text itself. */}
      {!displaySeries && (
        <p className="ss-momentum__note">
          Score race unavailable &mdash; one or more rallies has no recorded winner. Showing
          attacking control only:{" "}
          <span className="ss-momentum__inline-key">
            <i style={{ background: PLAYER_COLOR[0] }} /> P0
          </span>
          <span className="ss-momentum__inline-key">
            <i style={{ background: PLAYER_COLOR[1] }} /> P1
          </span>
        </p>
      )}

      <svg
        ref={svgRef}
        className="ss-momentum__svg"
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        role="img"
        aria-label="Momentum: cumulative score race and attacking-control ribbon"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {displaySeries && (
          <g className="ss-momentum__chart">
            {/* Recessive gridlines -- muted ink, never a player color. */}
            {[0, 0.5, 1].map((t) => (
              <line
                key={t}
                x1={PAD_X}
                x2={PAD_X + chartW}
                y1={plotTop + t * (plotBottom - plotTop)}
                y2={plotTop + t * (plotBottom - plotTop)}
                className="ss-momentum__grid"
              />
            ))}

            <polyline points={pathFor("p0")} className="ss-momentum__line ss-momentum__line--p0" />
            <polyline points={pathFor("p1")} className="ss-momentum__line ss-momentum__line--p1" />

            {/* Direct end-labels: identity never by color alone. */}
            <text
              x={xForFrame(displaySeries[displaySeries.length - 1].frame) - 4}
              y={yForScore(displaySeries[displaySeries.length - 1].p0) - 6}
              className="ss-momentum__endlabel ss-momentum__endlabel--p0"
              textAnchor="end"
            >
              P0 &middot; {displaySeries[displaySeries.length - 1].p0}
            </text>
            <text
              x={xForFrame(displaySeries[displaySeries.length - 1].frame) - 4}
              y={yForScore(displaySeries[displaySeries.length - 1].p1) + 14}
              className="ss-momentum__endlabel ss-momentum__endlabel--p1"
              textAnchor="end"
            >
              P1 &middot; {displaySeries[displaySeries.length - 1].p1}
            </text>
          </g>
        )}

        {/* Control ribbon: a categorical band, deliberately NOT sharing the
            score-race y-scale (it has no numeric axis of its own). */}
        <g
          className="ss-momentum__ribbon"
          transform={`translate(0, ${CHART_H + RIBBON_GAP})`}
        >
          {ribbon.map((w, idx) => {
            const x0 = xForFrame(w.startFrame);
            const x1 = xForFrame(w.endFrame);
            const fill =
              w.leader == null ? "var(--line)" : PLAYER_COLOR[w.leader];
            return (
              <rect
                key={idx}
                x={x0}
                y={0}
                width={Math.max(x1 - x0, 0.5)}
                height={RIBBON_H}
                fill={fill}
                opacity={w.leader == null ? 0.5 : 0.85}
              />
            );
          })}
        </g>

        {/* Crosshair (hover) */}
        {hover && (
          <line
            x1={hover.x}
            x2={hover.x}
            y1={0}
            y2={VB_H}
            className="ss-momentum__crosshair"
          />
        )}

        {/* Live playback cursor -- distinct from the hover crosshair (a
            bright hairline, matching Player's own playhead treatment). */}
        {nFrames > 0 && (
          <line x1={cursorX} x2={cursorX} y1={0} y2={VB_H} className="ss-momentum__cursor" />
        )}
      </svg>

      {hover && (
        <div
          className="ss-momentum__tooltip"
          style={{ left: `${(hover.x / VB_W) * 100}%` }}
        >
          <div className="mono ss-momentum__tooltip-frame">
            {formatFrame(hover.frame, fps)}
          </div>
          {hoverScore && (
            <div className="ss-momentum__tooltip-scores">
              {/* Fix round 1: text stays in ink tokens, never the series
                  color (the swatch dot carries identity instead) -- see
                  `.ss-momentum__tooltip-scores` in Momentum.css. */}
              <span>
                <i style={{ background: PLAYER_COLOR[0] }} /> P0 {hoverScore.p0}
              </span>
              <span>
                <i style={{ background: PLAYER_COLOR[1] }} /> P1 {hoverScore.p1}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatFrame(frame, fps) {
  if (!fps) return `f${frame}`;
  const s = frame / fps;
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${rem}`;
}
