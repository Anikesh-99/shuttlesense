import { useEffect, useState } from "react";
import { rallySummaries } from "../lib/stats.js";
import "./RallyList.css";

/**
 * RallyList: one row per rally (from `rallySummaries`), each clickable to
 * seek the video to that rally's start frame. The row for whichever rally
 * currently contains the live playhead is highlighted.
 *
 * Read-side of the Task 18 ref contract (see Player.jsx / Momentum.jsx):
 * like Momentum, RallyList runs its own small rAF loop reading
 * `onTimeRef.current.frame` rather than Report lifting `currentFrame`
 * into shared state -- it only calls setState when the ACTIVE ROW INDEX
 * changes (not on every frame), so in practice this re-renders only a
 * few times per rally, not 60x/sec.
 *
 * Winner cell identity (final-review fix, see Momentum.jsx's doc comment
 * for the full two-axes rationale): `rally.winner` is ShuttleSet's
 * MATCH-scoped label identity (`0`/`1` = who eventually wins/loses the
 * match), NOT the pipeline's per-frame court-side skeleton slot -- the two
 * have NO reliable correspondence (measured 5.7%/68.6% agreement across
 * two samples). This cell must therefore never use `PLAYER_COLOR`
 * (green/blue, the skeleton palette) for the winner text, and must prefer
 * the real competitor NAME (`players` prop, `[name0, name1]`, same shape
 * and indexing as Momentum's) over a bare "Player N" -- exactly like
 * Momentum's score-race labeling. `players` is absent for every match-job
 * upload and for any sample missing ShuttleSet ground truth; the cell then
 * falls back to a neutral-ink "Winner: P{n}" (still never PLAYER_COLOR).
 */
export default function RallyList({ report, players, onTimeRef, onSeek }) {
  const [activeIndex, setActiveIndex] = useState(-1);

  const rallies = report?.rallies || [];
  const strokes = report?.strokes || [];
  const fps = report?.fps || 30;
  const summaries = rallySummaries(rallies, strokes);

  const hasPlayers =
    Array.isArray(players) &&
    players.length === 2 &&
    players.every((p) => typeof p === "string" && p);
  const winnerText = (winner) => {
    if (winner == null) return "No winner recorded";
    return hasPlayers ? `${players[winner]} won` : `Winner: P${winner}`;
  };

  useEffect(() => {
    if (!onTimeRef) return undefined;
    let raf;
    let lastIndex = -2;
    const tick = () => {
      const frame = onTimeRef.current?.frame ?? 0;
      const idx = summaries.findIndex((r) => frame >= r.startFrame && frame < r.endFrame);
      if (idx !== lastIndex) {
        lastIndex = idx;
        setActiveIndex(idx);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onTimeRef, report]);

  if (summaries.length === 0) {
    return (
      <section className="ss-rallylist">
        <h2>Rallies</h2>
        <p className="ss-rallylist__empty">No rallies detected in this footage.</p>
      </section>
    );
  }

  return (
    <section className="ss-rallylist">
      <h2>Rallies</h2>
      <ol className="ss-rallylist__rows">
        {summaries.map((r, idx) => {
          const rally = rallies[idx];
          const winner = rally ? rally.winner : null;
          return (
            <li key={r.index}>
              <button
                type="button"
                className={
                  "ss-rallylist__row" + (idx === activeIndex ? " ss-rallylist__row--active" : "")
                }
                onClick={() => onSeek && onSeek(r.startFrame)}
              >
                <span className="ss-rallylist__idx mono">
                  {String(r.index + 1).padStart(2, "0")}
                </span>
                <span className="ss-rallylist__span mono">
                  {formatFrame(r.startFrame, fps)}&ndash;{formatFrame(r.endFrame, fps)}
                </span>
                <span className="ss-rallylist__shots mono">{r.shots} shots</span>
                <span className="ss-rallylist__endedby">
                  {r.endedBy ? `Ended by ${r.endedBy}` : "No strokes recorded"}
                </span>
                {/* Neutral ink only (see doc comment above) -- never
                    PLAYER_COLOR/the skeleton palette, no matter which
                    branch of `winnerText` fires. */}
                <span className="ss-rallylist__winner">{winnerText(winner)}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function formatFrame(frame, fps) {
  if (!fps) return `f${frame}`;
  const s = frame / fps;
  const m = Math.floor(s / 60);
  const rem = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${rem}`;
}
