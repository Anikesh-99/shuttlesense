import { useCallback, useEffect, useRef, useState } from "react";
import Player from "../components/Player.jsx";
import Momentum from "../components/Momentum.jsx";
import RallyList from "../components/RallyList.jsx";
import { fetchReport, fetchSampleMeta, fetchTracks, videoUrl } from "../api.js";
import "./Report.css";

/**
 * Report page: fetches report.json + tracks.json (+ builds the video URL)
 * for a sample or finished match job and renders the annotated Player,
 * the Momentum chart, and the rally list -- in that order, per the
 * approved mockup (player hero, momentum below, rallies below that).
 *
 * `kind` is "sample" | "match", `id` is the sample id or job id -- both
 * come from App.jsx's hash-route parser.
 *
 * Player ref/onTimeRef contract (Task 18, see Player.jsx's doc comment
 * for the full rationale): `playheadRef` is a plain mutated ref Player
 * writes `{time, frame}` into every animation frame WITHOUT causing
 * Report to re-render; Momentum and RallyList each read it via their own
 * light rAF loops. `playerRef` is the write-side -- Player exposes an
 * imperative `seekToFrame(frame)` so `handleSeek` below can drive
 * playback from a plain frame index without Report needing to know
 * anything about seconds/duration itself.
 */
export default function Report({ kind, id }) {
  const [report, setReport] = useState(null);
  const [tracks, setTracks] = useState(null);
  const [players, setPlayers] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const playheadRef = useRef({ time: 0, frame: 0 });
  const playerRef = useRef(null);

  const handleSeek = useCallback((frame) => {
    playerRef.current?.seekToFrame(frame);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setReport(null);
    setTracks(null);
    setPlayers(null);

    Promise.all([fetchReport(kind, id), fetchTracks(kind, id)])
      .then(([r, t]) => {
        if (cancelled) return;
        setReport(r);
        setTracks(t);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    // Fix round 1: `players` (real competitor names, for the score-race
    // legend -- see Momentum.jsx's doc comment) is fetched SEPARATELY and
    // is purely an enhancement: a "match" job never has one (no meta.json
    // at all for uploads), and even for a sample, a missing/malformed
    // `players` field on an older/hand-edited meta.json must never block
    // or fail the report itself -- so this promise's rejection is swallowed
    // here, not merged into the report/tracks Promise.all above.
    if (kind === "sample") {
      fetchSampleMeta(id)
        .then((meta) => {
          if (cancelled) return;
          setPlayers(Array.isArray(meta?.players) ? meta.players : null);
        })
        .catch(() => {
          // meta.json missing/malformed/endpoint error -- Momentum simply
          // falls back to "Player 0"/"Player 1", same as a match job.
        });
    }

    return () => {
      cancelled = true;
    };
  }, [kind, id]);

  if (loading) {
    return (
      <div className="ss-report ss-report--state">
        <p className="ss-report__eyebrow">AI MATCH ANALYSIS</p>
        <p>Loading report&hellip;</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="ss-report ss-report--state">
        <p className="ss-report__eyebrow">AI MATCH ANALYSIS</p>
        <h2>Couldn&rsquo;t load this report</h2>
        <p className="ss-report__error mono">{error}</p>
        <p>
          Check that the {kind} id <code className="mono">{id}</code> exists and, for a match
          job, that analysis has finished.
        </p>
      </div>
    );
  }

  const rallyCount = report.rallies.length;
  const strokeCount = report.strokes.length;
  const durationS = report.fps ? report.n_frames / report.fps : 0;

  return (
    <div className="ss-report">
      <header className="ss-report__header">
        <div>
          <p className="ss-report__eyebrow">AI MATCH ANALYSIS</p>
          <h1 className="ss-report__title">
            {kind === "match" ? "Match" : "Sample"} <span className="mono">{id}</span>
          </h1>
        </div>
        <dl className="ss-report__stats">
          <div>
            <dt>Rallies</dt>
            <dd className="mono">{rallyCount}</dd>
          </div>
          <div>
            <dt>Strokes</dt>
            <dd className="mono">{strokeCount}</dd>
          </div>
          <div>
            <dt>Length</dt>
            <dd className="mono">{formatDuration(durationS)}</dd>
          </div>
        </dl>
      </header>

      <Player
        ref={playerRef}
        report={report}
        tracks={tracks}
        videoUrl={videoUrl(kind, id)}
        onTimeRef={playheadRef}
      />

      <Momentum report={report} players={players} onTimeRef={playheadRef} onSeek={handleSeek} />

      <RallyList report={report} onTimeRef={playheadRef} onSeek={handleSeek} />
    </div>
  );
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  const rem = String(s % 60).padStart(2, "0");
  return `${m}:${rem}`;
}
