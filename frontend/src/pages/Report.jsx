import { useEffect, useRef, useState } from "react";
import Player from "../components/Player.jsx";
import { fetchReport, fetchTracks, videoUrl } from "../api.js";
import "./Report.css";

/**
 * Report page: fetches report.json + tracks.json (+ builds the video URL)
 * for a sample or finished match job and renders the annotated Player,
 * plus a rally list derived straight from `report.rallies`.
 *
 * `kind` is "sample" | "match", `id` is the sample id or job id -- both
 * come from App.jsx's hash-route parser.
 */
export default function Report({ kind, id }) {
  const [report, setReport] = useState(null);
  const [tracks, setTracks] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const playheadRef = useRef({ time: 0, frame: 0 });

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setReport(null);
    setTracks(null);

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
        report={report}
        tracks={tracks}
        videoUrl={videoUrl(kind, id)}
        onTimeRef={playheadRef}
      />

      <section className="ss-report__rallies">
        <h2>Rallies</h2>
        {rallyCount === 0 ? (
          <p className="ss-report__empty">No rallies detected in this footage.</p>
        ) : (
          <ol className="ss-report__rallylist">
            {report.rallies.map((r, idx) => (
              <li key={idx} className="ss-report__rallyrow">
                <span className="ss-report__rallyidx mono">{String(idx + 1).padStart(2, "0")}</span>
                <span className="ss-report__rallyspan mono">
                  {formatFrame(r.start_frame, report.fps)}&ndash;{formatFrame(r.end_frame, report.fps)}
                </span>
                <span
                  className={
                    "ss-report__winner" +
                    (r.winner === 0 ? " ss-report__winner--p0" : "") +
                    (r.winner === 1 ? " ss-report__winner--p1" : "")
                  }
                >
                  {r.winner === 0 ? "Player 0 won" : r.winner === 1 ? "Player 1 won" : "No winner recorded"}
                </span>
              </li>
            ))}
          </ol>
        )}
      </section>
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

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  const rem = String(s % 60).padStart(2, "0");
  return `${m}:${rem}`;
}
