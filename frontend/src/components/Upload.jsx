import { useCallback, useEffect, useRef, useState } from "react";
import { fetchMatchStatus, uploadMatch } from "../api.js";
import "./Upload.css";

const POLL_INTERVAL_MS = 2000;

const PHASE_TEXT = {
  idle: "Choose a video (.mp4, .mov, .mkv) to analyze.",
  uploading: "Uploading video…",
  queued: "Queued for analysis…",
  processing: "Analyzing rallies and strokes…",
};

/**
 * Upload: file input + POST /api/matches, then polls GET /api/matches/:id
 * every 2s until the job's `status` is "done" (redirect to #/match/:id AND
 * call `onClose` so the modal doesn't linger over the freshly-navigated
 * report) or "failed" (show the backend's friendly `error` text with a
 * retry option, modal stays open). Rendered as a modal overlay by
 * App.jsx; `onClose` also dismisses it manually via the close button/
 * backdrop click without navigating (only meaningful before a terminal
 * state -- closing mid-poll just stops polling, it does not cancel the
 * backend job).
 */
export default function Upload({ onClose }) {
  const [file, setFile] = useState(null);
  const [phase, setPhase] = useState("idle"); // idle | uploading | queued | processing | failed
  const [error, setError] = useState(null);
  const pollRef = useRef(null);
  const onCloseRef = useRef(onClose);
  // Keep the ref in sync via an effect, not a bare assignment during
  // render -- assigning a ref during render is impure (oxlint's
  // react(refs) rule flags it) even though it "works"; an effect is the
  // correct place to synchronize a ref with a prop.
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const startPolling = useCallback(
    (jobId) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const row = await fetchMatchStatus(jobId);
          if (row.status === "done") {
            stopPolling();
            window.location.hash = `#/match/${encodeURIComponent(jobId)}`;
            // Fix round 1: the modal must not linger over the freshly
            // navigated report -- without this, the user lands on
            // #/match/:id with Upload still overlaying it. Read `onClose`
            // off a ref (rather than a `startPolling` dependency) so a
            // parent re-render mid-poll can never recreate the interval.
            onCloseRef.current?.();
          } else if (row.status === "failed") {
            stopPolling();
            setPhase("failed");
            setError(row.error || "Analysis failed for an unknown reason.");
          } else {
            setPhase(row.status); // "queued" | "processing"
          }
        } catch {
          // Transient network hiccup while polling -- the next tick may
          // succeed; a single failed poll must not flip the whole flow
          // into an error state.
        }
      }, POLL_INTERVAL_MS);
    },
    [stopPolling],
  );

  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || phase === "uploading") return;
    setPhase("uploading");
    setError(null);
    try {
      const { job_id } = await uploadMatch(file);
      setPhase("queued");
      startPolling(job_id);
    } catch (err) {
      setPhase("failed");
      setError(err.message || String(err));
    }
  };

  const handleRetry = () => {
    stopPolling();
    setPhase("idle");
    setError(null);
  };

  const busy = phase === "uploading" || phase === "queued" || phase === "processing";

  return (
    <div className="ss-upload-overlay" role="presentation" onClick={onClose}>
      <div
        className="ss-upload"
        role="dialog"
        aria-modal="true"
        aria-label="Analyze your own video"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="ss-upload__head">
          <h2>Analyze your own video</h2>
          <button type="button" className="ss-upload__close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        {phase !== "failed" ? (
          <form className="ss-upload__form" onSubmit={handleSubmit}>
            <p className="ss-upload__status">{PHASE_TEXT[phase] || PHASE_TEXT.idle}</p>
            <input
              type="file"
              accept="video/mp4,video/quicktime,video/x-matroska,.mp4,.mov,.mkv"
              onChange={handleFileChange}
              disabled={busy}
            />
            <button type="submit" className="ss-upload__submit" disabled={!file || busy}>
              {busy ? "Working…" : "Upload & analyze"}
            </button>
          </form>
        ) : (
          <div className="ss-upload__form">
            <p className="ss-upload__error mono">{error}</p>
            <button type="button" className="ss-upload__submit" onClick={handleRetry}>
              Try again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
