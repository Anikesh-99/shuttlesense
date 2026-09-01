import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./Player.css";

// ---------------------------------------------------------------------------
// Pure helpers (exported for vitest -- see player.test.js). None of these
// touch the DOM or React state; every one takes an explicit `frame`/`t`
// rather than reading "now" off a video element, so they're trivially
// testable and never look ahead of the data they're given.
// ---------------------------------------------------------------------------

/** Map a video's currentTime (seconds) to a tracks/report frame index. */
export function frameForTime(t, fps) {
  return Math.round(t * fps);
}

/** How many frames make up the +-0.4s "active stroke" window at `fps`. */
const STROKE_WINDOW_S = 0.4;

/**
 * Find the stroke event closest to `frame` within +-0.4s (converted to
 * frames at `fps`), or null if none qualifies. `strokes` is the report's
 * flat `strokes` array (unsorted-safe -- this does a linear scan, fine for
 * per-match event counts).
 */
export function activeStroke(strokes, frame, fps) {
  const windowFrames = STROKE_WINDOW_S * fps;
  let best = null;
  let bestDist = Infinity;
  for (const s of strokes) {
    const dist = Math.abs(s.frame - frame);
    if (dist <= windowFrames && dist < bestDist) {
      best = s;
      bestDist = dist;
    }
  }
  return best;
}

/**
 * Scale a [x, y] keypoint from the report's native video pixel space
 * (`report.width` x `report.height`) into the on-screen canvas box
 * (`canvasW` x `canvasH`). The <video> element is rendered at its native
 * aspect ratio (no CSS object-fit distortion), and the canvas is sized to
 * match it exactly via ResizeObserver, so a plain per-axis ratio is
 * correct -- no letterbox offset to account for.
 */
export function scaleKpt(kpt, videoW, videoH, canvasW, canvasH) {
  const [x, y] = kpt;
  if (!videoW || !videoH) return [0, 0];
  return [(x * canvasW) / videoW, (y * canvasH) / videoH];
}

// ---------------------------------------------------------------------------
// Drawing constants
// ---------------------------------------------------------------------------

const PLAYER_COLORS = ["#63d5a0", "#6fa8ff"]; // player 0 = green, player 1 = blue
const PRESENCE_SCORE_THR = 0.3; // mirrors backend PIPELINE.PRESENCE_THR
const JOINT_RADIUS = 3.5;
const EDGE_WIDTH = 2.5;

/** A player slot is "present" in a frame if its mean keypoint score clears
 * the presence threshold -- an absent slot is all-zero/near-zero kpts and
 * must never be drawn as a skeleton collapsed at the origin. */
function isPresent(scoresForPlayerFrame) {
  if (!scoresForPlayerFrame || scoresForPlayerFrame.length === 0) return false;
  const mean =
    scoresForPlayerFrame.reduce((a, b) => a + b, 0) / scoresForPlayerFrame.length;
  return mean >= PRESENCE_SCORE_THR;
}

function formatClock(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const rem = String(s % 60).padStart(2, "0");
  return `${m}:${rem}`;
}

// ---------------------------------------------------------------------------
// Player component
// ---------------------------------------------------------------------------

/**
 * Annotated video player: a <video> with an absolutely-positioned <canvas>
 * skeleton/stroke overlay and a custom seek bar (rally segments + playhead)
 * beneath it.
 *
 * Props:
 *   report   - MatchReport.to_dict() shape: {fps, width, height, n_frames,
 *              rallies:[{start_frame,end_frame,winner}], strokes:[...]}
 *   tracks   - {fps, edges:[[i,j]...], kpts:[[[x,y]x17]x2]xT, scores:[[sx17]x2]xT}
 *   videoUrl - string src for the <video>
 *   onTimeRef - optional React ref; Player writes {time, frame} into
 *              onTimeRef.current every animation frame WITHOUT causing a
 *              re-render here, so a sibling (e.g. a Task 18 momentum chart)
 *              can read the live playhead without every frame re-rendering
 *              this whole tree.
 */
export default function Player({ report, tracks, videoUrl, onTimeRef }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const seekBarRef = useRef(null);
  const rafRef = useRef(null);

  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);

  const fps = tracks?.fps || report?.fps || 30;
  const videoW = report?.width || 0;
  const videoH = report?.height || 0;
  const nFrames = report?.n_frames || 0;
  const rallies = report?.rallies || [];
  const strokes = report?.strokes || [];
  const edges = tracks?.edges || [];
  const kptsAll = tracks?.kpts || [];
  const scoresAll = tracks?.scores || [];

  // Size the canvas to the rendered <video> box.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setCanvasSize({ w: Math.round(width), h: Math.round(height) });
      }
    });
    ro.observe(video);
    return () => ro.disconnect();
  }, []);

  const currentFrame = useMemo(
    () => Math.min(Math.max(frameForTime(currentTime, fps), 0), Math.max(nFrames - 1, 0)),
    [currentTime, fps, nFrames],
  );

  const stroke = useMemo(
    () => activeStroke(strokes, currentFrame, fps),
    [strokes, currentFrame, fps],
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { w, h } = canvasSize;
    if (w === 0 || h === 0) return;

    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    if (!showOverlay) return;

    const frameKpts = kptsAll[currentFrame];
    const frameScores = scoresAll[currentFrame];

    if (frameKpts) {
      for (let player = 0; player < frameKpts.length; player++) {
        const kpts = frameKpts[player];
        const scores = frameScores ? frameScores[player] : null;
        if (!isPresent(scores)) continue;

        const color = PLAYER_COLORS[player] || "#f5b942";
        const scaled = kpts.map((k) => scaleKpt(k, videoW, videoH, w, h));

        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = EDGE_WIDTH;
        ctx.lineCap = "round";

        for (const [i, j] of edges) {
          if (!scaled[i] || !scaled[j]) continue;
          if (scores && (scores[i] < PRESENCE_SCORE_THR || scores[j] < PRESENCE_SCORE_THR)) {
            continue;
          }
          ctx.beginPath();
          ctx.moveTo(scaled[i][0], scaled[i][1]);
          ctx.lineTo(scaled[j][0], scaled[j][1]);
          ctx.stroke();
        }

        scaled.forEach(([x, y], idx) => {
          if (scores && scores[idx] < PRESENCE_SCORE_THR) return;
          ctx.beginPath();
          ctx.arc(x, y, JOINT_RADIUS, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    }

    if (stroke) {
      const color = PLAYER_COLORS[stroke.player] || "#f5b942";
      const label = `${stroke.stroke.toUpperCase()} · ${Math.round(stroke.confidence * 100)}%`;
      ctx.font = "600 13px 'IBM Plex Mono', ui-monospace, monospace";
      const textW = ctx.measureText(label).width;
      const padX = 10;
      const padY = 7;
      const chipW = textW + padX * 2;
      const chipH = 18 + padY;
      const chipX = 14;
      const chipY = 14;

      ctx.fillStyle = "rgba(10, 13, 18, 0.82)";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      roundRect(ctx, chipX, chipY, chipW, chipH, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(chipX + padX / 2 + 3, chipY + chipH / 2, 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#eef1f6";
      ctx.textBaseline = "middle";
      ctx.fillText(label, chipX + padX + 6, chipY + chipH / 2 + 1);
    }
  }, [canvasSize, showOverlay, currentFrame, kptsAll, scoresAll, edges, videoW, videoH, stroke]);

  // rAF loop: reads video.currentTime directly (not React state) so the
  // overlay stays in lockstep with playback, and only pushes into React
  // state (currentTime) for UI bits that need it (seek bar, chip text).
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;

    const tick = () => {
      const t = video.currentTime;
      setCurrentTime(t);
      if (onTimeRef) {
        onTimeRef.current = { time: t, frame: frameForTime(t, fps) };
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fps]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (video) setDuration(video.duration || 0);
  };

  const seekToClientX = useCallback(
    (clientX) => {
      const bar = seekBarRef.current;
      const video = videoRef.current;
      if (!bar || !video || !duration) return;
      const rect = bar.getBoundingClientRect();
      const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
      video.currentTime = ratio * duration;
    },
    [duration],
  );

  const handleSeekPointerDown = (e) => {
    seekToClientX(e.clientX);
    const onMove = (ev) => seekToClientX(ev.clientX);
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const playheadRatio = duration ? currentTime / duration : 0;

  const handleSeekKeyDown = (e) => {
    const video = videoRef.current;
    if (!video || !duration) return;
    const step = 1 / fps;
    if (e.key === "ArrowRight") {
      video.currentTime = Math.min(video.currentTime + step, duration);
      e.preventDefault();
    } else if (e.key === "ArrowLeft") {
      video.currentTime = Math.max(video.currentTime - step, 0);
      e.preventDefault();
    } else if (e.key === "Home") {
      video.currentTime = 0;
      e.preventDefault();
    } else if (e.key === "End") {
      video.currentTime = duration;
      e.preventDefault();
    }
  };

  return (
    <div className="ss-player">
      <div className="ss-player__stage">
        <video
          ref={videoRef}
          src={videoUrl}
          className="ss-player__video"
          onLoadedMetadata={handleLoadedMetadata}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          playsInline
        />
        <canvas
          ref={canvasRef}
          className="ss-player__canvas"
          style={{ width: canvasSize.w, height: canvasSize.h }}
        />
        <div className="ss-player__bracket ss-player__bracket--tl" />
        <div className="ss-player__bracket ss-player__bracket--tr" />
        <div className="ss-player__bracket ss-player__bracket--bl" />
        <div className="ss-player__bracket ss-player__bracket--br" />
      </div>

      <div className="ss-player__controls">
        <button
          type="button"
          className="ss-player__playbtn"
          onClick={togglePlay}
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>

        <span className="ss-player__clock mono">
          {formatClock(currentTime)} / {formatClock(duration)}
        </span>

        <div
          ref={seekBarRef}
          className="ss-player__seekbar"
          onPointerDown={handleSeekPointerDown}
          onKeyDown={handleSeekKeyDown}
          role="slider"
          aria-label="Seek"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(currentTime)}
          tabIndex={0}
        >
          {rallies.map((r, idx) => {
            const start = nFrames ? r.start_frame / nFrames : 0;
            const end = nFrames ? r.end_frame / nFrames : 0;
            const width = Math.max(end - start, 0);
            return (
              <div
                key={idx}
                className="ss-player__rally"
                style={{ left: `${start * 100}%`, width: `${width * 100}%` }}
                title={`Rally ${idx + 1}${r.winner != null ? ` · winner: player ${r.winner}` : ""}`}
              />
            );
          })}
          <div className="ss-player__playhead" style={{ left: `${playheadRatio * 100}%` }} />
        </div>

        <label className="ss-player__toggle">
          <input
            type="checkbox"
            checked={showOverlay}
            onChange={(e) => setShowOverlay(e.target.checked)}
          />
          <span>Overlays</span>
        </label>
      </div>
    </div>
  );
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
