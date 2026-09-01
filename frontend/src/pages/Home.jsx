import { useEffect, useState } from "react";
import { fetchSamples } from "../api.js";
import "./Home.css";

/**
 * Home / landing route ("#" or no hash): the zero-action demo requirement
 * -- on mount, fetch `/api/samples` and IMMEDIATELY navigate to
 * `#/sample/<first>` (no click required). `location.hash = ...` triggers
 * App's hashchange listener, which swaps this component out for Report,
 * so there's nothing more for Home to render in the success case beyond a
 * brief "loading" flash.
 *
 * Two non-zero-action fallbacks, both legitimate (can't redirect to
 * nothing): a fetch error, or an empty samples list (Task 19 hasn't
 * published any yet) -- both surface a message plus a nudge toward the
 * "Analyze your own video" button in the topbar, which is the only other
 * way into the app.
 */
export default function Home() {
  const [state, setState] = useState("loading"); // loading | empty | error
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetchSamples()
      .then((samples) => {
        if (cancelled) return;
        if (samples.length > 0) {
          // replace, not push, so the (auto) navigation doesn't leave a
          // dead "home" entry in browser history to land back on.
          window.location.replace(`#/sample/${encodeURIComponent(samples[0].id)}`);
          return;
        }
        setState("empty");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || String(err));
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="ss-home">
      <p className="ss-home__eyebrow">SHUTTLESENSE</p>
      <h1 className="ss-home__title">Every rally, tracked.</h1>

      {state === "loading" && <p className="ss-home__muted">Loading a sample report&hellip;</p>}

      {state === "error" && (
        <p className="ss-home__error mono">Couldn&rsquo;t load samples: {error}</p>
      )}

      {state === "empty" && (
        <p className="ss-home__muted">
          No samples published yet. Use &ldquo;Analyze your own video&rdquo; above to upload
          footage, or open a match report directly at{" "}
          <code className="mono">#/match/&lt;job-id&gt;</code>.
        </p>
      )}
    </div>
  );
}
