import { useEffect, useState } from "react";
import "./Home.css";

/** Home: lists pre-baked samples from GET /api/samples, each linking to its
 * report page via hash route (#/sample/:id). This is the only "index" the
 * app needs for Task 17's scope -- match-job upload flow is out of scope
 * here (Task 17 brief is the report/player, not the upload UI). */
export default function Home() {
  const [samples, setSamples] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/samples")
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setSamples(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="ss-home">
      <p className="ss-home__eyebrow">SHUTTLESENSE</p>
      <h1 className="ss-home__title">Every rally, tracked.</h1>
      <p className="ss-home__lede">
        Pick a sample below to open its annotated report &mdash; pose skeletons, stroke tags,
        and rally boundaries laid over the original footage.
      </p>

      {error && <p className="ss-home__error mono">Couldn&rsquo;t load samples: {error}</p>}

      {samples === null && !error && <p className="ss-home__muted">Loading samples&hellip;</p>}

      {samples !== null && samples.length === 0 && (
        <p className="ss-home__muted">
          No samples published yet. Once one lands, it&rsquo;ll show up here, or open a match
          report directly at <code className="mono">#/match/&lt;job-id&gt;</code>.
        </p>
      )}

      {samples && samples.length > 0 && (
        <ul className="ss-home__list">
          {samples.map((s) => (
            <li key={s.id}>
              <a className="ss-home__card" href={`#/sample/${encodeURIComponent(s.id)}`}>
                <span className="ss-home__card-title">{s.title}</span>
                <span className="ss-home__card-id mono">{s.id}</span>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
