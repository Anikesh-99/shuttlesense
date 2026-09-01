import { useEffect, useState } from "react";
import Home from "./pages/Home.jsx";
import Report from "./pages/Report.jsx";
import Upload from "./components/Upload.jsx";
import "./App.css";

/**
 * Parse `location.hash` into a route. Supports "#/sample/:id" and
 * "#/match/:id" (both required by the brief) and otherwise falls back to
 * the home/sample-index route. No router dependency -- the whole app is
 * two routes, a hashchange listener is plenty.
 */
export function parseHash(hash) {
  const clean = (hash || "").replace(/^#\/?/, "");
  const parts = clean.split("/").filter(Boolean);
  if (parts.length >= 2 && (parts[0] === "sample" || parts[0] === "match")) {
    return { kind: parts[0], id: parts.slice(1).join("/") };
  }
  return null;
}

function useHashRoute() {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onHashChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  return route;
}

function App() {
  const route = useHashRoute();
  const [uploadOpen, setUploadOpen] = useState(false);

  return (
    <div className="ss-app">
      <a className="ss-app__skip" href="#ss-main">
        Skip to content
      </a>
      <header className="ss-app__topbar">
        <a className="ss-app__brand" href="#/">
          Shuttle<span className="ss-app__brand-accent">Sense</span>
        </a>
        <button
          type="button"
          className="ss-app__upload-cta"
          onClick={() => setUploadOpen(true)}
        >
          Analyze your own video
        </button>
      </header>
      <main id="ss-main">
        {route ? <Report kind={route.kind} id={route.id} /> : <Home />}
      </main>
      {uploadOpen && <Upload onClose={() => setUploadOpen(false)} />}
    </div>
  );
}

export default App;
