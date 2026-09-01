// Thin fetch wrapper over the backend API described in
// backend/app/routes.py. Kept dependency-free (no axios) since the surface
// is tiny: list samples, and for either a sample or a finished match job,
// fetch its report/tracks JSON and build its video URL.
//
// `kind` is always one of "sample" | "match" and maps to the backend's
// "/api/samples/:id/..." vs "/api/matches/:id/..." routes.

const API_BASE = "/api";

function resourceBase(kind, id) {
  const segment = kind === "match" ? "matches" : "samples";
  return `${API_BASE}/${segment}/${encodeURIComponent(id)}`;
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSamples() {
  return getJson(`${API_BASE}/samples`);
}

export async function fetchMatchStatus(jobId) {
  return getJson(`${API_BASE}/matches/${encodeURIComponent(jobId)}`);
}

export async function fetchReport(kind, id) {
  return getJson(`${resourceBase(kind, id)}/report`);
}

export async function fetchTracks(kind, id) {
  return getJson(`${resourceBase(kind, id)}/tracks`);
}

export function videoUrl(kind, id) {
  return `${resourceBase(kind, id)}/video`;
}
