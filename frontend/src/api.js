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

/** POST a video file as multipart/form-data to /api/matches. Resolves to
 * `{job_id}` on 202; throws an Error whose message is the backend's
 * `detail` string when available (400 bad file type/too long, 413 too
 * big), falling back to "<status> <statusText>" otherwise. */
export async function uploadMatch(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/matches`, { method: "POST", body: form });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      // non-JSON error body -- keep the status-line fallback above
    }
    throw new Error(detail);
  }
  return res.json();
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
