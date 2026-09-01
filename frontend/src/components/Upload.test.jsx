// Fix round 1 regression test: after a successful upload+analysis run,
// Upload must call `onClose` (so the modal doesn't linger over the
// freshly-navigated report) in addition to redirecting the hash. Real
// component render via @testing-library/react (jsdom), api.js mocked at
// the module boundary -- this exercises Upload's actual polling/state
// logic, not a hand-simulated version of it.
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import Upload from "./Upload.jsx";
import { fetchMatchStatus, uploadMatch } from "../api.js";

vi.mock("../api.js", () => ({
  uploadMatch: vi.fn(),
  fetchMatchStatus: vi.fn(),
}));

beforeEach(() => {
  window.location.hash = "";
  vi.clearAllMocks();
});

afterEach(() => {
  window.location.hash = "";
});

function selectFileAndSubmit(container) {
  const file = new File(["fake-bytes"], "clip.mp4", { type: "video/mp4" });
  const input = container.querySelector('input[type="file"]');
  fireEvent.change(input, { target: { files: [file] } });
  fireEvent.click(screen.getByText("Upload & analyze"));
}

test("closes the modal AND redirects to #/match/:id once polling reports done", async () => {
  uploadMatch.mockResolvedValue({ job_id: "job-42" });
  fetchMatchStatus.mockResolvedValue({ status: "done", error: null });
  const onClose = vi.fn();

  const { container } = render(<Upload onClose={onClose} />);
  selectFileAndSubmit(container);

  // Real 2s poll interval (Upload.jsx's POLL_INTERVAL_MS) -- waitFor's
  // default timeout comfortably covers one real tick without needing
  // fake timers (which are fiddly to combine correctly with the
  // upload-promise -> setInterval -> fetch-promise chain here).
  await waitFor(() => expect(fetchMatchStatus).toHaveBeenCalledWith("job-42"), {
    timeout: 4000,
  });
  await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1), { timeout: 4000 });

  expect(window.location.hash).toBe("#/match/job-42");
});

test("does NOT close the modal when polling reports failed -- stays open with the error", async () => {
  uploadMatch.mockResolvedValue({ job_id: "job-99" });
  fetchMatchStatus.mockResolvedValue({ status: "failed", error: "unreadable video" });
  const onClose = vi.fn();

  const { container } = render(<Upload onClose={onClose} />);
  selectFileAndSubmit(container);

  await waitFor(() => expect(screen.getByText("unreadable video")).toBeInTheDocument(), {
    timeout: 4000,
  });

  expect(onClose).not.toHaveBeenCalled();
  expect(window.location.hash).toBe("");
});
