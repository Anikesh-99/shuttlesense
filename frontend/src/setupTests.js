// Extends vitest's `expect` with @testing-library/jest-dom matchers
// (toBeInTheDocument, etc) for the component tests added in Task 18's
// "Fix round 1" (Upload.test.jsx, Momentum.test.jsx). Wired up via
// vite.config.js's `test.setupFiles`.
import "@testing-library/jest-dom/vitest";

// This project doesn't enable vitest's `test.globals`, so
// @testing-library/react's normal auto-cleanup-after-each (which detects
// a global `afterEach`) never registers -- without this, multiple tests'
// renders pile up in the same jsdom document within one file. Wiring it
// explicitly here, once, covers every test file.
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
