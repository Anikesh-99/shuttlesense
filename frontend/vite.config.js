import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  // Fix round 1: vitest's own transform pipeline in this toolchain
  // version doesn't pick up @vitejs/plugin-react's oxc/rolldown-native JSX
  // config the way the real `vite build`/`vite dev` pipeline does (it
  // falls back to esbuild's classic JSX transform, which needs `React` in
  // scope) -- explicit here so component tests (added in this fix round)
  // don't require every source file to import React just to satisfy the
  // test runner. Scoped to `command !== 'build'` (i.e. vitest's internal
  // "serve"-mode server) so `vite build` keeps using the real oxc JSX
  // transform untouched, instead of just silently overriding it (which
  // still worked, but printed a "both esbuild and oxc options were set"
  // warning on every build).
  esbuild:
    command === 'build'
      ? undefined
      : { jsx: 'automatic', jsxImportSource: 'react' },
  test: {
    // Task 18 fix round 1: component tests (Upload's close-on-done,
    // Momentum's ribbon-only legend) render real JSX via
    // @testing-library/react, which needs a DOM -- the pure-function
    // tests from Task 17/18 (player.test.js, stats.test.js) don't touch
    // the DOM and are unaffected by this (jsdom is a superset).
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.js'],
  },
}))
