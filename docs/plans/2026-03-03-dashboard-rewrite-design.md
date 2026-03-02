# Dashboard Rewrite Design

**Date:** 2026-03-03
**Status:** Approved

## Problem

The existing `digital_twin_viz/` frontend (6 files, ~1900 lines) crashes on load due to duplicate `const` declarations across `simulator.js` and `app.js`. It has never been seen working. The dark pipeline-diagram aesthetic does not match the clean academic style needed for thesis defence.

## Solution

Delete all existing frontend files. Write a single `index.html` with inline `<style>` and `<script>`. Keep `server.py` unchanged (WebSocket backend works correctly).

## Layout

3-column grid, white/light-grey background, academic style:

- **Left — Environment:** Large ROYG colour circle (X=1 green → X=0 red), chromatic index X value, stage label, temperature, humidity, day/target
- **Centre — RL Agent:** Compact 16D obs vector display, 3 Q-value bars (Maintain/Heat/Cool), chosen action highlighted
- **Right — Trajectory:** Canvas chart with X (green line) + temperature (orange line) + action colour dots at bottom
- **Bottom strip:** Start / Pause / Step / Reset buttons, speed slider (1–20×)
- **Header bar:** Title, connection status badge, day counter

## Architecture

- Single `index.html` (~400 lines) — inline CSS + inline JS
- WebSocket: `ws://127.0.0.1:8765`, auto-reconnect every 2s
- ROYG colour mapped via 6-stop interpolation (same stops as `run_sim.py`)
- Chart: native Canvas 2D API, no external charting library
- Q-value bars: CSS width % (no canvas needed)
- Served via `python -m http.server 8080` from `digital_twin_viz/`

## Files to Delete

- `app.js`, `simulator.js`, `renderer.js`, `style.css`, `index.html`

## Files to Create

- `index.html` (single file, replaces all of the above)

## Files to Keep

- `server.py` (unchanged)
