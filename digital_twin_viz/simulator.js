/**
 * simulator.js — NO LOCAL SIMULATION
 *
 * All simulation runs on the Python backend via WebSocket.
 * This file only contains color/stage utilities used by the renderer.
 */

const STAGE_NAMES = ['Green', 'Breaker', 'Turning', 'Pink', 'Light Red', 'Red'];
const STAGE_COLORS = [
    '#3a7d44',  // Green
    '#7da53a',  // Breaker
    '#c9a825',  // Turning
    '#e88c3a',  // Pink
    '#d94f3a',  // Light Red
    '#c0302a',  // Red
];

function clamp(val, lo, hi) { return Math.max(lo, Math.min(hi, val)); }

function ripenessToColor(ripeness) {
    const t = clamp(ripeness / 5.0, 0, 1);
    const idx = t * (STAGE_COLORS.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, STAGE_COLORS.length - 1);
    const frac = idx - lo;
    const cA = hexToRgb(STAGE_COLORS[lo]);
    const cB = hexToRgb(STAGE_COLORS[hi]);
    return `rgb(${Math.round(cA.r + (cB.r - cA.r) * frac)}, ${Math.round(cA.g + (cB.g - cA.g) * frac)}, ${Math.round(cA.b + (cB.b - cA.b) * frac)})`;
}

function hexToRgb(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return { r, g, b };
}

function uniform(a, b) { return a + Math.random() * (b - a); }
