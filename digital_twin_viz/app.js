/**
 * app.js — Pipeline visualization controller
 *
 * Connects to the Python WebSocket backend and renders the
 * system pipeline: Environment → Vision → RL Agent → Action
 */

// ── WebSocket ─────────────────────────────────────────────────
let ws = null;
let connected = false;
let latestState = null;
let frame = 0;

const STAGE_NAMES = ['Green', 'Breaker', 'Turning', 'Pink', 'Light Red', 'Red'];
const STAGE_COLORS = ['#3a7d44', '#7da53a', '#c9a825', '#e88c3a', '#d94f3a', '#c0302a'];
const ACTION_INFO = {
    maintain: { icon: '⏸️', label: 'MAINTAIN', detail: 'Keep current conditions', color: '#8b949e' },
    heat: { icon: '🔥', label: 'HEAT', detail: 'Raise temperature +2°C', color: '#f85149' },
    cool: { icon: '❄️', label: 'COOL', detail: 'Lower temperature −2°C', color: '#58a6ff' },
    harvest: { icon: '🔪', label: 'HARVEST', detail: 'Pick tomato now!', color: '#d29922' },
};

// ── Tomato Canvas ─────────────────────────────────────────────
const tomatoCanvas = document.getElementById('tomatoCanvas');
const tCtx = tomatoCanvas.getContext('2d');
const chartCanvas = document.getElementById('chartCanvas');
const chartCtx = chartCanvas.getContext('2d');

function connect() {
    ws = new WebSocket('ws://localhost:8765');

    ws.onopen = () => {
        connected = true;
        document.getElementById('simStatus').classList.add('active');
        document.getElementById('connBadge').textContent = 'CONNECTED';
        document.getElementById('connBadge').classList.add('connected');
    };

    ws.onmessage = (event) => {
        latestState = JSON.parse(event.data);
        updatePipeline(latestState);
    };

    ws.onclose = () => {
        connected = false;
        document.getElementById('simStatus').classList.remove('active');
        document.getElementById('connBadge').textContent = 'DISCONNECTED';
        document.getElementById('connBadge').classList.remove('connected');
        setTimeout(connect, 2000);
    };

    ws.onerror = () => { };
}

function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
    }
}

// ── Pipeline Updates ──────────────────────────────────────────
function updatePipeline(s) {
    if (!s) return;

    // ── Environment ──
    document.getElementById('envTemp').textContent = `${s.temperature.toFixed(1)}°C`;
    document.getElementById('envTemp').style.color = s.temperature > 22 ? '#f85149' : s.temperature < 16 ? '#58a6ff' : '#3fb950';
    document.getElementById('envHum').textContent = `${s.humidity.toFixed(0)}%`;
    document.getElementById('envDay').textContent = `Day ${s.days.toFixed(1)} / ${s.targetDay}`;
    document.getElementById('dayBadge').textContent = `Day ${s.days.toFixed(1)} / ${s.targetDay}`;

    // ── Vision Classification ──
    const stage = Math.min(s.ripenessStage, 5);
    document.getElementById('classStage').textContent = STAGE_NAMES[stage];
    document.getElementById('classStage').style.color = STAGE_COLORS[stage];
    document.getElementById('classRipeness').textContent = s.ripeness.toFixed(2);

    // Simulated classification bars (based on ripeness distance from each stage center)
    for (let i = 0; i < 6; i++) {
        const dist = Math.abs(s.ripeness - i);
        const conf = Math.max(0, 1 - dist * 0.5);
        const bar = document.getElementById(`vb${i}`);
        bar.style.width = `${conf * 100}%`;
        bar.style.background = i === stage ? STAGE_COLORS[i] : 'var(--purple)';
        bar.style.opacity = i === stage ? '1' : '0.4';
    }

    // ── RL Agent ──
    if (s.observation) {
        const obs = s.observation.map(v => v.toFixed(1)).join(', ');
        document.getElementById('obsVector').textContent = `obs: [${obs}]`;
    }

    // Q-values
    if (s.qValues) {
        const qv = s.qValues;
        const maxQ = Math.max(...qv);
        const minQ = Math.min(...qv);
        const range = maxQ - minQ || 1;
        for (let i = 0; i < 4; i++) {
            const pct = ((qv[i] - minQ) / range) * 100;
            const bar = document.getElementById(`q${i}`);
            bar.style.width = `${pct}%`;
            bar.className = 'qbar-fill' + (qv[i] === maxQ ? ' best' : '');
            document.getElementById(`qn${i}`).textContent = qv[i].toFixed(2);
        }
    }

    // Chosen action
    if (s.action && ACTION_INFO[s.action]) {
        const a = ACTION_INFO[s.action];
        document.getElementById('actionText').textContent = `${a.icon} ${a.label}`;
        document.getElementById('chosenAction').style.borderColor = a.color;

        // Action applied node
        document.getElementById('actionBigIcon').textContent = a.icon;
        document.getElementById('actionDetail').textContent = a.detail;

        // Arrows light up
        document.querySelectorAll('.arrow').forEach(el => el.classList.add('active'));
    }

    // Rewards
    if (s.reward !== undefined) {
        const rEl = document.getElementById('stepReward');
        rEl.textContent = s.reward.toFixed(3);
        rEl.className = 'reward-val ' + (s.reward >= 0 ? 'positive' : 'negative');
    }
    if (s.totalReward !== undefined) {
        const tEl = document.getElementById('totalReward');
        tEl.textContent = s.totalReward.toFixed(1);
        tEl.className = 'reward-val ' + (s.totalReward >= 0 ? 'positive' : 'negative');
    }

    // Done state
    if (s.event === 'done') {
        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnPause').disabled = true;
        if (s.harvestQuality !== undefined) {
            document.getElementById('actionDetail').textContent =
                `Quality: ${(s.harvestQuality * 100).toFixed(1)}% · Timing error: ${s.timingError?.toFixed(1) ?? '?'} days`;
        }
    }
}

// ── Render Loop (tomato + chart) ──────────────────────────────
function renderLoop() {
    frame++;
    renderTomato();
    renderChart();
    requestAnimationFrame(renderLoop);
}

function renderTomato() {
    const ctx = tCtx;
    const W = tomatoCanvas.width;
    const H = tomatoCanvas.height;
    const ripeness = latestState ? latestState.ripeness : 0;
    const temp = latestState ? latestState.temperature : 20;
    const actionId = latestState ? latestState.actionId : -1;

    ctx.clearRect(0, 0, W, H);

    // Background
    const warmth = Math.max(0, Math.min(1, (temp - 15) / 10));
    ctx.fillStyle = `rgb(${10 + warmth * 15}, ${14 - warmth * 5}, ${20 - warmth * 8})`;
    ctx.fillRect(0, 0, W, H);

    // Ground
    ctx.fillStyle = '#1a1208';
    ctx.fillRect(0, H - 30, W, 30);
    ctx.fillStyle = '#2a1e14';
    ctx.fillRect(0, H - 32, W, 4);

    // Vine
    const cx = W / 2;
    ctx.strokeStyle = '#3a5a2a';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, H - 30);
    ctx.lineTo(cx, 30);
    ctx.stroke();

    // Branch
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, 70);
    ctx.quadraticCurveTo(cx + 20, 68, cx + 10, 80);
    ctx.stroke();

    // Leaves
    for (let i = 0; i < 4; i++) {
        const ly = 40 + i * 35;
        const side = i % 2 === 0 ? -1 : 1;
        ctx.save();
        ctx.translate(cx + side * 8, ly);
        ctx.rotate(side * 0.4 + Math.sin(frame * 0.015 + i) * 0.1);
        ctx.beginPath();
        ctx.ellipse(side * 15, 0, 14, 5, side * 0.3, 0, Math.PI * 2);
        ctx.fillStyle = `hsl(${125 + i * 3}, 45%, ${28 + i * 2}%)`;
        ctx.fill();
        ctx.restore();
    }

    // TOMATO
    const tY = 100;
    const tR = 25 + ripeness * 3;  // grows slightly as it ripens
    const bobY = Math.sin(frame * 0.02) * 1.5;

    // Stem
    ctx.strokeStyle = '#3a6b2a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx + 10, 80);
    ctx.quadraticCurveTo(cx + 5, tY - tR - 5, cx, tY - tR + bobY);
    ctx.stroke();

    // Calyx
    ctx.fillStyle = '#4a8b3a';
    for (let i = 0; i < 5; i++) {
        const a = (i / 5) * Math.PI * 2 - Math.PI / 2;
        ctx.beginPath();
        ctx.ellipse(cx + Math.cos(a) * 5, tY - tR + bobY + Math.sin(a) * 5, 4, 2, a, 0, Math.PI * 2);
        ctx.fill();
    }

    // Body
    const color = ripenessToColor(ripeness);
    ctx.beginPath();
    ctx.arc(cx, tY + bobY, tR, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Highlight
    const hlG = ctx.createRadialGradient(cx - tR * 0.3, tY + bobY - tR * 0.3, 1, cx, tY + bobY, tR);
    hlG.addColorStop(0, 'rgba(255,255,255,0.3)');
    hlG.addColorStop(0.4, 'rgba(255,255,255,0.05)');
    hlG.addColorStop(1, 'rgba(0,0,0,0.15)');
    ctx.beginPath();
    ctx.arc(cx, tY + bobY, tR, 0, Math.PI * 2);
    ctx.fillStyle = hlG;
    ctx.fill();

    // Ripeness label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(ripeness.toFixed(2), cx, tY + bobY + tR + 20);

    const stage = latestState ? Math.min(latestState.ripenessStage, 5) : 0;
    ctx.font = '11px Inter, sans-serif';
    ctx.fillStyle = STAGE_COLORS[stage];
    ctx.fillText(STAGE_NAMES[stage], cx, tY + bobY + tR + 34);

    // Action effects
    if (actionId === 1) {
        // Heat glow
        ctx.fillStyle = 'rgba(248, 81, 73, 0.08)';
        ctx.fillRect(0, 0, W, H);
        for (let i = 0; i < 5; i++) {
            const fy = H - 30 - Math.abs(Math.sin(frame * 0.05 + i * 1.3)) * 30 - i * 5;
            const fx = 15 + i * (W - 30) / 4;
            ctx.fillStyle = `rgba(255, ${80 + i * 20}, 30, ${0.4 - i * 0.06})`;
            ctx.font = '16px sans-serif';
            ctx.fillText('🔥', fx, fy);
        }
    }
    if (actionId === 2) {
        ctx.fillStyle = 'rgba(88, 166, 255, 0.06)';
        ctx.fillRect(0, 0, W, H);
        for (let i = 0; i < 4; i++) {
            const sy = 20 + ((frame * 0.8 + i * 50) % (H - 40));
            const sx = 20 + i * (W - 40) / 3;
            ctx.fillStyle = 'rgba(200, 230, 255, 0.3)';
            ctx.font = '10px sans-serif';
            ctx.fillText('❄', sx, sy);
        }
    }
    if (actionId === 3) {
        ctx.strokeStyle = '#d29922';
        ctx.lineWidth = 3;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(cx - tR - 8, tY + bobY - tR - 8, tR * 2 + 16, tR * 2 + 16);
        ctx.setLineDash([]);
    }

    // Camera indicator
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(10, 8, 50, 16);
    ctx.fillStyle = '#bc8cff';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('📷 CAM', 14, 20);
}

function ripenessToColor(r) {
    const t = Math.max(0, Math.min(1, r / 5));
    const idx = t * (STAGE_COLORS.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, STAGE_COLORS.length - 1);
    const f = idx - lo;
    const cA = hexToRgb(STAGE_COLORS[lo]);
    const cB = hexToRgb(STAGE_COLORS[hi]);
    return `rgb(${Math.round(cA.r + (cB.r - cA.r) * f)}, ${Math.round(cA.g + (cB.g - cA.g) * f)}, ${Math.round(cA.b + (cB.b - cA.b) * f)})`;
}

function hexToRgb(hex) {
    return { r: parseInt(hex.slice(1, 3), 16), g: parseInt(hex.slice(3, 5), 16), b: parseInt(hex.slice(5, 7), 16) };
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Chart ─────────────────────────────────────────────────────
function renderChart() {
    const ctx = chartCtx;
    const W = chartCanvas.width;
    const H = chartCanvas.height;
    ctx.clearRect(0, 0, W, H);

    if (!latestState || !latestState.history || latestState.history.hours.length < 2) return;

    const hist = latestState.history;
    const pad = { l: 45, r: 15, t: 12, b: 22 };
    const cW = W - pad.l - pad.r;
    const cH = H - pad.t - pad.b;
    const maxH = Math.max(...hist.hours);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    for (let i = 0; i <= 5; i++) {
        const y = pad.t + (i / 5) * cH;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cW, y); ctx.stroke();
    }
    ctx.fillStyle = '#5f6578';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) ctx.fillText(i.toString(), pad.l - 5, pad.t + (1 - i / 5) * cH + 3);

    // Ripeness
    drawLine(ctx, hist.hours, hist.ripeness, maxH, 5, pad, cW, cH, '#3fb950', 2);

    // Temperature normalized
    const tN = hist.temperature.map(t => (t - 12.5) / 2.5);
    drawLine(ctx, hist.hours, tN, maxH, 5, pad, cW, cH, '#f85149', 1.2);

    // Action colors as dots
    const actionColors = ['#8b949e', '#f85149', '#58a6ff', '#d29922'];
    for (let i = 0; i < hist.hours.length; i++) {
        const x = pad.l + (hist.hours[i] / maxH) * cW;
        ctx.beginPath();
        ctx.arc(x, pad.t + cH + 10, 2, 0, Math.PI * 2);
        ctx.fillStyle = actionColors[hist.actions[i]];
        ctx.fill();
    }

    // Legend
    ctx.fillStyle = '#3fb950'; ctx.fillRect(pad.l + 5, pad.t, 10, 2);
    ctx.fillStyle = '#8b949e'; ctx.font = '9px Inter'; ctx.textAlign = 'left';
    ctx.fillText('Ripeness', pad.l + 18, pad.t + 4);
    ctx.fillStyle = '#f85149'; ctx.fillRect(pad.l + 85, pad.t, 10, 2);
    ctx.fillStyle = '#8b949e'; ctx.fillText('Temp', pad.l + 98, pad.t + 4);
}

function drawLine(ctx, xData, yData, xMax, yMax, pad, cW, cH, color, w) {
    ctx.strokeStyle = color;
    ctx.lineWidth = w;
    ctx.beginPath();
    for (let i = 0; i < xData.length; i++) {
        const x = pad.l + (xData[i] / xMax) * cW;
        const y = pad.t + (1 - clamp(yData[i], 0, yMax) / yMax) * cH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

// ── Controls ──────────────────────────────────────────────────
function startSim() {
    send({ cmd: 'start' });
    document.getElementById('btnStart').disabled = true;
    document.getElementById('btnPause').disabled = false;
}
function pauseSim() {
    send({ cmd: 'pause' });
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnPause').disabled = true;
}
function stepSim() {
    send({ cmd: 'step' });
}
function resetSim() {
    send({ cmd: 'reset' });
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnPause').disabled = true;
    document.querySelectorAll('.arrow').forEach(el => el.classList.remove('active'));
    document.getElementById('actionBigIcon').textContent = '⏳';
    document.getElementById('actionDetail').textContent = 'No action yet';
    document.getElementById('stepReward').textContent = '—';
    document.getElementById('totalReward').textContent = '0.0';
    document.getElementById('actionText').textContent = 'Waiting...';
}
function setMode(m) {
    send({ cmd: 'set_mode', mode: m });
    document.querySelectorAll('.toggle').forEach(el => el.classList.remove('active'));
    document.getElementById('mode' + m.charAt(0).toUpperCase() + m.slice(1)).classList.add('active');
}

document.getElementById('speedSlider').addEventListener('input', () => {
    const speed = parseInt(document.getElementById('speedSlider').value);
    document.getElementById('speedLabel').textContent = `${speed}x`;
    send({ cmd: 'set_speed', speed });
});

// ── Init ──────────────────────────────────────────────────────
connect();
requestAnimationFrame(renderLoop);
