/**
 * renderer.js — 2D Canvas Renderer for the Greenhouse
 *
 * Draws the greenhouse interior, vine rows, tomatoes with dynamic
 * color based on ripeness, temperature haze, and environmental effects.
 */

class GreenhouseRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.frame = 0;
    }

    render(sim) {
        const ctx = this.ctx;
        const W = this.canvas.width;
        const H = this.canvas.height;
        this.frame++;

        // Clear
        ctx.clearRect(0, 0, W, H);

        // Sky / background gradient based on temperature
        const warmth = (sim.temperature - 12.5) / 12.5; // 0..1
        const skyTop = this._lerpColor('#1a2332', '#3a2222', warmth * 0.4);
        const skyBot = this._lerpColor('#1e2a3a', '#2e1a1a', warmth * 0.3);
        const grad = ctx.createLinearGradient(0, 0, 0, H);
        grad.addColorStop(0, skyTop);
        grad.addColorStop(1, skyBot);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H);

        // Greenhouse structure
        this._drawGreenhouse(ctx, W, H, sim);

        // Ground / soil
        ctx.fillStyle = '#2a1e14';
        ctx.fillRect(30, H - 60, W - 60, 50);
        // soil texture
        ctx.fillStyle = '#3b2c1e';
        for (let i = 0; i < 30; i++) {
            const sx = 35 + Math.random() * (W - 70);
            const sy = H - 55 + Math.random() * 40;
            ctx.fillRect(sx, sy, uniform(3, 8), 2);
        }

        // Vine supports
        this._drawVineStructure(ctx, W, H);

        // Tomatoes
        for (const tom of sim.tomatoes) {
            this._drawTomato(ctx, tom, sim);
        }

        // Temperature haze effect
        if (sim.temperature > 22) {
            ctx.fillStyle = `rgba(255, 100, 50, ${(sim.temperature - 22) * 0.02})`;
            ctx.fillRect(0, 0, W, H);
        }

        // Humidity particles
        if (sim.humidity > 75) {
            this._drawMist(ctx, W, H, sim.humidity);
        }

        // HUD overlay on canvas
        this._drawHUD(ctx, W, H, sim);
    }

    _drawGreenhouse(ctx, W, H, sim) {
        // Glass walls
        ctx.strokeStyle = 'rgba(100, 150, 200, 0.3)';
        ctx.lineWidth = 3;

        // Left wall
        ctx.beginPath();
        ctx.moveTo(25, H - 10);
        ctx.lineTo(25, 80);
        ctx.stroke();

        // Right wall
        ctx.beginPath();
        ctx.moveTo(W - 25, H - 10);
        ctx.lineTo(W - 25, 80);
        ctx.stroke();

        // Roof (arch)
        ctx.beginPath();
        ctx.moveTo(25, 80);
        ctx.quadraticCurveTo(W / 2, 15, W - 25, 80);
        ctx.strokeStyle = 'rgba(100, 150, 200, 0.4)';
        ctx.lineWidth = 4;
        ctx.stroke();

        // Roof cross-beams
        ctx.strokeStyle = 'rgba(100, 150, 200, 0.15)';
        ctx.lineWidth = 1;
        for (let i = 1; i <= 5; i++) {
            const t = i / 6;
            const x = 25 + t * (W - 50);
            ctx.beginPath();
            ctx.moveTo(x, H - 10);
            const roofY = 80 - (1 - Math.pow(2 * t - 1, 2)) * 55;
            ctx.lineTo(x, roofY);
            ctx.stroke();
        }

        // Temperature indicator — heater glow
        if (sim.lastAction === 1) {
            const heaterGrad = ctx.createRadialGradient(60, H - 80, 5, 60, H - 80, 60);
            heaterGrad.addColorStop(0, 'rgba(255, 100, 30, 0.3)');
            heaterGrad.addColorStop(1, 'rgba(255, 100, 30, 0)');
            ctx.fillStyle = heaterGrad;
            ctx.fillRect(0, H - 140, 120, 120);

            // Heater icon
            ctx.fillStyle = '#ff6b2b';
            ctx.font = '20px sans-serif';
            ctx.fillText('🔥', 48, H - 70);
        }

        // Cooler glow
        if (sim.lastAction === 2) {
            const coolGrad = ctx.createRadialGradient(W - 60, H - 80, 5, W - 60, H - 80, 60);
            coolGrad.addColorStop(0, 'rgba(50, 150, 255, 0.3)');
            coolGrad.addColorStop(1, 'rgba(50, 150, 255, 0)');
            ctx.fillStyle = coolGrad;
            ctx.fillRect(W - 120, H - 140, 120, 120);

            ctx.fillStyle = '#4da6ff';
            ctx.font = '20px sans-serif';
            ctx.fillText('❄️', W - 72, H - 70);
        }
    }

    _drawVineStructure(ctx, W, H) {
        // Vertical supports
        ctx.strokeStyle = '#4a3a2a';
        ctx.lineWidth = 3;
        const cols = 6;
        for (let c = 0; c < cols; c++) {
            const x = 100 + c * 85;
            ctx.beginPath();
            ctx.moveTo(x, H - 60);
            ctx.lineTo(x, 120);
            ctx.stroke();
        }
        // Horizontal wires
        ctx.strokeStyle = '#5a4a3a';
        ctx.lineWidth = 1;
        for (let r = 0; r < 3; r++) {
            const y = 155 + r * 90;
            ctx.beginPath();
            ctx.moveTo(60, y);
            ctx.lineTo(W - 60, y);
            ctx.stroke();
        }

        // Vine leaves
        ctx.fillStyle = '#2d6b3e';
        for (let c = 0; c < cols; c++) {
            for (let r = 0; r < 3; r++) {
                const bx = 100 + c * 85;
                const by = 140 + r * 90;
                // Draw several leaves
                for (let l = 0; l < 3; l++) {
                    const lx = bx + uniform(-20, 20);
                    const ly = by + uniform(-15, 15);
                    ctx.save();
                    ctx.translate(lx, ly);
                    ctx.rotate(uniform(-0.5, 0.5));
                    ctx.beginPath();
                    ctx.ellipse(0, 0, 12, 6, 0, 0, Math.PI * 2);
                    ctx.fillStyle = `hsl(${120 + uniform(-15, 15)}, ${50 + uniform(-10, 10)}%, ${30 + uniform(-5, 5)}%)`;
                    ctx.fill();
                    ctx.restore();
                }
            }
        }
    }

    _drawTomato(ctx, tom, sim) {
        if (tom.harvested) return;

        const color = ripenessToColor(tom.ripeness);
        const bobY = Math.sin(this.frame * 0.02 + tom.wobble) * 1.5;
        const x = tom.x;
        const y = tom.y + bobY;
        const r = tom.size * (0.7 + tom.ripeness / 5.0 * 0.3); // grows as it ripens

        // Shadow
        ctx.beginPath();
        ctx.ellipse(x + 2, y + r + 3, r * 0.8, r * 0.3, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.15)';
        ctx.fill();

        // Stem
        ctx.strokeStyle = '#3a6b2a';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y - r);
        ctx.lineTo(x + uniform(-3, 3), y - r - 10);
        ctx.stroke();

        // Small calyx (star shape at top)
        ctx.fillStyle = '#4a8b3a';
        for (let i = 0; i < 5; i++) {
            const angle = (i / 5) * Math.PI * 2 - Math.PI / 2;
            ctx.beginPath();
            ctx.ellipse(
                x + Math.cos(angle) * 4,
                y - r + Math.sin(angle) * 4,
                3, 1.5, angle, 0, Math.PI * 2
            );
            ctx.fill();
        }

        // Main tomato body
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Highlight
        const hlGrad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 1, x, y, r);
        hlGrad.addColorStop(0, 'rgba(255,255,255,0.25)');
        hlGrad.addColorStop(0.5, 'rgba(255,255,255,0.05)');
        hlGrad.addColorStop(1, 'rgba(0,0,0,0.1)');
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = hlGrad;
        ctx.fill();

        // Ripeness label (small)
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.font = '9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(tom.ripeness.toFixed(1), x, y + r + 14);
    }

    _drawMist(ctx, W, H, humidity) {
        const intensity = (humidity - 75) / 25;
        for (let i = 0; i < 15; i++) {
            const mx = (this.frame * 0.3 + i * 97) % W;
            const my = 100 + Math.sin(this.frame * 0.01 + i) * 50 + i * 20;
            const mr = 20 + i * 3;
            ctx.beginPath();
            ctx.arc(mx, my, mr, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(200, 220, 255, ${0.02 * intensity})`;
            ctx.fill();
        }
    }

    _drawHUD(ctx, W, H, sim) {
        // Stage label top-left
        const stage = STAGE_NAMES[Math.min(5, Math.round(sim.avgRipeness()))];
        const stageColor = STAGE_COLORS[Math.min(5, Math.round(sim.avgRipeness()))];
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.roundRect(35, 25, 130, 35, 8);
        ctx.fill();
        ctx.fillStyle = stageColor;
        ctx.font = 'bold 13px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Stage: ${stage}`, 45, 47);

        // Day counter top-right
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.roundRect(W - 155, 25, 120, 35, 8);
        ctx.fill();
        ctx.fillStyle = '#60a5fa';
        ctx.font = '13px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(`Day ${sim.daysElapsed.toFixed(1)} / ${sim.targetDay.toFixed(0)}`, W - 45, 47);

        // Temperature bottom-left
        const tempColor = sim.temperature > 22 ? '#f87171' : sim.temperature < 16 ? '#60a5fa' : '#34d399';
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.roundRect(35, H - 40, 100, 28, 8);
        ctx.fill();
        ctx.fillStyle = tempColor;
        ctx.font = '12px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`🌡️ ${sim.temperature.toFixed(1)}°C`, 45, H - 22);
    }

    _lerpColor(a, b, t) {
        const ca = hexToRgb(a);
        const cb = hexToRgb(b);
        const r = Math.round(ca.r + (cb.r - ca.r) * t);
        const g = Math.round(ca.g + (cb.g - ca.g) * t);
        const bl = Math.round(ca.b + (cb.b - ca.b) * t);
        return `rgb(${r},${g},${bl})`;
    }
}

/**
 * Chart renderer for the timeline
 */
class ChartRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
    }

    render(history) {
        const ctx = this.ctx;
        const W = this.canvas.width;
        const H = this.canvas.height;
        ctx.clearRect(0, 0, W, H);

        if (history.hours.length < 2) return;

        const pad = { l: 50, r: 20, t: 10, b: 25 };
        const cW = W - pad.l - pad.r;
        const cH = H - pad.t - pad.b;

        const maxH = Math.max(...history.hours);

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = pad.t + (i / 5) * cH;
            ctx.beginPath();
            ctx.moveTo(pad.l, y);
            ctx.lineTo(pad.l + cW, y);
            ctx.stroke();
        }

        // Y-axis labels
        ctx.fillStyle = '#5f6578';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const y = pad.t + (1 - i / 5) * cH;
            ctx.fillText(i.toString(), pad.l - 8, y + 3);
        }

        // X-axis label
        ctx.textAlign = 'center';
        ctx.fillText('Hours', pad.l + cW / 2, H - 3);

        // Draw ripeness line
        this._drawLine(ctx, history.hours, history.avgRipeness, maxH, 5, pad, cW, cH, '#34d399', 2);
        // Draw temperature line (normalized to 0-5 range: (T-12.5)/2.5)
        const tempNorm = history.temp.map(t => (t - 12.5) / 2.5);
        this._drawLine(ctx, history.hours, tempNorm, maxH, 5, pad, cW, cH, '#f87171', 1.5);

        // Legend
        ctx.fillStyle = '#34d399';
        ctx.fillRect(pad.l + 10, pad.t + 5, 12, 3);
        ctx.fillStyle = '#9aa0b0';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Ripeness', pad.l + 26, pad.t + 10);

        ctx.fillStyle = '#f87171';
        ctx.fillRect(pad.l + 100, pad.t + 5, 12, 3);
        ctx.fillStyle = '#9aa0b0';
        ctx.fillText('Temperature', pad.l + 116, pad.t + 10);
    }

    _drawLine(ctx, xData, yData, xMax, yMax, pad, cW, cH, color, width) {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        for (let i = 0; i < xData.length; i++) {
            const x = pad.l + (xData[i] / xMax) * cW;
            const y = pad.t + (1 - clamp(yData[i], 0, yMax) / yMax) * cH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
}
