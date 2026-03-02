#!/usr/bin/env python3
"""
Edge-RL Live Simulation — Pygame + Box2D Visualizer
====================================================
Runs the trained DQN policy on the TomatoRipeningEnv and renders a
real-time physics-based dashboard suitable for thesis defence demos.

Controls:
    P  — Screenshot (saved to project root)
    R  — Restart episode (after completion or any time)
    Q  — Quit
    SPACE — Pause / Resume

Layout (1400 × 860):
    ┌────────────┬──────────────────────────┬──────────────────┐
    │  LEFT      │       CENTRE             │  RIGHT           │
    │  Metrics   │   Diurnal Sky + Tomato   │  Live Charts (4) │
    │  300 px    │   800 px                 │  300 px          │
    └────────────┴──────────────────────────┴──────────────────┘
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from collections import deque

import numpy as np
import pygame
import yaml
from Box2D import b2World

from stable_baselines3 import DQN
from ml_training.rl.environment import TomatoRipeningEnv

# ── Layout ────────────────────────────────────────────────────────────
W, H          = 1400, 860
PANEL_L       = 300
PANEL_R       = 300
CENTRE_W      = W - PANEL_L - PANEL_R
FPS           = 60
HIST          = 200

# ── Colors ────────────────────────────────────────────────────────────
C_BG     = (8,  10, 18)
C_PANEL  = (18, 20, 32)
C_BORDER = (40, 44, 64)
C_ACCENT = (0,  180, 255)
C_GREEN  = (46, 204, 113)
C_RED    = (231, 76,  60)
C_BLUE   = (88, 166, 255)
C_YELLOW = (241, 196, 15)
C_TEXT   = (200, 205, 220)
C_DIM    = (90,  95, 115)
C_WHITE  = (240, 245, 255)

# ROYG gradient: index X=1.0 (unripe green) → X=0.0 (ripe red)
_ROYG = [
    (0.20, 0.78, 0.22),   # X=1.0  deep green
    (0.55, 0.75, 0.15),   # X=0.85 yellow-green
    (0.90, 0.80, 0.08),   # X=0.65 yellow
    (0.95, 0.55, 0.08),   # X=0.40 orange
    (0.90, 0.22, 0.10),   # X=0.15 light-red
    (0.75, 0.08, 0.08),   # X=0.0  deep red
]

USDA_STAGES = [
    (0.90, "MATURE GREEN",  C_GREEN),
    (0.75, "BREAKER",       (150, 200, 50)),
    (0.55, "TURNING",       C_YELLOW),
    (0.35, "PINK",          (220, 130, 80)),
    (0.15, "LIGHT RED",     C_RED),
    (0.00, "RED — HARVEST", (180, 20, 20)),
]

ACTION_INFO = {
    0: ("MAINTAIN", C_DIM),
    1: ("▲ HEAT",   C_RED),
    2: ("▼ COOL",   C_BLUE),
}


# ── Pure helpers ──────────────────────────────────────────────────────
def royg_color(x: float) -> tuple[int, int, int]:
    """Smooth ROYG gradient: x=1 → green, x=0 → red."""
    x = max(0.0, min(1.0, x))
    t = (1.0 - x) * (len(_ROYG) - 1)
    lo = int(t);  hi = min(lo + 1, len(_ROYG) - 1);  f = t - lo
    return tuple(int((_ROYG[lo][c] + (_ROYG[hi][c] - _ROYG[lo][c]) * f) * 255)
                 for c in range(3))


def usda_stage(x: float):
    for thr, name, col in USDA_STAGES:
        if x > thr:
            return name, col
    return USDA_STAGES[-1][1], USDA_STAGES[-1][2]


def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ── Sky background ────────────────────────────────────────────────────
def draw_sky(surf, x0, y0, w, h, hour: float):
    """Sinusoidal day/night gradient — peak brightness at 14:00."""
    t = (math.sin(math.pi * (hour - 6) / 13) if 6 <= hour <= 19 else 0)
    t = max(0.0, t)
    night = (5, 8, 20);  day = (15, 40, 80);  noon = (25, 80, 150)
    sky = lerp_color(lerp_color(night, day, t), noon, max(0, t - 0.5) * 2) if t < 0.5 \
          else lerp_color(day, noon, (t - 0.5) * 2)
    pygame.draw.rect(surf, sky, (x0, y0, w, h))

    # Sun / moon arc
    is_day = 6 <= hour <= 19
    bt = (hour - 6) / 13 if is_day else ((hour - 19) % 24) / 11
    bx = x0 + int(bt * w)
    by = y0 + 48 - int(math.sin(bt * math.pi) * 36)
    if is_day:
        pygame.draw.circle(surf, (255, 230, 60), (bx, by), 22)
        pygame.draw.circle(surf, (255, 245, 130), (bx, by), 14)
    else:
        pygame.draw.circle(surf, (200, 210, 235), (bx, by), 12)
        pygame.draw.circle(surf, (140, 155, 180), (bx, by), 7)


# ── Particle system ───────────────────────────────────────────────────
class _P:
    __slots__ = ("x","y","vx","vy","life","maxl","char","color")
    def __init__(self, x,y,vx,vy,life,char,color):
        self.x=x; self.y=y; self.vx=vx; self.vy=vy
        self.life=self.maxl=life; self.char=char; self.color=color
    def step(self): self.x+=self.vx; self.y+=self.vy; self.life-=1
    def alive(self): return self.life>0
    def alpha(self): return int(255*self.life/self.maxl)


def _heat(ps, cx, cy, n=3):
    for _ in range(n):
        a = math.pi/2 + np.random.uniform(-0.5, 0.5)
        s = np.random.uniform(0.8, 2.0)
        ps.append(_P(cx+np.random.randint(-30,30), cy+np.random.randint(20,55),
                     math.cos(a)*s, -math.sin(a)*s,
                     np.random.randint(25,50), "🔥",
                     (255, np.random.randint(70,150), 20)))


def _cool(ps, cx, cy, n=2):
    for _ in range(n):
        ps.append(_P(cx+np.random.randint(-60,60), np.random.randint(10,cy-60),
                     np.random.uniform(-0.3,0.3), np.random.uniform(0.4,1.2),
                     np.random.randint(30,60), "❄",
                     (150, 210, 255)))


# ── Chart ─────────────────────────────────────────────────────────────
def draw_chart(surf, rect, data: deque, color, title,
               y_min, y_max, unit="", threshold=None, t_col=C_RED):
    rx, ry, rw, rh = rect
    pygame.draw.rect(surf, (12,14,24), rect, border_radius=4)
    pygame.draw.rect(surf, C_BORDER, rect, 1, border_radius=4)
    pad = 6
    cw = rw - pad*2;  ch = rh - pad*2 - 14
    ox = rx+pad;       oy = ry+pad+14

    fnt = pygame.font.SysFont("Arial", 12, bold=True)
    val_str = f"{data[-1]:.2f}{unit}" if data else "—"
    surf.blit(fnt.render(f"{title}: {val_str}", True, color), (rx+pad, ry+3))

    if len(data) < 2:
        return

    for i in range(5):
        gy = oy + ch - int(i/4*ch)
        pygame.draw.line(surf, (28,32,48), (ox, gy), (ox+cw, gy))
        lbl_v = y_min + (y_max-y_min)*i/4
        surf.blit(fnt.render(f"{lbl_v:.0f}", True, C_DIM), (rx, gy-6))

    if threshold is not None and y_min <= threshold <= y_max:
        ty = oy + ch - int((threshold-y_min)/(y_max-y_min)*ch)
        pygame.draw.line(surf, t_col, (ox, ty), (ox+cw, ty), 1)

    arr = list(data)
    span = max(y_max-y_min, 1e-6)
    pts = [(ox+int(i/max(len(arr)-1,1)*cw),
            oy+ch-int(max(0,min(1,(v-y_min)/span))*ch))
           for i,v in enumerate(arr)]
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, 2)


# ── Label helper ──────────────────────────────────────────────────────
def draw_metric(surf, fs, fm, label, value, x, y, vc=C_WHITE):
    surf.blit(fs.render(label, True, C_DIM),   (x, y))
    surf.blit(fm.render(value, True, vc),       (x, y+18))


# ── Model discovery ───────────────────────────────────────────────────
def find_model() -> str:
    for pattern in ("outputs/rl_*/best_model/best_model.zip",
                    "outputs/rl_*/final_model.zip"):
        hits = sorted(Path(".").glob(pattern), reverse=True)
        if hits:
            p = str(hits[0])
            return p.removesuffix(".zip")
    return None


# ── Main ─────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Edge-RL · Tomato Ripening Digital Twin")
    clock = pygame.time.Clock()

    fs   = pygame.font.SysFont("Arial",     13)
    fm   = pygame.font.SysFont("Verdana",   17, bold=True)
    flg  = pygame.font.SysFont("Verdana",   26, bold=True)
    fttl = pygame.font.SysFont("Verdana",   14, bold=True)
    fmono= pygame.font.SysFont("Courier New",11)
    femo = pygame.font.SysFont("Segoe UI Emoji", 18)

    # Load model
    mpath = find_model()
    if not mpath:
        print("ERROR: No trained model found in outputs/rl_*/"); sys.exit(1)
    print(f"Loading: {mpath}")

    with open("ml_training/config.yaml") as f:
        config = yaml.safe_load(f)

    try:
        model = DQN.load(mpath)
        print("✅ DQN loaded")
    except Exception as e:
        print(f"ERROR loading model: {e}"); sys.exit(1)

    # History
    hist_x   = deque(maxlen=HIST)
    hist_t   = deque(maxlen=HIST)
    hist_h   = deque(maxlen=HIST)
    hist_rew = deque(maxlen=HIST)

    particles: list[_P] = []

    def new_episode():
        env = TomatoRipeningEnv(config=config)
        obs, _ = env.reset()
        hist_x.clear(); hist_t.clear(); hist_h.clear(); hist_rew.clear()
        particles.clear()
        world = b2World(gravity=(0, -1))
        body  = world.CreateDynamicBody(position=(5.0, 10.0))
        body.CreateCircleFixture(radius=1.5, density=1.0, restitution=0.4)
        return env, obs, world, body, 0.0

    env, obs, world, tomato_body, cum_rew = new_episode()
    done   = False
    action = 0
    x_v = t_v = hum_v = 0.0
    day_v = hod = 0.0
    last_info = {}

    active = True
    while active:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                active = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    active = False
                elif event.key == pygame.K_r:
                    env, obs, world, tomato_body, cum_rew = new_episode()
                    done = False; action = 0
                    x_v = t_v = hum_v = day_v = hod = 0.0
                elif event.key == pygame.K_p:
                    fname = f"edge_rl_{pygame.time.get_ticks()}.png"
                    pygame.image.save(screen, fname)
                    print(f"Saved: {fname}")
                elif event.key == pygame.K_SPACE:
                    # toggle pause flag stored in closure
                    main._paused = not getattr(main, "_paused", False)

        paused = getattr(main, "_paused", False)

        # Simulation step
        if not done and not paused:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, term, trunc, last_info = env.step(action)
            done = term or trunc
            cum_rew += float(reward)

            s    = env.simulator.get_state()
            x_v  = float(s["_true_ripeness"])
            t_v  = float(s["_true_temperature"])
            hum_v= float(s["_true_humidity"])
            day_v= float(s["days_elapsed"])
            hod  = float(s.get("hour_of_day", s["hours_elapsed"] % 24.0))

            hist_x.append(x_v)
            hist_t.append(t_v)
            hist_h.append(hum_v)
            hist_rew.append(cum_rew)

            tcx = PANEL_L + CENTRE_W // 2
            tcy = H // 2 - 20
            if action == 1:  _heat(particles, tcx, tcy + 85)
            elif action == 2: _cool(particles, tcx, tcy)

            world.Step(1.0/FPS, 6, 2)

        for p in particles:
            p.step()
        particles[:] = [p for p in particles if p.alive()]

        # ── Draw ─────────────────────────────────────────────────────
        screen.fill(C_BG)

        tcx = PANEL_L + CENTRE_W // 2
        tcy = H // 2 - 20

        # Sky
        draw_sky(screen, PANEL_L, 0, CENTRE_W, H - 80, hod)
        pygame.draw.rect(screen, (22, 18, 10), (PANEL_L, H-80, CENTRE_W, 80))
        pygame.draw.rect(screen, (35, 28, 15), (PANEL_L, H-80, CENTRE_W,  4))

        # Vine + leaves
        pygame.draw.line(screen, (35,70,28), (tcx, H-80), (tcx, tcy-85), 4)
        for i in range(4):
            ly  = tcy - 65 + i*30
            sd  = 1 if i%2==0 else -1
            pygame.draw.polygon(screen, (40,90,32),
                [(tcx,ly),(tcx+sd*28,ly-8),(tcx+sd*35,ly),(tcx+sd*28,ly+8)])

        # Tomato body
        t_col  = royg_color(x_v)
        t_dark = tuple(max(0, c-50) for c in t_col)
        t_rad  = 72 + int((1.0-x_v)*10)
        bob    = int(math.sin(pygame.time.get_ticks()*0.002)*3)

        if action == 1 and not done:
            ar = t_rad+30+int(math.sin(pygame.time.get_ticks()*0.01)*8)
            pygame.draw.circle(screen, (80,20,10), (tcx, tcy+bob), ar)
        elif action == 2 and not done:
            ar = t_rad+25+int(math.sin(pygame.time.get_ticks()*0.008)*6)
            pygame.draw.circle(screen, (10,40,80), (tcx, tcy+bob), ar)

        pygame.draw.circle(screen, t_dark, (tcx, tcy+bob), t_rad+4)
        pygame.draw.circle(screen, t_col,  (tcx, tcy+bob), t_rad)

        # Highlight
        hl = pygame.Surface((t_rad*2, t_rad*2), pygame.SRCALPHA)
        pygame.draw.circle(hl, (255,255,255,50),
                           (int(t_rad*0.65), int(t_rad*0.6)), int(t_rad*0.38))
        screen.blit(hl, (tcx-t_rad, tcy+bob-t_rad))

        # Calyx
        sy = tcy + bob - t_rad
        pygame.draw.line(screen, (50,110,40), (tcx,sy), (tcx,sy-28), 3)
        for i in range(5):
            a = i/5*math.pi*2 - math.pi/2
            pygame.draw.ellipse(screen, (55,120,45),
                (tcx+int(math.cos(a)*10)-5, sy+int(math.sin(a)*6)-3, 12, 6))

        # X + USDA under tomato
        xl = flg.render(f"X = {x_v:.3f}", True, C_WHITE)
        screen.blit(xl, xl.get_rect(center=(tcx, tcy+bob+t_rad+22)))
        sn, sc = usda_stage(x_v)
        sl = fttl.render(sn, True, sc)
        screen.blit(sl, sl.get_rect(center=(tcx, tcy+bob+t_rad+44)))

        # Action label above tomato
        if not done:
            al, ac = ACTION_INFO[action]
            albl = fm.render(al, True, ac)
            screen.blit(albl, albl.get_rect(center=(tcx, tcy+bob-t_rad-26)))

        # Particles
        for p in particles:
            ps = femo.render(p.char, True, p.color)
            ps.set_alpha(p.alpha())
            screen.blit(ps, (int(p.x), int(p.y)))

        # Episode-done overlay
        if done:
            ov = pygame.Surface((CENTRE_W, H), pygame.SRCALPHA)
            ov.fill((0,0,0,145))
            screen.blit(ov, (PANEL_L, 0))
            el = flg.render("✅ EPISODE COMPLETE — Press R to restart", True, C_GREEN)
            screen.blit(el, el.get_rect(center=(PANEL_L+CENTRE_W//2, H//2)))
            if last_info.get("auto_harvest"):
                qs = (f"Quality: {last_info.get('harvest_quality',0):.3f}  |  "
                      f"Timing err: {last_info.get('timing_error',0):.2f} d")
                ql = fttl.render(qs, True, C_YELLOW)
                screen.blit(ql, ql.get_rect(center=(PANEL_L+CENTRE_W//2, H//2+40)))

        # Thermal warning
        if hist_t and (hist_t[-1] > 35.0 or hist_t[-1] < 12.5):
            pygame.draw.rect(screen, (120,0,0), (PANEL_L, H-80, CENTRE_W, 30), border_radius=4)
            w2 = fs.render("⚠  THERMAL SAFETY GUARDRAIL ACTIVE", True, C_WHITE)
            screen.blit(w2, w2.get_rect(center=(PANEL_L+CENTRE_W//2, H-65)))

        # Paused banner
        if paused:
            pb = fm.render("⏸  PAUSED  (SPACE to resume)", True, C_YELLOW)
            screen.blit(pb, pb.get_rect(center=(PANEL_L+CENTRE_W//2, 30)))

        # ── LEFT PANEL ────────────────────────────────────────────────
        pygame.draw.rect(screen, C_PANEL, (0, 0, PANEL_L, H))
        pygame.draw.line(screen, C_BORDER, (PANEL_L,0),(PANEL_L,H),1)

        ttl = fm.render("Edge-RL  Digital Twin", True, C_ACCENT)
        screen.blit(ttl, (16, 14))
        pygame.draw.line(screen, C_BORDER, (10,38),(PANEL_L-10,38),1)

        target_d = getattr(env, "target_day", 7.0)
        hod_str  = f"{int(hod):02d}:00  {'☀' if 6<=hod<=19 else '🌙'}"
        metrics = [
            ("USDA Stage",    sn,                     sc),
            ("Chromatic X",   f"{x_v:.4f}",           C_GREEN if x_v>0.15 else C_RED),
            ("Temperature",   f"{t_v:.2f} °C",        C_RED if t_v>30 else C_ACCENT),
            ("Humidity",      f"{hum_v:.1f} %",       C_BLUE),
            ("Day",           f"{day_v:.3f} / {target_d:.1f}", C_WHITE),
            ("Hour of Day",   hod_str,                C_YELLOW),
            ("Cumul. Reward", f"{cum_rew:+.2f}",      C_GREEN if cum_rew>=0 else C_RED),
            ("Action",        ACTION_INFO[action][0], ACTION_INFO[action][1]),
        ]
        for i, (label, val, col) in enumerate(metrics):
            my = 52 + i*88
            pygame.draw.rect(screen, (24,27,44), (10, my, PANEL_L-20, 78), border_radius=6)
            draw_metric(screen, fs, fm, label, val, 20, my+8, col)

        # ── RIGHT PANEL ───────────────────────────────────────────────
        rx = W - PANEL_R
        pygame.draw.rect(screen, C_PANEL, (rx, 0, PANEL_R, H))
        pygame.draw.line(screen, C_BORDER, (rx,0),(rx,H),1)

        ch_h = (H-20)//4 - 8
        charts = [
            (hist_x,   C_GREEN,  "Chromatic X",   0.0,  1.0,  "",   0.15, C_RED),
            (hist_t,   C_RED,    "Temperature",  12.5, 40.0, "°C", 35.0, (200,60,60)),
            (hist_h,   C_BLUE,   "Humidity",     40.0, 99.0, "%",  None, None),
            (hist_rew, C_ACCENT, "Cumul. Reward",-50., 30.0, "",    0.0, C_DIM),
        ]
        for i, (data,col,title,ylo,yhi,unit,thr,tc) in enumerate(charts):
            r = (rx+6, 8+i*(ch_h+10), PANEL_R-12, ch_h)
            draw_chart(screen, r, data, col, title, ylo, yhi, unit,
                       threshold=thr, t_col=tc or C_DIM)

        # Hints
        hint = fmono.render("P=screenshot  R=restart  SPACE=pause  Q=quit", True, C_DIM)
        screen.blit(hint, hint.get_rect(center=(PANEL_L+CENTRE_W//2, H-12)))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main._paused = False
    main()