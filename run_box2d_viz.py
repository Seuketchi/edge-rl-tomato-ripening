#!/usr/bin/env python3
import math, sys, pygame, numpy as np, yaml
from Box2D import b2World, b2CircleShape
from stable_baselines3 import DQN
from ml_training.rl.environment import TomatoRipeningEnv

# ──────────────────────── THESIS RESEARCH THEME ────────────────────────
SCREEN_W, SCREEN_H = 1350, 850
FPS = 60
HISTORY_LIMIT = 150
T_BASE = 12.5  # [cite: 206]
T_MAX = 35.0   # [cite: 251]

# Professional Dashboard Colors
CLR_BG = (10, 12, 18)
CLR_PANEL = (28, 30, 42)
CLR_ACCENT = (0, 180, 255) # Edge-RL Cyan [cite: 11]
CLR_GREEN = (46, 204, 113) # Success [cite: 396]
CLR_RED = (231, 76, 60)    # Temperature [cite: 13]

def get_usda_stage(x):
    """Categorizes X into USDA standard ripening stages."""
    if x > 0.90: return "MATURE GREEN"
    if x > 0.80: return "BREAKER"
    if x > 0.60: return "TURNING"
    if x > 0.40: return "PINK"
    if x > 0.15: return "LIGHT RED"
    return "RED (HARVEST)" # [cite: 198]

def draw_scientific_plot(screen, rect, data, color, title, units=""):
    x, y, w, h = rect
    pygame.draw.rect(screen, (15, 15, 22), rect, border_radius=5)
    if len(data) < 2: return
    d_min, d_max = min(data), max(data)
    rng = max(0.00001, d_max - d_min)
    pts = [(x + (i/HISTORY_LIMIT)*w, y + h - ((v-d_min)/rng)*h) for i,v in enumerate(data)]
    pygame.draw.lines(screen, color, False, pts, 2)
    f = pygame.font.SysFont("Arial", 14, bold=True)
    screen.blit(f.render(f"{title}: {data[-1]:.4f}{units}", True, (200, 200, 200)), (x, y - 20))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Thesis: Edge-RL Autonomous Ripening Pipeline")
    clock = pygame.time.Clock()
    
    f_huge = pygame.font.SysFont("Verdana", 90, bold=True)
    f_med = pygame.font.SysFont("Verdana", 22, bold=True)
    f_sm = pygame.font.SysFont("Arial", 14)

    try:
        with open("ml_training/config.yaml") as f: config = yaml.safe_load(f)
        model = DQN.load("outputs/rl_20260217_095300/best_model/best_model.zip")
        env = TomatoRipeningEnv(config=config)
    except Exception as e: print(f"Init Error: {e}"); return

    active = True
    while active:
        obs, _ = env.reset()
        done, hist = False, {"x":[], "t":[], "r":[], "d":[]}
        world = b2World(gravity=(0,0)) # [cite: 209]
        tomato = world.CreateDynamicBody(position=(21, 13))
        tomato.CreateCircleFixture(radius=4.5, density=1.0)

        while not done and active:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: active = False
                if e.type == pygame.KEYDOWN and e.key == pygame.K_p: # Save for thesis paper
                    pygame.image.save(screen, f"edge_rl_result_{pygame.time.get_ticks()}.png")
            
            action, _ = model.predict(obs, deterministic=True) # [cite: 92]
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            s = env.simulator.get_state()
            day, x_v, t_v, r_v = s.get("day",0), s.get("_true_ripeness",0), s.get("_true_temperature",0), s.get("cumulative_reward",0)
            target_d = getattr(env, 'target_day', 7.0)

            for k,v in zip(["x","t","r","d"], [x_v, t_v, r_v, day]):
                hist[k].append(v)
                if len(hist[k]) > HISTORY_LIMIT: hist[k].pop(0)

            world.Step(1.0/FPS, 6, 2)
            screen.fill(CLR_BG)

            # 1. VISUAL PERCEPTION (IPO Model) [cite: 66, 68]
            cx, cy = int(21 * 30), int(850 - 13 * 30)
            t_col = (255, 50, 50) if x_v < 0.2 else (50, 255, 100) # ROYG Mapping [cite: 12]
            
            # Action Glow (Explainable AI)
            if action == 1: pygame.draw.circle(screen, (100, 30, 30), (cx, cy), 160) # Heating
            elif action == 2: pygame.draw.circle(screen, (30, 60, 100), (cx, cy), 160) # Cooling

            pygame.draw.circle(screen, (220, 225, 240), (cx, cy), 140, 2)
            pygame.draw.circle(screen, t_col, (cx, cy), 135)

            # 2. DATA PANEL
            pygame.draw.rect(screen, CLR_PANEL, (0, 0, 340, SCREEN_H))
            screen.blit(f_med.render("Edge-RL Core", True, CLR_ACCENT), (30, 40))
            
            # Metrics from Table III [cite: 394]
            metrics = [
                ("USDA Ripening Stage", get_usda_stage(x_v), CLR_ACCENT),
                ("Precision Timeline", f"Day {day:.5f}", (255, 255, 255)),
                ("Temperature (Sensor)", f"{t_v:.2f} C", CLR_RED),
                ("Cumulative Reward", f"{r_v:.2f}", CLR_GREEN)
            ]
            for i, (l, v, c) in enumerate(metrics):
                y = 120 + i*110
                screen.blit(f_sm.render(l, True, (150, 160, 180)), (30, y))
                screen.blit(f_med.render(v, True, c), (30, y + 25))

            # 3. SAFETY OVERRIDE 
            if t_v > T_MAX or t_v < T_BASE:
                pygame.draw.rect(screen, (100, 0, 0), (30, 600, 280, 50), border_radius=8)
                screen.blit(f_sm.render("HARDWARE SAFETY OVERRIDE ACTIVE", True, (255, 255, 255)), (45, 615))

            # 4. ANALYTICAL GRAPHS [cite: 411, 413]
            gx, gw, gh = 960, 360, 160
            draw_scientific_plot(screen, (gx, 60, gw, gh), hist["d"], CLR_ACCENT, "TIME", "d")
            draw_scientific_plot(screen, (gx, 250, gw, gh), hist["t"], CLR_RED, "TEMP", "C")
            draw_scientific_plot(screen, (gx, 440, gw, gh), hist["x"], CLR_ACCENT, "INDEX X")
            draw_scientific_plot(screen, (gx, 630, gw, gh), hist["r"], CLR_GREEN, "REWARD")

            pygame.display.flip()
            clock.tick(FPS)

        # REPEAT LOGIC
        while done and active:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r: done = False
                if event.type == pygame.QUIT: active = False
            pygame.display.flip()

    pygame.quit()

if __name__ == "__main__": main()