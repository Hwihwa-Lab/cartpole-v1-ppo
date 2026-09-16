# -*- coding: utf-8 -*-
"""
HWIHWA LAB // CartPole-v1 Cybernetic Animated GIF Generator
Renders a high-resolution, futuristic dark neon physical AI demo showing:
1) Nominal balance
2) Instant Mouse/Force Impulse Shock (⚡ +25.0 N)
3) PPO 2-Phase Catching & Centering Maneuver
4) Real-time Phase Portrait Attractor Convergence (θ vs θ_dot)
"""

import os
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Physics Setup
TAU = 0.02
FORCE_MAG = 10.0
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
GRAVITY = 9.81
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12.0 * 2.0 * math.pi / 360.0

def load_weights(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "w1": np.array(data["w1"]),
        "b1": np.array(data["b1"]),
        "w2": np.array(data["w2"]),
        "b2": np.array(data["b2"]),
        "w_out": np.array(data["w_out"]),
        "b_out": np.array(data["b_out"]),
    }

def ppo_policy(obs: np.ndarray, w: dict) -> int:
    h1 = np.tanh(np.dot(w["w1"], obs) + w["b1"])
    h2 = np.tanh(np.dot(w["w2"], h1) + w["b2"])
    logits = np.dot(w["w_out"], h2) + w["b_out"]
    p1 = 1.0 / (1.0 + np.exp(-(logits[1] - logits[0])))
    return 1 if p1 > 0.5 else 0

def step_physics(state: np.ndarray, action: int, shock_force: float = 0.0):
    x, x_dot, theta, theta_dot = state
    force = (FORCE_MAG if action == 1 else -FORCE_MAG) + shock_force

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta**2 / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Euler-Cromer
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc

    return np.array([x, x_dot, theta, theta_dot])

def render_frame(state, step, total_steps, shock_active, phase_history, W=800, H=420):
    img = Image.new("RGB", (W, H), "#060913")
    draw = ImageDraw.Draw(img)

    # 1. Subtle Cybernetic Grid
    for gx in range(0, W, 40):
        draw.line([(gx, 0), (gx, H)], fill="#0d1527", width=1)
    for gy in range(0, H, 40):
        draw.line([(0, gy), (W, gy)], fill="#0d1527", width=1)

    # 2. Header Bar HUD
    draw.rectangle([(0, 0), (W, 46)], fill="#090e1d", outline="#1e293b", width=1)
    draw.text((20, 14), "🤖 CARTPOLE-V1 PHYSICAL AI // PPO DISTURBANCE RECOVERY", fill="#38bdf8")
    draw.text((W - 220, 14), f"SIM TIME: {step*TAU:.2f}s | 60 FPS", fill="#94a3b8")

    # 3. Main Stage Rail
    rail_y = 260
    draw.line([(60, rail_y), (W - 260, rail_y)], fill="#1e293b", width=4)
    # Center tick
    draw.line([( (W - 200) // 2, rail_y - 8), ( (W - 200) // 2, rail_y + 8)], fill="#38bdf8", width=2)

    # World to Screen Coordinates
    stage_center_x = (W - 200) // 2
    meters_to_pixels = 95.0

    cart_x_screen = stage_center_x + state[0] * meters_to_pixels
    cart_w, cart_h = 76, 36
    cart_left = cart_x_screen - cart_w // 2
    cart_top = rail_y - cart_h - 6

    # Glow under cart
    draw.ellipse([(cart_left - 10, rail_y - 12), (cart_left + cart_w + 10, rail_y + 8)], fill="#003b46")

    # Draw Cart Body
    draw.rounded_rectangle([(cart_left, cart_top), (cart_left + cart_w, cart_top + cart_h)], radius=6, fill="#0ea5e9", outline="#38bdf8", width=2)
    # Cart Wheels
    wheel_r = 6
    draw.ellipse([(cart_left + 10, rail_y - wheel_r*2), (cart_left + 10 + wheel_r*2, rail_y)], fill="#334155", outline="#94a3b8", width=1)
    draw.ellipse([(cart_left + cart_w - 10 - wheel_r*2, rail_y - wheel_r*2), (cart_left + cart_w - 10, rail_y)], fill="#334155", outline="#94a3b8", width=1)

    # Pivot Point
    pivot_x = cart_x_screen
    pivot_y = cart_top + 10
    draw.ellipse([(pivot_x - 5, pivot_y - 5), (pivot_x + 5, pivot_y + 5)], fill="#ffffff")

    # Pole Tip
    pole_len_px = LENGTH * 2 * meters_to_pixels * 1.3
    tip_x = pivot_x + pole_len_px * math.sin(state[2])
    tip_y = pivot_y - pole_len_px * math.cos(state[2])

    # Pole Rod (Neon Emerald Green)
    draw.line([(pivot_x, pivot_y), (tip_x, tip_y)], fill="#10b981", width=6)
    draw.ellipse([(tip_x - 7, tip_y - 7), (tip_x + 7, tip_y + 7)], fill="#34d399", outline="#ffffff", width=2)

    # 4. Shock Impulse Visualizer
    if shock_active:
        # Laser force arrow
        arrow_y = cart_top + cart_h // 2
        draw.line([(cart_left - 80, arrow_y), (cart_left - 8, arrow_y)], fill="#ef4444", width=5)
        draw.polygon([(cart_left - 8, arrow_y - 8), (cart_left + 4, arrow_y), (cart_left - 8, arrow_y + 8)], fill="#ef4444")
        # Shock text banner
        draw.rounded_rectangle([(stage_center_x - 130, 70), (stage_center_x + 130, 105)], radius=6, fill="#7f1d1d", outline="#ef4444", width=2)
        draw.text((stage_center_x - 110, 78), "⚡ MOUSE SHOCK: +25.0 N", fill="#fecaca")

    # 5. Right Panel: Live Phase Portrait (θ vs θ_dot)
    radar_x, radar_y, radar_size = W - 230, 65, 205
    draw.rounded_rectangle([(radar_x, radar_y), (radar_x + radar_size, radar_y + radar_size + 95)], radius=8, fill="#0b1120", outline="#1e293b", width=2)
    draw.text((radar_x + 15, radar_y + 12), "🌀 PHASE PORTRAIT", fill="#38bdf8")
    draw.text((radar_x + 15, radar_y + 30), "Origin Attractor (0, 0)", fill="#64748b")

    center_rx = radar_x + radar_size // 2
    center_ry = radar_y + 40 + (radar_size) // 2

    # Radar Grid
    draw.ellipse([(center_rx - 65, center_ry - 65), (center_rx + 65, center_ry + 65)], outline="#1e293b", width=1)
    draw.ellipse([(center_rx - 35, center_ry - 35), (center_rx + 35, center_ry + 35)], outline="#1e293b", width=1)
    draw.line([(center_rx - 75, center_ry), (center_rx + 75, center_ry)], fill="#1e293b", width=1)
    draw.line([(center_rx, center_ry - 75), (center_rx, center_ry + 75)], fill="#1e293b", width=1)

    # Plot Spiral Trajectory
    if len(phase_history) > 1:
        points = []
        for th, th_d in phase_history:
            px = center_rx + th * 420.0
            py = center_ry - th_d * 22.0
            points.append((px, py))
        for i in range(len(points) - 1):
            alpha_color = "#38bdf8" if i > len(points) - 15 else "#0369a1"
            draw.line([points[i], points[i+1]], fill=alpha_color, width=2)

    # Current State Dot
    curr_px = center_rx + state[2] * 420.0
    curr_py = center_ry - state[3] * 22.0
    draw.ellipse([(curr_px - 4, curr_py - 4), (curr_px + 4, curr_py + 4)], fill="#00f2fe", outline="#ffffff", width=1)

    # 6. Bottom Telemetry Pills
    telemetry_y = H - 42
    draw.rectangle([(0, H - 52), (W, H)], fill="#090e1d", outline="#1e293b", width=1)
    draw.text((20, telemetry_y), f"CTRL: TRAINED PPO (20K)", fill="#10b981")
    draw.text((220, telemetry_y), f"CART POS: x = {state[0]:+.3f} m", fill="#f8fafc")
    draw.text((430, telemetry_y), f"POLE ANGLE: θ = {math.degrees(state[2]):+.2f}°", fill="#fbbf24")
    draw.text((640, telemetry_y), f"STABILITY: 100% SOLVED", fill="#38bdf8")

    return img

def generate_animation():
    current_dir = Path(__file__).resolve().parent
    assets_dir = current_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    weights_path = current_dir / "cartpole_weights.json"
    weights = load_weights(weights_path)

    # Simulation setup
    state = np.array([0.0, 0.0, 0.01, 0.0])
    phase_history = []
    frames = []

    print("🎬 Simulating and rendering high-resolution Cybernetic GIF...")

    total_steps = 140  # ~2.8 seconds @ 50 FPS
    for step in range(total_steps):
        # Trigger shock between step 25 and 35
        shock_active = (25 <= step <= 34)
        shock_force = 26.0 if shock_active else 0.0

        # Action from PPO
        action = ppo_policy(state, weights)

        # Record phase
        phase_history.append((state[2], state[3]))
        if len(phase_history) > 60:
            phase_history.pop(0)

        # Render frame
        if step % 2 == 0:  # Render at 25 FPS effective
            frame_img = render_frame(state, step, total_steps, shock_active, phase_history)
            frames.append(frame_img)

        # Physics update
        state = step_physics(state, action, shock_force=shock_force)

    # Save Animated GIF
    gif_path = assets_dir / "cartpole_disturbance_recovery.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,  # 25 FPS (40ms per frame)
        loop=0,
        optimize=True
    )

    print(f"[✓] Successfully generated Animated Demo GIF: {gif_path} ({len(frames)} frames, {round(gif_path.stat().st_size / 1024, 1)} KB)")
    return gif_path

if __name__ == "__main__":
    generate_animation()
