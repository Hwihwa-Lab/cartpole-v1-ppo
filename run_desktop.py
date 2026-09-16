# -*- coding: utf-8 -*-
"""
HWIHWA LAB // CartPole-v1 Local Desktop Cockpit
Real-time 60FPS Pygame Simulation with Manual Teleop & PPO Autonomous Mode
"""

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym

# Optional PyTorch & SB3 for AI control
try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

import pygame

# Initialize Pygame
pygame.init()
pygame.font.init()

# Color Palette (Dark Cockpit Theme)
BG_COLOR = (13, 17, 23)        # Slate Dark
CART_COLOR = (56, 189, 248)     # Sky Blue
POLE_COLOR = (244, 63, 94)      # Crimson Coral
TRACK_COLOR = (51, 65, 85)      # Slate Border
TEXT_PRIMARY = (248, 250, 252)  # Bright White
TEXT_MUTED = (148, 163, 184)    # Slate Muted
AI_BADGE = (34, 197, 94)        # Neon Green
HUMAN_BADGE = (234, 179, 8)     # Amber Yellow
PANEL_BG = (22, 27, 34)         # Card Panel

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 500

def create_hud_font():
    try:
        return pygame.font.SysFont("Consolas", 14), pygame.font.SysFont("Trebuchet MS", 18, bold=True)
    except Exception:
        return pygame.font.Font(None, 18), pygame.font.Font(None, 24)

def run_cockpit(model_path: str = "models/cartpole_ppo.zip"):
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("🕹️ Hwihwa Lab // CartPole-v1 Mission Cockpit")
    clock = pygame.time.Clock()
    font_mono, font_title = create_hud_font()

    env = gym.make("CartPole-v1")
    obs, _ = env.reset()

    # Try loading PPO model
    ai_model = None
    if HAS_SB3 and os.path.exists(model_path):
        try:
            ai_model = PPO.load(model_path)
            print(f"[✓] Loaded trained PPO brain from: {model_path}")
        except Exception as e:
            print(f"[!] Could not load model: {e}")

    # Mode: 'AI' or 'HUMAN'
    mode = 'AI' if ai_model is not None else 'HUMAN'
    paused = False
    
    score = 0
    max_score = 0
    running = True
    manual_action = 0

    print("\n" + "="*50)
    print("🚀 CartPole Cockpit Controls:")
    print("  [M]     : Toggle Mode (AI Neural vs Human Teleop)")
    print("  [LEFT]  : Push Cart Left  (Human Mode)")
    print("  [RIGHT] : Push Cart Right (Human Mode)")
    print("  [SPACE] : Pause / Resume")
    print("  [R]     : Manual Reset")
    print("  [ESC]   : Exit")
    print("="*50 + "\n")

    while running:
        dt = clock.tick(60) / 1000.0  # Cap at 60 FPS

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_m:
                    if ai_model is not None:
                        mode = 'HUMAN' if mode == 'AI' else 'AI'
                    else:
                        print("[!] AI Model file not found. Train first with train.py!")
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    score = 0

        # Continuous key press for human mode
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            manual_action = 0
        elif keys[pygame.K_RIGHT]:
            manual_action = 1

        # Simulation Step
        if not paused:
            if mode == 'AI' and ai_model is not None:
                action, _ = ai_model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = manual_action

            obs, reward, terminated, truncated, _ = env.step(action)
            score += 1
            if score > max_score:
                max_score = score

            if terminated or truncated:
                obs, _ = env.reset()
                score = 0

        # --- Rendering ---
        screen.fill(BG_COLOR)

        # CartPole Physical Dimensions
        x_pos, x_vel, theta, theta_vel = obs
        # Cart track: x is roughly [-2.4, 2.4]
        world_width = 4.8
        scale = WINDOW_WIDTH / world_width
        cart_y = 350
        cart_w = 60
        cart_h = 30
        pole_len = 120

        cart_x = int(x_pos * scale + WINDOW_WIDTH / 2)

        # Draw Track Line
        pygame.draw.line(screen, TRACK_COLOR, (40, cart_y + cart_h // 2 + 5), (WINDOW_WIDTH - 40, cart_y + cart_h // 2 + 5), 4)

        # Draw Cart (Rounded rectangle look)
        cart_rect = pygame.Rect(cart_x - cart_w // 2, cart_y, cart_w, cart_h)
        pygame.draw.rect(screen, CART_COLOR, cart_rect, border_radius=6)
        pygame.draw.rect(screen, (255, 255, 255), cart_rect, width=2, border_radius=6)

        # Draw Wheels
        wheel_radius = 6
        pygame.draw.circle(screen, (200, 220, 240), (cart_x - 18, cart_y + cart_h + 3), wheel_radius)
        pygame.draw.circle(screen, (200, 220, 240), (cart_x + 18, cart_y + cart_h + 3), wheel_radius)

        # Draw Pole (rotated by theta)
        # Note: In standard CartPole, theta=0 is upright. In Pygame y is downward.
        # pole tip: x = cart_x + pole_len * sin(theta), y = cart_y - pole_len * cos(theta)
        tip_x = int(cart_x + pole_len * np.sin(theta))
        tip_y = int(cart_y - pole_len * np.cos(theta))

        pygame.draw.line(screen, POLE_COLOR, (cart_x, cart_y), (tip_x, tip_y), 8)
        pygame.draw.circle(screen, (255, 255, 255), (cart_x, cart_y), 6)   # Joint
        pygame.draw.circle(screen, (255, 100, 120), (tip_x, tip_y), 7)    # Tip marker

        # --- HUD Overhead Display ---
        # Top Panel
        panel_rect = pygame.Rect(20, 20, WINDOW_WIDTH - 40, 75)
        pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=10)
        pygame.draw.rect(screen, TRACK_COLOR, panel_rect, width=1, border_radius=10)

        # Title & Badge
        title_surf = font_title.render("CARTPOLE-V1 MISSION COCKPIT", True, TEXT_PRIMARY)
        screen.blit(title_surf, (35, 30))

        # Mode Badge
        badge_color = AI_BADGE if mode == 'AI' else HUMAN_BADGE
        badge_text = "● AI AUTONOMOUS [PPO]" if mode == 'AI' else "● MANUAL TELEOP [KEYS]"
        badge_surf = font_title.render(badge_text, True, badge_color)
        screen.blit(badge_surf, (WINDOW_WIDTH - 320, 30))

        # Subtitle Controls
        help_text = "[M] Switch Mode  |  [←/→] Push Cart  |  [R] Reset  |  [Space] Pause"
        screen.blit(font_mono.render(help_text, True, TEXT_MUTED), (35, 62))

        # Bottom Telemetry Panel
        bot_panel = pygame.Rect(20, WINDOW_HEIGHT - 75, WINDOW_WIDTH - 40, 55)
        pygame.draw.rect(screen, PANEL_BG, bot_panel, border_radius=8)
        pygame.draw.rect(screen, TRACK_COLOR, bot_panel, width=1, border_radius=8)

        # Telemetry info
        deg = np.degrees(theta)
        telemetry_str = f"SCORE: {score:03d} / 500  |  MAX: {max_score:03d}  |  POS: {x_pos:+.2f}m  |  ANGLE: {deg:+.1f}°  |  VEL: {x_vel:+.2f}m/s"
        screen.blit(font_mono.render(telemetry_str, True, (56, 189, 248)), (35, WINDOW_HEIGHT - 58))

        pygame.display.flip()

    env.close()
    pygame.quit()
    print("[*] Cockpit session terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole Real-time Pygame Cockpit")
    parser.add_argument("--model", type=str, default="models/cartpole_ppo.zip", help="Path to PPO model")
    args = parser.parse_args()
    run_cockpit(model_path=args.model)
