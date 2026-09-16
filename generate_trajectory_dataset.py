# -*- coding: utf-8 -*-
"""
HWIHWA LAB // CartPole-v1 Planetary Physical AI Trajectory Dataset Generator
Generates high-frequency (50Hz / dt=0.02s) time-series state-action trajectories
across 4 planetary gravity regimes, 3 disturbance modes, and 3 controllers.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Simulation Constants
TAU = 0.02
MAX_STEPS = 500
X_THRESHOLD = 2.4
THETA_THRESHOLD = 12.0 * 2.0 * np.pi / 360.0  # 12 degrees
FORCE_MAG = 10.0
MASSCART = 1.0
MASSPOLE = 0.1
LENGTH = 0.5
TOTAL_MASS = MASSCART + MASSPOLE
POLEMASS_LENGTH = MASSPOLE * LENGTH

PLANETS = {
    "Moon": 1.62,
    "Mars": 3.72,
    "Earth": 9.81,
    "Jupiter": 24.79
}

DISTURBANCE_MODES = {
    "Clean": {"wind": 0.0, "noise": 0.0},
    "Wind": {"wind": 2.2, "noise": 0.0},
    "Noise": {"wind": 0.0, "noise": 0.05}
}

CONTROLLERS = ["Trained_PPO", "Optimal_LQR", "Undercooked_PPO"]
LQR_GAINS = np.array([-1.0, -1.8, -35.0, -7.5])

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

def ppo_policy(obs: np.ndarray, w: Dict[str, np.ndarray]) -> int:
    h1 = np.tanh(np.dot(w["w1"], obs) + w["b1"])
    h2 = np.tanh(np.dot(w["w2"], h1) + w["b2"])
    logits = np.dot(w["w_out"], h2) + w["b_out"]
    p1 = 1.0 / (1.0 + np.exp(-(logits[1] - logits[0])))
    return 1 if p1 > 0.5 else 0

def lqr_policy(obs: np.ndarray) -> int:
    u = np.dot(LQR_GAINS, obs)
    return 1 if u > 0 else 0

def step_physics(state: np.ndarray, action: int, gravity: float, wind_force: float = 0.0):
    x, x_dot, theta, theta_dot = state
    force = (FORCE_MAG if action == 1 else -FORCE_MAG) + wind_force

    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    thetaacc = (gravity * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta**2 / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Euler-Cromer integration
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc

    terminated = bool(
        x < -X_THRESHOLD or x > X_THRESHOLD or
        theta < -THETA_THRESHOLD or theta > THETA_THRESHOLD
    )
    return np.array([x, x_dot, theta, theta_dot]), terminated

def generate_dataset(output_dir: Path, episodes_per_combo: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = Path(__file__).resolve().parent / "cartpole_weights.json"
    weights = load_weights(weights_path)

    all_records = []
    episode_global_id = 0

    print("🚀 Generating High-Frequency Trajectory Dataset...")
    print(f"Configurations: 4 Planets x 3 Disturbance Modes x 3 Controllers x {episodes_per_combo} Episodes")

    for planet_name, gravity in PLANETS.items():
        for dist_name, dist_params in DISTURBANCE_MODES.items():
            for ctrl_name in CONTROLLERS:
                for ep in range(episodes_per_combo):
                    episode_global_id += 1
                    # Random initial state
                    state = np.random.uniform(low=-0.05, high=0.05, size=(4,))

                    for step in range(MAX_STEPS):
                        time_sec = round(step * TAU, 3)

                        # Observation with sensor noise
                        if dist_params["noise"] > 0:
                            obs = state + np.random.normal(0, dist_params["noise"], size=(4,))
                        else:
                            obs = state.copy()

                        # Action selection
                        if ctrl_name == "Trained_PPO":
                            action = ppo_policy(obs, weights)
                        elif ctrl_name == "Optimal_LQR":
                            action = lqr_policy(obs)
                        else:  # Undercooked PPO (50% noise decision)
                            if np.random.random() < 0.5:
                                action = ppo_policy(obs, weights)
                            else:
                                action = np.random.choice([0, 1])

                        # Physics step
                        next_state, terminated = step_physics(state, action, gravity, dist_params["wind"])
                        truncated = (step == MAX_STEPS - 1 and not terminated)
                        reward = 1.0 if not terminated else 0.0

                        record = {
                            "episode_id": episode_global_id,
                            "step": step,
                            "time_sec": time_sec,
                            "planet": planet_name,
                            "gravity": gravity,
                            "disturbance_mode": dist_name,
                            "wind_force_N": dist_params["wind"],
                            "sensor_noise_std": dist_params["noise"],
                            "controller": ctrl_name,
                            "cart_pos_x": round(float(state[0]), 5),
                            "cart_vel_x_dot": round(float(state[1]), 5),
                            "pole_angle_rad": round(float(state[2]), 5),
                            "pole_angle_deg": round(float(np.degrees(state[2])), 3),
                            "pole_vel_theta_dot": round(float(state[3]), 5),
                            "action": int(action),
                            "reward": reward,
                            "terminated": terminated,
                            "truncated": truncated
                        }
                        all_records.append(record)

                        state = next_state
                        if terminated:
                            break

    # 1. Save JSONL
    jsonl_path = output_dir / "cartpole_planetary_trajectories.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"[✓] Saved JSONL Trajectory Dataset: {jsonl_path} ({len(all_records):,} rows)")

    # 2. Save CSV
    csv_path = output_dir / "cartpole_planetary_trajectories.csv"
    if all_records:
        keys = list(all_records[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in all_records:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[✓] Saved CSV Trajectory Dataset: {csv_path} ({len(all_records):,} rows)")

    # 3. Save Parquet if pandas/pyarrow available
    try:
        import pandas as pd
        df = pd.DataFrame(all_records)
        parquet_path = output_dir / "cartpole_planetary_trajectories.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"[✓] Saved Parquet Trajectory Dataset: {parquet_path}")
    except Exception as e:
        print(f"[!] Parquet export note: {e}. (JSONL & CSV created successfully).")

    # 4. Save Dataset Card README.md
    dataset_card_path = output_dir / "README.md"
    dataset_card_content = f"""---
license: mit
tags:
- reinforcement-learning
- physical-ai
- robotics
- trajectory-dataset
- sim-to-real
- cartpole-v1
- offline-rl
- imitation-learning
size_categories:
- 10K<n<100K
---

# 🪐 CartPole-v1 Planetary Physical AI Trajectory Dataset

**Private Foundation Dataset for Sim-to-Real Dynamics, Offline Reinforcement Learning, and Anomaly Detection.**

## 📊 Dataset Overview
- **Total State-Action Records**: {len(all_records):,} timesteps @ 50 Hz (dt=0.02s)
- **Total Episodes**: {episode_global_id} complete episodes
- **Gravitational Regimes**: Moon (1.62 m/s²), Mars (3.72 m/s²), Earth (9.81 m/s²), Jupiter (24.79 m/s²)
- **Disturbance Profiles**: Clean nominal, +2.2N Constant Lateral Wind, Gaussian Sensor Noise (σ=0.05)
- **Controllers Included**:
  1. `Trained_PPO`: Deep Neural Policy [Linear(4,64) ➔ Linear(64,64) ➔ Linear(64,2)]
  2. `Optimal_LQR`: Classical Optimal Linear Quadratic Regulator Gain Matrix u = -Kx
  3. `Undercooked_PPO`: Perturbed Non-convergent Stochastic Baseline

## 📂 Feature Schema
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `episode_id` | `int` | Unique episode sequence index |
| `step` | `int` | Simulation timestep (0 to 500) |
| `time_sec` | `float` | Elapsed continuous time (dt=0.02s) |
| `planet` | `string` | Celestial body identifier |
| `gravity` | `float` | Gravitational acceleration (m/s²) |
| `disturbance_mode` | `string` | Perturbation category (`Clean`, `Wind`, `Noise`) |
| `wind_force_N` | `float` | Applied lateral wind bias in Newtons |
| `sensor_noise_std` | `float` | Standard deviation of Gaussian observation noise |
| `controller` | `string` | Active policy architecture |
| `cart_pos_x` | `float` | Cart position on 1D track (-2.4m to +2.4m) |
| `cart_vel_x_dot` | `float` | Linear velocity of cart (m/s) |
| `pole_angle_rad` | `float` | Inverted pendulum angular deviation in radians |
| `pole_angle_deg` | `float` | Inverted pendulum angular deviation in degrees |
| `pole_vel_theta_dot` | `float` | Pendulum angular velocity (rad/s) |
| `action` | `int` | Discrete push force applied (0: Left, 1: Right) |
| `reward` | `float` | Per-step equilibrium reward (1.0 or 0.0) |
| `terminated` | `bool` | True if system collapsed beyond stability limits |
| `truncated` | `bool` | True if episode completed max duration (500 steps) |

## 🔒 Access & Governance
*Curated and maintained by **HWIHWA LAB**.*
"""
    with open(dataset_card_path, "w", encoding="utf-8") as f:
        f.write(dataset_card_content)
    print(f"[✓] Created Dataset Card: {dataset_card_path}")

    return len(all_records)

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    generate_dataset(data_dir, episodes_per_combo=10)
