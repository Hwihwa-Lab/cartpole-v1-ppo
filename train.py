# -*- coding: utf-8 -*-
"""
HWIHWA LAB // CartPole-v1 PPO Neural Balancer
Training & Evaluation Pipeline with JS Weight Exporter
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

def train_cartpole(timesteps: int = 25_000, seed: int = 42, model_dir: str = "models"):
    print(f"[*] Initializing CartPole-v1 Environment (Seed: {seed})...")
    env = gym.make("CartPole-v1")
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, "cartpole_ppo.zip")
    
    print(f"[*] Instantiating PPO Policy (MlpPolicy)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=seed
    )
    
    print(f"[*] Training PPO agent for {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps)
    print(f"[✓] Training completed! Saving model to {model_path}...")
    model.save(model_path)
    
    # Evaluate model
    print("\n[*] Evaluating trained PPO policy across 10 evaluation episodes...")
    eval_scores = []
    for ep in range(10):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        eval_scores.append(ep_reward)
    
    mean_score = float(np.mean(eval_scores))
    std_score = float(np.std(eval_scores))
    min_score = float(np.min(eval_scores))
    max_score = float(np.max(eval_scores))
    
    print(f"[✓] Evaluation Result: Mean {mean_score:.1f} ± {std_score:.1f} (Min: {min_score:.1f}, Max: {max_score:.1f})")
    
    # Export weights to JSON for Web Simulator (cartpole_sim.js)
    export_policy_weights(model, "cartpole_weights.json")
    
    eval_info = {
        "env_id": "CartPole-v1",
        "algorithm": "PPO",
        "timesteps": timesteps,
        "episodes": len(eval_scores),
        "mean_reward": mean_score,
        "std_reward": std_score,
        "min_reward": min_score,
        "max_reward": max_score,
        "solved": bool(mean_score >= 475.0),
        "scores": eval_scores
    }
    
    with open("eval_info.json", "w", encoding="utf-8") as f:
        json.dump(eval_info, f, indent=2, ensure_ascii=False)
    print("[✓] Saved benchmark metadata to eval_info.json")
    
    env.close()
    return eval_info

def export_policy_weights(model: PPO, output_json: str):
    """
    Exports PyTorch MLP weights into a JSON structure compatible with
    browser-based pure JavaScript matrix inference (cartpole_sim.js).
    """
    try:
        policy = model.policy
        # SB3 default MlpPolicy policy_net: [Linear(4, 64), Tanh, Linear(64, 64), Tanh] -> action_net: Linear(64, 2)
        state_dict = policy.state_dict()
        
        weights_data = {
            "w1": state_dict["mlp_extractor.policy_net.0.weight"].cpu().numpy().tolist(),
            "b1": state_dict["mlp_extractor.policy_net.0.bias"].cpu().numpy().tolist(),
            "w2": state_dict["mlp_extractor.policy_net.2.weight"].cpu().numpy().tolist(),
            "b2": state_dict["mlp_extractor.policy_net.2.bias"].cpu().numpy().tolist(),
            "w_out": state_dict["action_net.weight"].cpu().numpy().tolist(),
            "b_out": state_dict["action_net.bias"].cpu().numpy().tolist(),
            "activation": "tanh"
        }
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(weights_data, f, indent=2)
        print(f"[✓] Successfully exported neural network weights for web runtime: {output_json}")
    except Exception as e:
        print(f"[!] Warning: Could not export weights to JSON ({e}). Web simulator will use heuristic backup.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole PPO Trainer & Exporter")
    parser.add_argument("--timesteps", type=int, default=25_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save model")
    args = parser.parse_args()
    
    train_cartpole(timesteps=args.timesteps, seed=args.seed, model_dir=args.model_dir)
