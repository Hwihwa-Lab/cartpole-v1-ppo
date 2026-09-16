# -*- coding: utf-8 -*-
"""
HWIHWA LAB Universal Hugging Face Deployment Pipeline
One-Click Deployment for CartPole-v1 PPO Neural Balancer.
Supports:
1) Hugging Face Spaces (Live Interactive Web Simulator - Static SDK)
2) Hugging Face Models (Official Benchmark Report, Weights & Bundle)
3) Hugging Face Datasets (Private High-Frequency Trajectory Benchmark Dataset)
4) All-in-One Dual / Triple Deploy
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path
from typing import Any

try:
    import huggingface_hub
    from huggingface_hub import HfApi, create_repo, get_token
except ImportError:
    huggingface_hub = None  # type: ignore
    HfApi = None  # type: ignore
    create_repo = None  # type: ignore
    get_token = None  # type: ignore

DEFAULT_REPO_NAME = "cartpole-v1-ppo"

def create_bundle_zip(current_dir: Path) -> str:
    zip_path = current_dir / "cartpole_ppo_bundle.zip"
    include_files = [
        "index.html", "style.css", "cartpole_sim.js", "cartpole_weights.json",
        "train.py", "run.py", "run_desktop.py", "eval_info.json", "requirements.txt",
        "README.md", "README_KR.md", "benchmark_experiments.py", "generate_trajectory_dataset.py"
    ]
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in include_files:
            fp = current_dir / f
            if fp.exists():
                zipf.write(fp, arcname=f)
            else:
                # Include models/cartpole_ppo.zip if exists
                model_fp = current_dir / "models" / "cartpole_ppo.zip"
                if model_fp.exists() and "models/cartpole_ppo.zip" not in zipf.namelist():
                    zipf.write(model_fp, arcname="models/cartpole_ppo.zip")
    print(f"📦 Created verified bundle archive: {zip_path.name}")
    return str(zip_path)

def deploy_spaces(api: Any, username: str, repo_name: str, token: str, current_dir: Path):
    repo_id = f"{username}/{repo_name}"
    print(f"\n🚀 [Spaces] Deploying to Hugging Face Spaces: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", exist_ok=True, token=token)

    space_files = [
        "index.html",
        "style.css",
        "cartpole_sim.js",
        "cartpole_weights.json",
        "eval_info.json"
    ]

    for filename in space_files:
        filepath = current_dir / filename
        if filepath.exists():
            print(f"  ⬆️ Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="space",
                token=token
            )

    # Always ensure Space README has sdk: static configuration
    space_readme = """---
title: CartPole-v1 Physical AI Cybernetic Lab
emoji: 🪐
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# 🪐 CartPole-v1 Physical AI Interactive Cybernetic Lab

Live interactive physics simulator for PPO Neural Policy vs Classical Optimal LQR.

- 🤖 **Model Hub**: [hwihwalab/cartpole-v1-ppo](https://huggingface.co/hwihwalab/cartpole-v1-ppo)
- 🐙 **GitHub**: [Hwihwa-Lab/cartpole-v1-ppo](https://github.com/Hwihwa-Lab/cartpole-v1-ppo)
"""
    api.upload_file(
        path_or_fileobj=space_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        token=token
    )

    print(f"[✓] Spaces deployment complete! Live at: https://huggingface.co/spaces/{repo_id}")


def deploy_models(api: Any, username: str, repo_name: str, token: str, current_dir: Path, bundle_zip: str):
    repo_id = f"{username}/{repo_name}"
    print(f"\n🚀 [Models] Deploying to Hugging Face Model Hub: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    # 1. Model Card (README.md)
    readme_path = current_dir / "README.md"
    if readme_path.exists():
        print(f"  ⬆️ Uploading Model Card (README.md)...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

    # 2. README_KR.md
    readme_kr = current_dir / "README_KR.md"
    if readme_kr.exists():
        api.upload_file(
            path_or_fileobj=str(readme_kr),
            path_in_repo="README_KR.md",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

    # 3. Model weights
    model_weight = current_dir / "models" / "cartpole_ppo.zip"
    if model_weight.exists():
        print(f"  ⬆️ Uploading trained PPO weights (cartpole_ppo.zip)...")
        api.upload_file(
            path_or_fileobj=str(model_weight),
            path_in_repo="cartpole_ppo.zip",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

    # 4. Weights JSON & Eval info
    for fn in ["cartpole_weights.json", "eval_info.json"]:
        fp = current_dir / fn
        if fp.exists():
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fn,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )

    # 5. Bundle ZIP
    if os.path.exists(bundle_zip):
        print(f"  ⬆️ Uploading project bundle archive ({os.path.basename(bundle_zip)})...")
        api.upload_file(
            path_or_fileobj=bundle_zip,
            path_in_repo=os.path.basename(bundle_zip),
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

    print(f"[✓] Model Hub deployment complete! Live at: https://huggingface.co/{repo_id}")

def deploy_dataset(api: Any, username: str, repo_name: str, token: str, current_dir: Path, private: bool = True):
    repo_id = f"{username}/{repo_name}"
    vis_label = "Private 🔒" if private else "Public 🌐"
    print(f"\n🚀 [Datasets] Deploying to Hugging Face Datasets ({vis_label}): {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    data_dir = current_dir / "data"
    if not data_dir.exists():
        print("[!] Generating trajectory dataset first...")
        from generate_trajectory_dataset import generate_dataset
        generate_dataset(data_dir, episodes_per_combo=10)

    # Upload Dataset Files
    dataset_files = [
        ("README.md", "README.md"),
        ("cartpole_planetary_trajectories.parquet", "cartpole_planetary_trajectories.parquet"),
        ("cartpole_planetary_trajectories.csv", "cartpole_planetary_trajectories.csv"),
        ("cartpole_planetary_trajectories.jsonl", "cartpole_planetary_trajectories.jsonl"),
    ]

    for local_name, repo_name_path in dataset_files:
        fp = data_dir / local_name
        if fp.exists():
            print(f"  ⬆️ Uploading Dataset file: {local_name} ({round(fp.stat().st_size / 1024 / 1024, 2)} MB)...")
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=repo_name_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )

    # Also upload benchmark_results.json if exists
    bench_file = current_dir / "benchmark_results.json"
    if bench_file.exists():
        api.upload_file(
            path_or_fileobj=str(bench_file),
            path_in_repo="benchmark_summary.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )

    print(f"[✓] Dataset deployment complete ({vis_label})! Live at: https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Hwihwa Lab Universal Hugging Face Deployment Pipeline")
    parser.add_argument("--repo-name", type=str, default=DEFAULT_REPO_NAME, help="Hugging Face Repository Name")
    parser.add_argument("--username", type=str, default=None, help="Hugging Face username/org (default: token user or hwihwalab)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Write Token")
    parser.add_argument("--mode", type=str, choices=["all", "spaces", "models", "datasets"], default="datasets", help="Deployment target mode")
    parser.add_argument("--public-dataset", action="store_true", help="Set dataset to public (default: private)")
    args = parser.parse_args()

    if HfApi is None:
        print("[!] Error: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token and get_token is not None:
        token = get_token()

    api = HfApi(token=token)
    try:
        user_info = api.whoami(token=token)
        detected_username = user_info.get("name", "hwihwalab")
        print(f"[✓] Authenticated as Hugging Face user: @{detected_username}")
    except Exception as e:
        detected_username = "hwihwalab"
        print(f"[!] Authentication warning: {e}. Defaulting username to: @{detected_username}")

    username = args.username or detected_username
    current_dir = Path(__file__).resolve().parent

    print("\n" + "="*60)
    print("🚀 HWIHWA LAB // CartPole-v1 PPO Deployment Pipeline")
    print(f"Target Mode: {args.mode.upper()}  |  Repo: {username}/{args.repo_name}")
    print("="*60)

    if args.mode in ["all", "spaces"]:
        deploy_spaces(api, username, args.repo_name, token, current_dir)

    if args.mode in ["all", "models"]:
        bundle_zip = create_bundle_zip(current_dir)
        deploy_models(api, username, args.repo_name, token, current_dir, bundle_zip)

    if args.mode in ["all", "datasets"]:
        is_private = not args.public_dataset
        deploy_dataset(api, username, args.repo_name, token, current_dir, private=is_private)

    print("\n" + "="*60)
    print("✨ ALL REQUESTED DEPLOYMENTS EXECUTED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
