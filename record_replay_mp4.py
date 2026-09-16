# -*- coding: utf-8 -*-
"""
HWIHWA LAB // Ultra-High-Definition Canvas-Focused `replay.mp4` Generator
Directly captures the simulation stage canvas (.canvas-viewport-area) with:
1) 100% Focused Stage: No unnecessary control buttons or settings clutter
2) Huge, clear Cart & Pole catching dynamics
3) Crisp Mouse Shock Impulse Laser & Shock Banner
4) Real-time Phase Portrait Attractor Convergence Radar
5) High-bitrate H.264 (CRF 16) for razor-sharp visual clarity
"""

import time
import io
from pathlib import Path
import numpy as np
from PIL import Image
import imageio
from playwright.sync_api import sync_playwright

def generate_crisp_canvas_replay():
    current_dir = Path(__file__).resolve().parent
    html_url = f"file:///{current_dir.as_posix()}/index.html"
    output_mp4 = current_dir / "replay.mp4"

    raw_frames = []

    print("🚀 Launching Headless Chromium to record ULTRA-CRISP Canvas Focus...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Higher resolution 1440x900 viewport for retina-grade crisp canvas rendering
        context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page = context.new_page()

        page.goto(html_url)
        page.wait_for_timeout(1000)  # Wait for weights, physics & chart initialization

        # Focus directly on the canvas viewport area (Stage only!)
        stage = page.locator(".canvas-viewport-area")
        box = stage.bounding_box()
        if not box:
            print("[!] Could not find .canvas-viewport-area, falling back to #simCanvas")
            stage = page.locator("#simCanvas")
            box = stage.bounding_box()

        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2

        print(f"  🎯 Stage Bounding Box: Width={box['width']}, Height={box['height']}")

        total_frames = 100  # ~4 seconds @ 25 FPS
        for i in range(total_frames):
            # Trigger realistic mouse drag disturbance impulse at frame 22-28
            if i == 20:
                page.mouse.move(cx - 30, cy + 30)
                page.mouse.down()
            elif i == 24:
                page.mouse.move(cx + 180, cy - 45)
            elif i == 28:
                page.mouse.up()

            # Capture direct stage element screenshot for 100% canvas focus
            screenshot_bytes = stage.screenshot(type="png")
            img = Image.open(io.BytesIO(screenshot_bytes))

            # Ensure even dimensions for H.264 macroblock compatibility (e.g., 1200x520)
            target_w = (img.width // 2) * 2
            target_h = (img.height // 2) * 2
            if img.width != target_w or img.height != target_h:
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

            raw_frames.append(np.array(img.convert("RGB")))
            time.sleep(0.038)

        browser.close()

    print(f"🎬 Encoding {len(raw_frames)} ultra-crisp frames to H.264 replay.mp4 (High Quality CRF)...")
    
    # High-bitrate encoding with ffmpeg parameters
    writer = imageio.get_writer(
        str(output_mp4),
        fps=25,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "17", "-preset", "slow"]
    )
    for f in raw_frames:
        writer.append_data(f)
    writer.close()

    size_kb = round(output_mp4.stat().st_size / 1024, 1)
    print(f"[✓] SUCCESS: Ultra-Crisp Canvas Replay ready at: {output_mp4} ({size_kb} KB)")
    return output_mp4

if __name__ == "__main__":
    generate_crisp_canvas_replay()
