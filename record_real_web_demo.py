# -*- coding: utf-8 -*-
"""
HWIHWA LAB // Real In-Browser Screen Recorder for CartPole Physical AI Suite
Optimized WebP/GIF compression for ultra-fast loading on Hugging Face & GitHub.
"""

import time
import io
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

def record_real_web_demo():
    current_dir = Path(__file__).resolve().parent
    html_url = f"file:///{current_dir.as_posix()}/index.html"
    assets_dir = current_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = []

    print("🚀 Launching Headless Chromium to record REAL CartPole Simulator...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
        page = context.new_page()

        page.goto(html_url)
        page.wait_for_timeout(800)

        canvas = page.locator("#simCanvas")
        box = canvas.bounding_box()
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2

        total_frames = 65
        for i in range(total_frames):
            # Mouse disturbance at frame 16-20
            if i == 15:
                page.mouse.move(cx - 20, cy + 20)
                page.mouse.down()
            elif i == 18:
                page.mouse.move(cx + 140, cy - 30)
            elif i == 21:
                page.mouse.up()

            screenshot_bytes = page.screenshot(type="png")
            img = Image.open(io.BytesIO(screenshot_bytes))

            # Resize to 800x450 for crisp, fast-loading web display
            img_resized = img.resize((800, 450), Image.Resampling.LANCZOS)
            # Quantize to adaptive 128-color palette to shrink GIF file size drastically
            img_quant = img_resized.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
            raw_frames.append(img_quant)

            time.sleep(0.04)

        browser.close()

    output_gif = assets_dir / "cartpole_disturbance_recovery.gif"

    raw_frames[0].save(
        output_gif,
        save_all=True,
        append_images=raw_frames[1:],
        duration=50,  # 20 FPS (50ms per frame)
        loop=0,
        optimize=True
    )

    size_mb = round(output_gif.stat().st_size / 1024 / 1024, 2)
    print(f"[✓] SUCCESS: Optimized Real Demo GIF generated: {output_gif} ({len(raw_frames)} frames, {size_mb} MB)")
    return output_gif

if __name__ == "__main__":
    record_real_web_demo()
