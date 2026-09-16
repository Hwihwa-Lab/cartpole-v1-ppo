# -*- coding: utf-8 -*-
"""
HWIHWA LAB // Ultra-High-Definition 1:1 Square (720x720) Replay Video Generator
Optimized specifically for Hugging Face Model Hub's Video Preview Sidebar Widget.
- 1:1 Square Aspect Ratio (720x720)
- Huge, bold Cart & Pole dynamics centered in the viewport
- Big shock impulse laser & particle physics
- Clean HUD overlay (Active force, score, shock banner)
- Ultra-sharp H.264 (CRF 17, 25 FPS)
"""

import time
import io
from pathlib import Path
import numpy as np
from PIL import Image
import imageio
from playwright.sync_api import sync_playwright

def generate_square_canvas_replay():
    current_dir = Path(__file__).resolve().parent
    html_url = f"file:///{current_dir.as_posix()}/index.html"
    output_mp4 = current_dir / "replay.mp4"

    raw_frames = []

    print("🚀 Launching Headless Chromium to record 1:1 SQUARE (720x720) CartPole Dynamics...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # 720x720 Square Viewport with device_scale_factor=1
        context = browser.new_context(viewport={"width": 720, "height": 720}, device_scale_factor=1)
        page = context.new_page()

        page.goto(html_url)
        page.wait_for_timeout(1000)  # Wait for weights & chart initialization

        # Maximize Stage to 100% of 720x720 viewport
        page.evaluate("""() => {
            const header = document.querySelector('.header-bar');
            if (header) header.style.display = 'none';
            const bento = document.querySelector('.bento-deck-golden');
            if (bento) bento.style.display = 'none';
            const labMain = document.querySelector('.lab-main');
            if (labMain) {
                labMain.style.padding = '0';
                labMain.style.height = '100vh';
            }
            const stageSec = document.querySelector('.stage-section');
            if (stageSec) stageSec.style.height = '100vh';
            const area = document.querySelector('.canvas-viewport-area');
            if (area) {
                area.style.height = '100vh';
                area.style.borderRadius = '0';
                area.style.border = 'none';
            }
            window.dispatchEvent(new Event('resize'));
        }""")
        page.wait_for_timeout(300)

        stage = page.locator(".canvas-viewport-area")
        box = stage.bounding_box()
        if not box:
            stage = page.locator("#simCanvas")
            box = stage.bounding_box()

        cx = 360
        cy = 360
        print(f"  🎯 Stage Bounding Box: Width={box['width']}, Height={box['height']}")

        total_frames = 110  # ~4.4 seconds @ 25 FPS
        for i in range(total_frames):
            # Apply dynamic mouse disturbance impulse
            if i == 25:
                page.mouse.move(cx - 20, cy + 50)
                page.mouse.down()
            elif i == 30:
                page.mouse.move(cx + 140, cy - 40)
            elif i == 35:
                page.mouse.up()

            # Optional second gentle tap at frame 70
            if i == 70:
                page.mouse.move(cx + 20, cy + 40)
                page.mouse.down()
            elif i == 74:
                page.mouse.move(cx - 80, cy - 30)
            elif i == 78:
                page.mouse.up()

            screenshot_bytes = stage.screenshot(type="png")
            img = Image.open(io.BytesIO(screenshot_bytes))

            # Resize precisely to 720x720 if slight pixel sub-pixel rounding occurs
            if img.width != 720 or img.height != 720:
                img = img.resize((720, 720), Image.Resampling.LANCZOS)

            raw_frames.append(np.array(img.convert("RGB")))
            time.sleep(0.035)

        browser.close()

    print(f"🎬 Encoding {len(raw_frames)} square frames (720x720) to H.264 replay.mp4...")
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
    print(f"[✓] SUCCESS: 1:1 Square (720x720) Replay saved to: {output_mp4} ({size_kb} KB)")
    return output_mp4

if __name__ == "__main__":
    generate_square_canvas_replay()
