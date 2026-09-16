# -*- coding: utf-8 -*-
"""
HWIHWA LAB // CartPole-v1 Physical AI & Robotics Lab Standalone App
CARTPOLE-V1 // PHYSICAL AI & ROBOTICS LAB | HWIHWA LAB
PPO NEURAL POLICY VS CLASSICAL LQR // SIM-TO-REAL BENCHMARK | HWIHWA LAB
"""

import os
import sys
import time
import socket
import threading
import subprocess
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
HOST = "127.0.0.1"

class NoCacheHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Zero-cache HTTP Handler ensures every code update is served live instantly."""
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean console logging
        sys.stderr.write(f"  ⚡ [{self.log_date_time_string()}] {format%args}\n")

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((HOST, p)) != 0:
                return p
    return start_port

def launch_standalone_window(base_url: str):
    """
    Launches Chrome/Edge in dedicated Application Mode (--app flag).
    Uses anti-caching flags and fresh session timestamp.
    """
    time.sleep(0.4)  # Wait for server startup
    fresh_url = f"{base_url}/?t={int(time.time() * 1000)}"

    edge_paths = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LocalAppData%\Microsoft\Edge\Application\msedge.exe"),
    ]
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]

    for exe in edge_paths + chrome_paths:
        if os.path.exists(exe):
            try:
                cmd = [
                    exe,
                    f"--app={fresh_url}",
                    "--disable-cache",
                    "--disk-cache-size=1",
                    "--media-cache-size=1",
                    "--window-size=1440,880",
                    "--window-position=60,30"
                ]
                subprocess.Popen(cmd)
                print(f"🚀 [Standalone Desktop App Launched] {os.path.basename(exe)} --app={fresh_url}")
                return
            except Exception as e:
                print(f"[!] App mode fallback error: {e}")

    # Fallback to default browser if neither edge nor chrome found
    print(f"🌐 [Opening Standard Browser] {fresh_url}")
    webbrowser.open(fresh_url)

def run_mission_control():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cur_dir)

    port = find_available_port(PORT)
    base_url = f"http://{HOST}:{port}"

    print("\n" + "=" * 74)
    print("🤖  CARTPOLE-V1 // PHYSICAL AI & ROBOTICS LAB | HWIHWA LAB")
    print("🔬  PPO NEURAL POLICY VS CLASSICAL LQR // SIM-TO-REAL BENCHMARK")
    print(f"🌐  Local Host: {base_url}")
    print("🖥️  Window Mode: Dedicated Standalone App (--app borderless)")
    print("    [Press Ctrl+C in this terminal to close lab]")
    print("=" * 74 + "\n")

    # Launch Standalone Window in background thread
    threading.Thread(target=launch_standalone_window, args=(base_url,), daemon=True).start()

    server = HTTPServer((HOST, port), NoCacheHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Physical AI Lab closed safely.")
        server.server_close()

if __name__ == "__main__":
    run_mission_control()
