#!/usr/bin/env python3
"""
Entry point for the overlay application.
Tkinter must own the main thread, so the coach loop runs in a background thread.
"""
import sys
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; set ANTHROPIC_API_KEY in the environment directly

from coach import Coach, load_config
from src.ui.overlay import OverlayWindow


def main() -> None:
    config = load_config()
    coach = Coach(config)

    overlay_cfg = config.get("overlay", {})
    overlay = OverlayWindow(
        fade_after=overlay_cfg.get("fade_after", 5.0),
    )
    coach.set_overlay(overlay)

    thread = threading.Thread(target=coach.run, daemon=True, name="CoachLoop")
    thread.start()

    try:
        overlay.mainloop()
    finally:
        coach._running = False
        thread.join(timeout=3)
        coach._shutdown()


if __name__ == "__main__":
    main()
