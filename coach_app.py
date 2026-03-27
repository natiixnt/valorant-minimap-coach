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
    overlay.on_voice_change = coach.tts.set_voice
    overlay.on_voice_preview = coach.tts.preview
    overlay.set_voice_options(coach.tts.list_voices(), coach.tts.current_voice_id)
    overlay.on_lang_change = coach.set_callout_lang
    from src.ui.overlay import load_settings as _ls
    _saved = _ls()
    coach.set_callout_lang(_saved.get("callout_lang", "EN"))

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
