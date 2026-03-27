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

import os
from PIL import Image, ImageTk

from coach import Coach, load_config
from src.ui.overlay import OverlayWindow

_ICON_BG = "#0d0f14"


def _build_icon(size: int = 512) -> Image.Image:
    """Generate monochrome crosshair icon for macOS template tinting."""
    from PIL import ImageDraw
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Background occupies ~76% of canvas (like macOS system icons)
    pad = size // 8
    draw.rounded_rectangle([pad, pad, size - pad - 1, size - pad - 1],
                           radius=size // 8, fill=_ICON_BG)
    cx, cy = size // 2, size // 2
    inner = size - 2 * pad
    lw  = inner // 20
    gap = inner // 5
    arm = inner // 10
    white = (255, 255, 255, 255)
    draw.rectangle([cx - lw // 2, cy - gap - arm, cx + lw // 2, cy - gap], fill=white)
    draw.rectangle([cx - lw // 2, cy + gap,       cx + lw // 2, cy + gap + arm], fill=white)
    draw.rectangle([cx - gap - arm, cy - lw // 2, cx - gap, cy + lw // 2], fill=white)
    draw.rectangle([cx + gap,       cy - lw // 2, cx + gap + arm, cy + lw // 2], fill=white)
    return img


def _set_app_icon(overlay) -> None:
    img = _build_icon()
    try:
        photo = ImageTk.PhotoImage(img)
        overlay.iconphoto(True, photo)
        overlay._icon_ref = photo
    except Exception:
        pass
    try:
        import io
        from AppKit import NSApplication, NSImage, NSData
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data   = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
        ns_img = NSImage.alloc().initWithData_(data)
        ns_img.setTemplate_(True)
        NSApplication.sharedApplication().setApplicationIconImage_(ns_img)
    except Exception:
        pass


def main() -> None:
    # Load API key from user_settings.json if not already in environment.
    # This allows exe users to set their key via the Settings UI without a .env file.
    from src.ui.overlay import load_settings as _load_settings
    _saved = _load_settings()
    _saved_key = _saved.get("anthropic_api_key", "")
    if _saved_key and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = _saved_key

    config = load_config()
    coach = Coach(config)

    overlay_cfg = config.get("overlay", {})
    overlay = OverlayWindow(
        fade_after=overlay_cfg.get("fade_after", 5.0),
    )
    _set_app_icon(overlay)
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
