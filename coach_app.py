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

_ICON_PATH = os.path.join(os.path.dirname(__file__), "src", "ui", "app_icon.png")
_ICON_SRC   = os.path.join(os.path.dirname(__file__), "src", "ui", "fa-solid-900.ttf")
_ICON_BG    = "#0d0f14"


def _system_accent() -> str:
    """Return macOS system accent color as hex, fallback to app default."""
    try:
        from AppKit import NSColor
        c = NSColor.controlAccentColor().colorUsingColorSpaceName_(
            "NSCalibratedRGBColorSpace"
        )
        r = int(c.redComponent() * 255)
        g = int(c.greenComponent() * 255)
        b = int(c.blueComponent() * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "#e84057"


def _build_icon(accent: str, size: int = 512) -> Image.Image:
    """Generate crosshair icon with given accent color."""
    from PIL import ImageDraw
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 8, fill=_ICON_BG)
    cx, cy = size // 2, size // 2
    lw  = size // 22
    gap = size // 7
    arm = size // 5
    draw.rectangle([cx - lw // 2, cy - gap - arm, cx + lw // 2, cy - gap], fill=accent)
    draw.rectangle([cx - lw // 2, cy + gap,       cx + lw // 2, cy + gap + arm], fill=accent)
    draw.rectangle([cx - gap - arm, cy - lw // 2, cx - gap, cy + lw // 2], fill=accent)
    draw.rectangle([cx + gap,       cy - lw // 2, cx + gap + arm, cy + lw // 2], fill=accent)
    return img


def _set_app_icon(overlay) -> None:
    accent = _system_accent()
    img    = _build_icon(accent)
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
        NSApplication.sharedApplication().setApplicationIconImage_(ns_img)
    except Exception:
        pass


def main() -> None:
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
