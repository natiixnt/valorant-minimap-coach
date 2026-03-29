"""
Esports overlay - custom color picker, persistent settings,
enemy sighting history, voice & language selection.
"""
import colorsys
import json
import os
import tkinter as tk
import time
from typing import Callable, List, Optional, Tuple

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from src.vision.enemy_tracker import EnemyTracker, TrackedEnemy

ctk.set_appearance_mode("dark")

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "user_settings.json")
_FA_FONT = os.path.join(os.path.dirname(__file__), "fa-solid-900.ttf")
_FA_GEAR    = "\uf013"
_FA_XMARK   = "\uf00d"
_FA_DROPPER = "\uf1fb"
_SIGHTING_RADIUS = 0.15


def _register_fa_font() -> str:
    """Register Font Awesome TTF so tkinter can use it as a vector font."""
    import sys
    if sys.platform == "darwin":
        try:
            import CoreText
            from Foundation import NSURL
            url = NSURL.fileURLWithPath_(os.path.abspath(_FA_FONT))
            CoreText.CTFontManagerRegisterFontsForURL(
                url, CoreText.kCTFontManagerScopeProcess, None
            )
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            import ctypes
            # FR_PRIVATE (0x10): font available only to this process, auto-removed on exit
            ctypes.windll.gdi32.AddFontResourceExW(os.path.abspath(_FA_FONT), 0x10, 0)
        except Exception:
            pass
    return "Font Awesome 6 Free Solid"


_FA_FAMILY = _register_fa_font()


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(data: dict) -> None:
    try:
        existing = load_settings()
        existing.update(data)
        tmp_path = SETTINGS_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp_path, SETTINGS_PATH)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
THEMES = {
    "VALORANT": dict(bg="#0d0f14", panel="#13161e", title="#090b0f",
                     accent="#e84057", enemy="#e84057", safe="#3ec17c",
                     text="#f0f0f0", dim="#4a5568", canvas_bg="#07090d", grid="#171b27"),
    "CYBER":    dict(bg="#0f1923", panel="#1a2332", title="#111827",
                     accent="#5bc4e8", enemy="#e84057", safe="#3ec17c",
                     text="#ecf0f1", dim="#7f8c9b", canvas_bg="#0a1018", grid="#1c2b3a"),
    "MATRIX":   dict(bg="#050f05", panel="#081408", title="#030803",
                     accent="#00ff41", enemy="#ff4444", safe="#00ff41",
                     text="#c8ffc8", dim="#2e5e2e", canvas_bg="#020602", grid="#081608"),
    "PHANTOM":  dict(bg="#120e1f", panel="#1b1630", title="#0d0a17",
                     accent="#a855f7", enemy="#f43f5e", safe="#22d3ee",
                     text="#e2e8f0", dim="#4a4060", canvas_bg="#080611", grid="#18102e"),
}

COLOR_FIELDS = [
    ("accent",    "ACCENT"),
    ("enemy",     "ENEMY"),
    ("safe",      "SAFE"),
    ("bg",        "BACKGROUND"),
    ("panel",     "PANEL"),
    ("title",     "TITLEBAR"),
    ("text",      "TEXT"),
    ("dim",       "DIMMED"),
    ("canvas_bg", "CANVAS BG"),
    ("grid",      "GRID"),
]

FONT_MONO      = ("Consolas", 10)
FONT_MONO_BOLD = ("Consolas", 10, "bold")
FONT_BIG       = ("Consolas", 22, "bold")
CANVAS_SIZE    = 130


# ---------------------------------------------------------------------------
# Minimap auto-detection
# ---------------------------------------------------------------------------
def _detect_minimap_circle(img_bgr: np.ndarray) -> Optional[dict]:
    """
    Find Valorant's circular minimap in a full-screen BGR screenshot.
    The minimap is a circle (dark olive/grey disc) in the top portion of the
    screen -- either top-left or top-right depending on player settings.
    Returns an mss-compatible region dict (bounding box) or None.
    """
    import cv2
    h, w = img_bgr.shape[:2]

    # Search the top 55 % of the screen (covers both default positions)
    roi_h = int(h * 0.55)
    roi = img_bgr[:roi_h]

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Expected minimap radius: 7 %–22 % of screen height
    min_r = max(30, int(h * 0.07))
    max_r = int(h * 0.22)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_r * 2,
        param1=60,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    # Score candidates: prefer larger radius and corner positions
    best: Optional[Tuple[int, int, int]] = None
    best_score = -1.0
    for cx, cy, r in circles[0]:
        cx, cy, r = int(cx), int(cy), int(r)
        x_edge = min(cx, w - cx)          # distance from nearest left/right edge
        score  = r - 0.3 * x_edge - 0.1 * cy
        if score > best_score:
            best_score = score
            best = (cx, cy, r)

    if best is None:
        return None

    cx, cy, r = best
    pad    = 4
    left   = max(0, cx - r - pad)
    top    = max(0, cy - r - pad)
    width  = min(w - left,      2 * (r + pad))
    height = min(roi_h - top,   2 * (r + pad))
    return {"top": top, "left": left, "width": width, "height": height}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ago(ts: float) -> str:
    e = max(0, time.time() - ts)
    return f"{int(e)}s" if e < 60 else f"{int(e / 60)}m"


def _pos_label(x: float, y: float) -> str:
    col = "LEFT" if x < 0.33 else ("RIGHT" if x > 0.66 else "MID")
    row = "TOP"  if y < 0.33 else ("BOT"   if y > 0.66 else "MID")
    if row == "MID" and col == "MID":
        return "CENTER"
    if row == "MID":
        return col
    if col == "MID":
        return row
    return f"{row}-{col}"


def _active_preset(colors: dict) -> str:
    for name, theme in THEMES.items():
        if all(colors.get(k) == v for k, v in theme.items()):
            return name
    return "CUSTOM"


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _darken(hex_color: str, factor: float = 0.75) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return _rgb_to_hex(int(r * factor), int(g * factor), int(b * factor))


def PipetteIcon(parent, command: Callable, fg: str, bg: str) -> "FAIconButton":
    """FA eyedropper icon for color rows."""
    return FAIconButton(
        parent, char=_FA_DROPPER, command=command,
        fg=fg, bg=bg, hover_bg=bg, hover_fg=fg,
        size=28, icon_size=14,
    )


# ---------------------------------------------------------------------------
# Font Awesome icon button - pure vector text via CoreText-registered font
# ---------------------------------------------------------------------------
class FAIconButton(ctk.CTkFrame):
    """Clickable widget using FA glyph rendered as vector text - crisp on any DPI."""

    def __init__(self, parent, char: str, command: Callable,
                 fg: str, bg: str, hover_bg: str, hover_fg: Optional[str] = None,
                 size: int = 42, icon_size: int = 16):
        super().__init__(parent, fg_color=bg, corner_radius=0,
                         width=size, height=size)
        self.pack_propagate(False)
        self._fg       = fg
        self._bg       = bg
        self._hover_bg = hover_bg
        self._hover_fg = hover_fg or fg

        self._lbl = ctk.CTkLabel(
            self, text=char, text_color=fg,
            fg_color="transparent",
            font=(_FA_FAMILY, icon_size),
            cursor="hand2",
        )
        self._lbl.place(x=0, y=0, relwidth=1, relheight=1)

        self._lbl.bind("<Button-1>", lambda _: command())
        self._lbl.bind("<Enter>",    self._on_enter)
        self._lbl.bind("<Leave>",    self._on_leave)

    def _on_enter(self, _) -> None:
        self.configure(fg_color=self._hover_bg)
        self._lbl.configure(fg_color=self._hover_bg, text_color=self._hover_fg)

    def _on_leave(self, _) -> None:
        self.configure(fg_color=self._bg)
        self._lbl.configure(fg_color=self._bg, text_color=self._fg)

    def recolor(self, fg: str, bg: str, hover_bg: str,
                hover_fg: Optional[str] = None) -> None:
        self._fg       = fg
        self._bg       = bg
        self._hover_bg = hover_bg
        self._hover_fg = hover_fg or fg
        self.configure(fg_color=bg)
        self._lbl.configure(fg_color=bg, text_color=fg)


# ---------------------------------------------------------------------------
# Custom in-app color picker
# ---------------------------------------------------------------------------
class ColorPickerDialog(ctk.CTkToplevel):
    SV_W, SV_H = 200, 190
    HUE_H      = 18

    def __init__(self, parent, initial_hex: str, colors: dict, on_confirm: Callable):
        super().__init__(parent)
        self._colors     = colors
        self._on_confirm = on_confirm
        self._orig_hex   = initial_hex
        self._suppress   = False

        r, g, b = _hex_to_rgb(initial_hex)
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        self._hue = h
        self._sat = s
        self._val = v

        self._sv_photo:  Optional[ImageTk.PhotoImage] = None
        self._hue_photo: Optional[ImageTk.PhotoImage] = None

        self.title("COLOR PICKER")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.configure(fg_color=colors["bg"])
        self._build()
        self._render_all()

    def _build(self) -> None:
        c = self._colors
        W = self.SV_W

        hdr = ctk.CTkFrame(self, fg_color=c["title"], corner_radius=0, height=36)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkFrame(hdr, fg_color=c["accent"], width=4, corner_radius=0).pack(
            side="left", fill="y")
        ctk.CTkLabel(hdr, text="  COLOR PICKER", text_color=c["accent"],
                     font=("Consolas", 10, "bold")).pack(side="left", padx=6)

        body = ctk.CTkFrame(self, fg_color=c["bg"], corner_radius=0)
        body.pack(padx=14, pady=10)

        self._sv_canvas = tk.Canvas(body, width=W, height=self.SV_H,
                                    highlightthickness=1, highlightbackground=c["dim"],
                                    cursor="crosshair")
        self._sv_canvas.pack()
        self._sv_canvas.bind("<ButtonPress-1>", self._sv_press)
        self._sv_canvas.bind("<B1-Motion>",     self._sv_drag)

        ctk.CTkLabel(body, text="", height=4, fg_color=c["bg"]).pack()

        self._hue_canvas = tk.Canvas(body, width=W, height=self.HUE_H,
                                     highlightthickness=1, highlightbackground=c["dim"],
                                     cursor="crosshair")
        self._hue_canvas.pack()
        self._hue_canvas.bind("<ButtonPress-1>", self._hue_press)
        self._hue_canvas.bind("<B1-Motion>",     self._hue_drag)

        ctk.CTkLabel(body, text="", height=8, fg_color=c["bg"]).pack()

        prev_row = ctk.CTkFrame(body, fg_color="transparent")
        prev_row.pack(fill="x")

        # Before/after swatches (square)
        self._before_sw = tk.Canvas(prev_row, width=36, height=36, highlightthickness=0)
        self._before_sw.configure(bg=self._orig_hex)
        self._before_sw.pack(side="left")

        self._after_sw = tk.Canvas(prev_row, width=36, height=36, highlightthickness=0)
        self._after_sw.configure(bg=self._orig_hex)
        self._after_sw.pack(side="left", padx=(2, 10))

        ctk.CTkLabel(prev_row, text="#", text_color=c["dim"],
                     font=("Consolas", 10, "bold")).pack(side="left")

        self._hex_var = tk.StringVar(value=self._orig_hex.lstrip("#"))
        ctk.CTkEntry(
            prev_row, textvariable=self._hex_var,
            width=90, height=36, fg_color=c["panel"],
            text_color=c["text"], border_color=c["dim"], border_width=1,
            font=("Consolas", 11, "bold"), corner_radius=3,
        ).pack(side="left")
        self._hex_var.trace_add("write", self._on_hex_typed)

        ctk.CTkLabel(body, text="", height=10, fg_color=c["bg"]).pack()

        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.pack(fill="x")

        ctk.CTkButton(
            btn_row, text="CONFIRM", width=90, height=30,
            fg_color=c["accent"], text_color=c["bg"],
            hover_color=c["safe"], font=("Consolas", 9, "bold"),
            corner_radius=2, command=self._confirm,
        ).pack(side="right")

        ctk.CTkButton(
            btn_row, text="CANCEL", width=80, height=30,
            fg_color=c["panel"], text_color=c["dim"],
            hover_color=c["dim"], font=("Consolas", 9, "bold"),
            corner_radius=2, command=self.destroy,
        ).pack(side="right", padx=(0, 6))

    def _render_all(self) -> None:
        self._render_hue()
        self._render_sv()
        self._update_preview()

    def _render_sv(self) -> None:
        W, H = self.SV_W, self.SV_H
        s_lin = np.linspace(0, 1, W, dtype=np.float32)
        v_lin = np.linspace(1, 0, H, dtype=np.float32)
        sv, vv = np.meshgrid(s_lin, v_lin)
        h   = self._hue
        c_v = vv * sv
        h6  = h * 6.0
        x_v = c_v * (1.0 - np.abs(h6 % 2 - 1))
        m   = vv - c_v
        sec = int(h6) % 6
        if   sec == 0: r, g, b = c_v, x_v, m
        elif sec == 1: r, g, b = x_v, c_v, m
        elif sec == 2: r, g, b = m,   c_v, x_v
        elif sec == 3: r, g, b = m,   x_v, c_v
        elif sec == 4: r, g, b = x_v, m,   c_v
        else:          r, g, b = c_v, m,   x_v
        arr = (np.stack([r, g, b], axis=2) * 255).clip(0, 255).astype(np.uint8)
        self._sv_photo = ImageTk.PhotoImage(Image.fromarray(arr, "RGB"))
        cv = self._sv_canvas
        cv.delete("all")
        cv.create_image(0, 0, anchor="nw", image=self._sv_photo)
        cx = int(self._sat * W)
        cy = int((1 - self._val) * H)
        cv.create_oval(cx-6, cy-6, cx+6, cy+6, outline="white", width=2)
        cv.create_oval(cx-2, cy-2, cx+2, cy+2, fill="black", outline="")

    def _render_hue(self) -> None:
        W, H = self.SV_W, self.HUE_H
        h_lin = np.linspace(0, 1, W, dtype=np.float32)
        h6  = h_lin * 6.0
        c_v = np.ones(W, dtype=np.float32)
        x_v = 1.0 - np.abs(h6 % 2 - 1)
        sec = np.floor(h6).astype(int) % 6
        r = np.where((sec==0)|(sec==5), c_v, np.where((sec==1)|(sec==4), x_v, 0.0))
        g = np.where((sec==1)|(sec==2), c_v, np.where((sec==0)|(sec==3), x_v, 0.0))
        b = np.where((sec==3)|(sec==4), c_v, np.where((sec==2)|(sec==5), x_v, 0.0))
        row = (np.stack([r, g, b], axis=1) * 255).clip(0, 255).astype(np.uint8)
        arr = np.repeat(row[np.newaxis, :, :], H, axis=0)
        self._hue_photo = ImageTk.PhotoImage(Image.fromarray(arr, "RGB"))
        cv = self._hue_canvas
        cv.delete("all")
        cv.create_image(0, 0, anchor="nw", image=self._hue_photo)
        hx = int(self._hue * W)
        cv.create_rectangle(max(0, hx-2), 0, min(W, hx+2), H, fill="white", outline="")

    def _update_preview(self) -> None:
        r, g, b = colorsys.hsv_to_rgb(self._hue, self._sat, self._val)
        hex_str = _rgb_to_hex(int(r*255), int(g*255), int(b*255))
        try:
            self._after_sw.configure(bg=hex_str)
        except tk.TclError:
            pass
        self._suppress = True
        self._hex_var.set(hex_str.lstrip("#"))
        self._suppress = False

    def _sv_press(self, e) -> None: self._set_sv(e.x, e.y)
    def _sv_drag(self, e)  -> None: self._set_sv(e.x, e.y)

    def _set_sv(self, x: int, y: int) -> None:
        self._sat = max(0.0, min(1.0, x / self.SV_W))
        self._val = max(0.0, min(1.0, 1.0 - y / self.SV_H))
        self._render_sv()
        self._update_preview()

    def _hue_press(self, e) -> None: self._set_hue(e.x)
    def _hue_drag(self, e)  -> None: self._set_hue(e.x)

    def _set_hue(self, x: int) -> None:
        self._hue = max(0.0, min(0.9999, x / self.SV_W))
        self._render_hue()
        self._render_sv()
        self._update_preview()

    def _on_hex_typed(self, *_) -> None:
        if self._suppress:
            return
        val = self._hex_var.get().strip().lstrip("#")
        if len(val) == 6:
            try:
                r, g, b = _hex_to_rgb("#" + val)
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                self._hue, self._sat, self._val = h, s, v
                self._render_hue()
                self._render_sv()
                try:
                    self._after_sw.configure(bg="#" + val)
                except tk.TclError:
                    pass
            except ValueError:
                pass

    def _confirm(self) -> None:
        r, g, b = colorsys.hsv_to_rgb(self._hue, self._sat, self._val)
        hex_val = _rgb_to_hex(int(r*255), int(g*255), int(b*255))
        self.destroy()
        try:
            self._on_confirm(hex_val)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Color row widget
# ---------------------------------------------------------------------------
class ColorRow(ctk.CTkFrame):
    SZ = 28

    def __init__(self, parent, key: str, label: str, value: str,
                 colors: dict, **kw):
        super().__init__(parent, fg_color=colors["panel"],
                         corner_radius=3, height=self.SZ + 10, **kw)
        self.pack_propagate(False)
        self._key    = key
        self._colors = colors
        self._picker_win: Optional[ColorPickerDialog] = None

        # Square swatch - click opens picker
        self._swatch = tk.Canvas(self, width=self.SZ, height=self.SZ,
                                  highlightthickness=1,
                                  highlightbackground=colors["dim"],
                                  cursor="hand2")
        self._swatch.configure(bg=value)
        self._swatch.pack(side="left", padx=(8, 6), pady=5)
        self._swatch.bind("<Button-1>", lambda _: self._open_picker())

        ctk.CTkLabel(self, text=label, text_color=colors["dim"],
                     font=("Consolas", 8, "bold"), width=80, anchor="w").pack(
            side="left", padx=(0, 4))

        # Pipette icon
        PipetteIcon(self, command=self._open_picker,
                    fg=colors["dim"], bg=colors["panel"]).pack(
            side="right", padx=(0, 8), pady=5)

        # Hex entry
        self._entry = ctk.CTkEntry(
            self, width=82, height=24,
            fg_color=colors["bg"], text_color=colors["text"],
            border_color=colors["dim"], border_width=1,
            font=("Consolas", 9), corner_radius=3,
        )
        self._entry.insert(0, value)
        self._entry.pack(side="right", padx=(0, 6), pady=7)
        self._entry.bind("<KeyRelease>", self._on_hex_typed)

    def get(self) -> str:
        return self._entry.get().strip()

    def set(self, value: str) -> None:
        self._entry.delete(0, "end")
        self._entry.insert(0, value)
        self._update_swatch(value)

    def _open_picker(self) -> None:
        if self._picker_win and self._picker_win.winfo_exists():
            self._picker_win.focus()
            return
        current = self.get()
        try:
            _hex_to_rgb(current)
        except Exception:
            current = "#ffffff"
        self._picker_win = ColorPickerDialog(
            self, current, self._colors,
            on_confirm=lambda h: self.set(h),
        )

    def _on_hex_typed(self, _=None) -> None:
        val = self.get()
        if len(val) == 7 and val.startswith("#"):
            try:
                int(val[1:], 16)
                self._update_swatch(val)
            except ValueError:
                pass

    def _update_swatch(self, color: str) -> None:
        try:
            self._swatch.configure(bg=color)
        except tk.TclError:
            pass


# ---------------------------------------------------------------------------
# Settings window
# ---------------------------------------------------------------------------
class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master: "OverlayWindow"):
        super().__init__(master)
        self._master = master
        self.title("SETTINGS")
        self.geometry("390x610")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.configure(fg_color=master._c["bg"])
        self._rows:        dict = {}
        self._preset_btns: dict = {}
        self._lang_btns:   dict = {}
        self._saved_custom = load_settings().get("colors", None)
        self._build()

    def _section(self, parent, text: str) -> None:
        ctk.CTkLabel(parent, text=text, text_color=self._master._c["accent"],
                     font=("Consolas", 9, "bold")).pack(
            anchor="w", padx=14, pady=(14, 6))

    def _make_scroll_area(self, c: dict):
        """Custom scroll area: tk.Canvas + frame. Works on Tk 9 / macOS."""
        outer = tk.Frame(self, bg=c["bg"])

        vbar = ctk.CTkScrollbar(
            outer, orientation="vertical",
            fg_color=c["bg"],
            button_color=c["dim"],
            button_hover_color=c["accent"],
            minimum_pixel_length=30,
            width=6,
        )
        vbar.pack(side="right", fill="y", padx=(0, 2))

        canvas = tk.Canvas(outer, bg=c["bg"], highlightthickness=0,
                           yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.configure(command=canvas.yview)

        content = tk.Frame(canvas, bg=c["bg"])
        win_id = canvas.create_window(0, 0, anchor="nw", window=content)

        canvas.configure(yscrollincrement=1)

        def _on_content(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas(e):
            canvas.itemconfig(win_id, width=e.width)

        content.bind("<Configure>", _on_content)
        canvas.bind("<Configure>", _on_canvas)

        # macOS Tk 9: <MouseWheel> never fires from touchpad.
        # Intercept NSScrollWheel events directly via PyObjC.
        self._ns_monitor = None
        try:
            from AppKit import NSEvent
            _NSScrollWheelMask = 1 << 22

            def _ns_handler(ns_event):
                dy = ns_event.scrollingDeltaY()
                if dy:
                    try:
                        if canvas.winfo_exists():
                            canvas.yview_scroll(int(-dy), "units")
                    except Exception:
                        pass
                return ns_event

            self._ns_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                _NSScrollWheelMask, _ns_handler
            )
        except Exception:
            # Fallback: standard MouseWheel (works with physical mouse wheel)
            def _wheel(event):
                canvas.yview_scroll(int(-event.delta), "units")
            canvas.bind("<MouseWheel>", _wheel)
            content.bind("<MouseWheel>", _wheel)
            self.bind("<MouseWheel>", _wheel)

        def _remove_monitor():
            if self._ns_monitor:
                try:
                    from AppKit import NSEvent
                    NSEvent.removeMonitor_(self._ns_monitor)
                    self._ns_monitor = None
                except Exception:
                    pass

        canvas.bind("<Destroy>", lambda _: _remove_monitor())
        self.bind("<Destroy>", lambda _: _remove_monitor())

        return outer, content

    def _divider(self, parent) -> None:
        ctk.CTkFrame(parent, fg_color=self._master._c["dim"],
                     height=1, corner_radius=0).pack(fill="x", padx=14, pady=(4, 10))

    def _build(self) -> None:
        c = self._master._c

        hdr = ctk.CTkFrame(self, fg_color=c["title"], corner_radius=0, height=44)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkFrame(hdr, fg_color=c["accent"], width=4, corner_radius=0).pack(
            side="left", fill="y")
        ctk.CTkLabel(hdr, text="  SETTINGS", text_color=c["accent"],
                     font=("Consolas", 12, "bold")).pack(side="left", padx=8)

        scroll, content = self._make_scroll_area(c)
        scroll.pack(fill="both", expand=True)

        self._build_map_override(content, c)
        self._divider(content)
        self._build_calibrate(content, c)
        self._divider(content)
        self._build_api_key(content, c)
        self._divider(content)
        self._build_hf_status(content, c)
        self._divider(content)
        self._build_enemy_team(content, c)
        self._divider(content)
        self._build_voice(content, c)
        self._divider(content)
        self._build_lang(content, c)
        self._divider(content)
        self._build_presets(content, c)
        self._divider(content)

        self._section(content, "CUSTOM COLORS")
        ctk.CTkLabel(content, text="  click swatch or pipette icon to open picker",
                     text_color=c["dim"], font=("Consolas", 8)).pack(
            anchor="w", padx=14, pady=(0, 6))

        for key, label in COLOR_FIELDS:
            row = ColorRow(content, key=key, label=label, value=c[key], colors=c)
            row.pack(fill="x", padx=14, pady=2)
            self._rows[key] = row

        ctk.CTkLabel(content, text="", height=6).pack()

        bottom = ctk.CTkFrame(self, fg_color=c["title"], corner_radius=0, height=50)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        ctk.CTkButton(
            bottom, text="APPLY - SAVE", width=120, height=30,
            fg_color=c["accent"], text_color=c["bg"],
            hover_color=_darken(c["accent"]), font=("Consolas", 9, "bold"),
            corner_radius=2, command=self._apply_and_save,
        ).pack(side="right", padx=12, pady=10)

        ctk.CTkButton(
            bottom, text="RESET", width=80, height=30,
            fg_color=c["panel"], text_color=c["dim"],
            hover_color=c["dim"], font=("Consolas", 9, "bold"),
            corner_radius=2, command=lambda: self._apply_preset("VALORANT"),
        ).pack(side="right", padx=(0, 4), pady=10)

    # ---- map override ----

    _KNOWN_MAPS = sorted([
        "ascent", "bind", "haven", "split", "icebox",
        "lotus", "sunset", "abyss", "breeze", "fracture", "pearl",
    ])

    def _build_map_override(self, parent, c: dict) -> None:
        self._section(parent, "MAP")
        ctk.CTkLabel(parent,
                     text="  Select map manually if auto-detection fails\n"
                          "  (auto-detection requires Anthropic API key)",
                     text_color=c["dim"], font=("Consolas", 8), justify="left").pack(
            anchor="w", padx=14, pady=(0, 6))

        row = tk.Frame(parent, bg=c["bg"])
        row.pack(fill="x", padx=14, pady=(0, 8))

        options = ["auto-detect"] + [m.capitalize() for m in self._KNOWN_MAPS]
        saved_map = load_settings().get("map_override", "")
        initial = saved_map.capitalize() if saved_map else "auto-detect"

        self._map_var = tk.StringVar(value=initial)
        self._map_combo = ctk.CTkComboBox(
            row, values=options, variable=self._map_var,
            width=160, height=28,
            fg_color=c["panel"], border_color=c["dim"],
            text_color=c["text"], font=("Consolas", 9),
            dropdown_fg_color=c["panel"], dropdown_text_color=c["text"],
            button_color=c["dim"], button_hover_color=c["accent"],
            state="readonly",
        )
        self._map_combo.pack(side="left")

    # ---- minimap calibration ----

    def _build_calibrate(self, parent, c: dict) -> None:
        self._section(parent, "MINIMAP REGION")
        ctk.CTkLabel(parent,
                     text="  AUTO DETECT finds the circular minimap automatically.\n"
                          "  Switch to Valorant in-game first, then click AUTO.",
                     text_color=c["dim"], font=("Consolas", 8), justify="left").pack(
            anchor="w", padx=14, pady=(0, 6))

        self._cal_status = ctk.CTkLabel(parent, text="  not calibrated",
                                        text_color=c["dim"], font=("Consolas", 8))
        self._cal_status.pack(anchor="w", padx=14, pady=(0, 4))

        row = tk.Frame(parent, bg=c["bg"])
        row.pack(fill="x", padx=14, pady=(0, 8))

        ctk.CTkButton(
            row, text="AUTO DETECT", width=110, height=28,
            fg_color=c["accent"], text_color=c["bg"],
            hover_color=_darken(c["accent"]), font=("Consolas", 9, "bold"),
            corner_radius=2, command=self._run_auto_detect,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            row, text="MANUAL", width=80, height=28,
            fg_color=c["panel"], text_color=c["dim"],
            hover_color=c["dim"], font=("Consolas", 9, "bold"),
            corner_radius=2, command=self._run_calibration,
        ).pack(side="left")

        # Show current region if already set
        saved = load_settings().get("minimap_region")
        if saved:
            self._cal_status.configure(
                text=f"  {saved['width']}x{saved['height']} @ ({saved['left']},{saved['top']})",
                text_color="#4caf50")

    def _run_auto_detect(self) -> None:
        import threading as _t
        self._cal_status.configure(text="  detecting in 3s... switch to Valorant", text_color="#ff9800")
        self.update()

        def _do():
            import time, mss
            time.sleep(3)
            with mss.mss() as sct:
                raw = sct.grab(sct.monitors[0])
            img = np.array(raw)[:, :, :3]
            region = _detect_minimap_circle(img)
            if region:
                save_settings({"minimap_region": region})
                if hasattr(self._master, "on_minimap_region_change") and self._master.on_minimap_region_change:
                    self._master.on_minimap_region_change(region)
                self.after(0, lambda: self._cal_status.configure(
                    text=f"  {region['width']}x{region['height']} @ ({region['left']},{region['top']})",
                    text_color="#4caf50"))
            else:
                self.after(0, lambda: self._cal_status.configure(
                    text="  not found -- try MANUAL calibration",
                    text_color="#f44336"))

        _t.Thread(target=_do, daemon=True).start()

    def _run_calibration(self) -> None:
        import threading as _t
        self._cal_status.configure(text="  switch to Valorant now...", text_color="#ff9800")
        self.update()

        def _do():
            import time, mss, numpy as np, cv2
            time.sleep(3)

            # Grab full desktop
            with mss.mss() as sct:
                raw = sct.grab(sct.monitors[0])
            img = np.array(raw)[:, :, :3]

            points = []

            def _on_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                    points.append((x, y))
                    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
                    if len(points) == 2:
                        cv2.rectangle(img, points[0], points[1], (0, 255, 0), 2)
                    cv2.imshow(title, img)

            title = "Click: top-left then bottom-right of minimap | any key when done"
            cv2.imshow(title, img)
            cv2.setMouseCallback(title, _on_click)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                region = {
                    "top":    min(y1, y2),
                    "left":   min(x1, x2),
                    "width":  abs(x2 - x1),
                    "height": abs(y2 - y1),
                }
                save_settings({"minimap_region": region})
                # Notify coach via master callback if available
                if hasattr(self._master, "on_minimap_region_change") and self._master.on_minimap_region_change:
                    self._master.on_minimap_region_change(region)
                self.after(0, lambda: self._cal_status.configure(
                    text=f"  {region['width']}x{region['height']} @ ({region['left']},{region['top']})",
                    text_color="#4caf50"))
            else:
                self.after(0, lambda: self._cal_status.configure(
                    text="  cancelled -- need 2 points", text_color="#f44336"))

        _t.Thread(target=_do, daemon=True).start()

    # ---- api key ----

    def _build_api_key(self, parent, c: dict) -> None:
        self._section(parent, "ANTHROPIC API KEY")
        ctk.CTkLabel(parent, text="  required for AI callouts (claude.ai/settings → API Keys)",
                     text_color=c["dim"], font=("Consolas", 8)).pack(
            anchor="w", padx=14, pady=(0, 6))

        row = tk.Frame(parent, bg=c["bg"])
        row.pack(fill="x", padx=14, pady=(0, 4))

        self._api_key_entry = ctk.CTkEntry(
            row, width=260, height=28,
            fg_color=c["panel"], border_color=c["dim"],
            text_color=c["text"], font=("Consolas", 9),
            show="*", placeholder_text="sk-ant-...",
        )
        self._api_key_entry.pack(side="left")

        # Show/hide toggle
        self._key_visible = False
        def _toggle_show():
            self._key_visible = not self._key_visible
            self._api_key_entry.configure(show="" if self._key_visible else "*")
        ctk.CTkButton(
            row, text="👁", width=28, height=28,
            fg_color=c["panel"], text_color=c["dim"],
            hover_color=c["dim"], font=("Consolas", 10),
            corner_radius=2, command=_toggle_show,
        ).pack(side="left", padx=(4, 0))

        # Test button -- makes a real 1-token API call in a background thread
        self._test_btn = ctk.CTkButton(
            row, text="TEST", width=46, height=28,
            fg_color=c["panel"], text_color=c["dim"],
            hover_color=c["accent"], font=("Consolas", 9, "bold"),
            corner_radius=2, command=self._test_api_key,
        )
        self._test_btn.pack(side="left", padx=(4, 0))

        # Prefill from saved settings or current env
        import os as _os
        saved_key = load_settings().get("anthropic_api_key", "") or _os.environ.get("ANTHROPIC_API_KEY", "")
        if saved_key:
            self._api_key_entry.insert(0, saved_key)

        # Status label
        self._key_status = ctk.CTkLabel(parent, text="  no key - AI callouts disabled",
                                        text_color="#f44336", font=("Consolas", 8))
        self._key_status.pack(anchor="w", padx=14, pady=(0, 4))
        self._refresh_key_status()
        self._api_key_entry.bind("<KeyRelease>", lambda _: self._refresh_key_status())

    def _refresh_key_status(self) -> None:
        """Format-only check (instant, no network)."""
        if not hasattr(self, "_api_key_entry") or not hasattr(self, "_key_status"):
            return
        val = self._api_key_entry.get().strip()
        if not val:
            self._key_status.configure(text="  no key - AI callouts disabled",
                                       text_color="#f44336")
        elif val.startswith("sk-ant-") and len(val) > 20:
            self._key_status.configure(text="  key entered - click TEST to verify",
                                       text_color="#ff9800")
        else:
            self._key_status.configure(text="  unexpected format (should start with sk-ant-)",
                                       text_color="#ff9800")

    def _test_api_key(self) -> None:
        """Send a real 1-token request to Anthropic to verify the key works."""
        import threading as _t
        key = self._api_key_entry.get().strip()
        if not key:
            self._key_status.configure(text="  enter a key first", text_color="#ff9800")
            return
        self._key_status.configure(text="  testing...", text_color="#ff9800")
        self._test_btn.configure(state="disabled")

        def _run():
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=key)
                client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "hi"}],
                )
                result = ("  key valid", "#4caf50")
            except Exception as e:
                msg = str(e)
                if "401" in msg or "authentication" in msg.lower() or "invalid" in msg.lower():
                    result = ("  invalid key", "#f44336")
                elif "429" in msg:
                    result = ("  rate limited - key probably valid", "#ff9800")
                else:
                    result = (f"  error: {msg[:50]}", "#ff9800")
            def _update():
                if self.winfo_exists():
                    self._key_status.configure(text=result[0], text_color=result[1])
                    self._test_btn.configure(state="normal")
            self.after(0, _update)

        _t.Thread(target=_run, daemon=True).start()

    # ---- HF data collection status ----

    def _build_hf_status(self, parent, c: dict) -> None:
        self._section(parent, "DATA COLLECTION")
        ctk.CTkLabel(parent,
                     text="  minimap screenshots are uploaded to HuggingFace to improve the AI.\n"
                          "  Opt-out: set data_collection.enabled: false in config.yaml",
                     text_color=c["dim"], font=("Consolas", 8), justify="left").pack(
            anchor="w", padx=14, pady=(0, 6))

        self._hf_status_lbl = ctk.CTkLabel(parent, text="  checking...",
                                           text_color=c["dim"], font=("Consolas", 8))
        self._hf_status_lbl.pack(anchor="w", padx=14, pady=(0, 4))

        row = tk.Frame(parent, bg=c["bg"])
        row.pack(fill="x", padx=14, pady=(0, 8))

        self._hf_test_btn = ctk.CTkButton(
            row, text="TEST CONNECTION", width=130, height=28,
            fg_color=c["panel"], text_color=c["dim"],
            hover_color=c["accent"], font=("Consolas", 9, "bold"),
            corner_radius=2, command=self._test_hf_connection,
        )
        self._hf_test_btn.pack(side="left")

        # Show initial status from collector (if accessible)
        self.after(500, self._refresh_hf_status)

    def _refresh_hf_status(self) -> None:
        if not hasattr(self, "_hf_status_lbl"):
            return
        collector = getattr(self._master, "_collector", None) or \
                    getattr(getattr(self._master, "coach", None), "collector", None)
        # Try to reach the coach via the parent OverlayWindow
        if collector is None:
            self._hf_status_lbl.configure(text="  (open from running app to see status)",
                                          text_color="#4a5568")
            return
        ok = collector.last_upload_ok
        if ok is None:
            self._hf_status_lbl.configure(text="  connecting to HuggingFace...",
                                          text_color="#ff9800")
            self.after(1000, self._refresh_hf_status)
        elif ok:
            self._hf_status_lbl.configure(text=f"  connected - repo: {collector._hf_repo}",
                                          text_color="#4caf50")
        else:
            self._hf_status_lbl.configure(text="  connection failed - check token & repo",
                                          text_color="#f44336")

    def _test_hf_connection(self) -> None:
        import threading as _t
        self._hf_status_lbl.configure(text="  testing...", text_color="#ff9800")
        self._hf_test_btn.configure(state="disabled")

        collector = getattr(self._master, "_collector", None) or \
                    getattr(getattr(self._master, "coach", None), "collector", None)
        if collector is None:
            self._hf_status_lbl.configure(text="  not available", text_color="#f44336")
            self._hf_test_btn.configure(state="normal")
            return

        def _run():
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=collector._hf_token or None)
                api.create_repo(repo_id=collector._hf_repo, repo_type="dataset", exist_ok=True)
                result = (f"  connected - repo: {collector._hf_repo}", "#4caf50")
                collector.last_upload_ok = True
            except Exception as e:
                result = (f"  failed: {str(e)[:60]}", "#f44336")
                collector.last_upload_ok = False

            def _upd():
                if self.winfo_exists():
                    self._hf_status_lbl.configure(text=result[0], text_color=result[1])
                    self._hf_test_btn.configure(state="normal")
            self.after(0, _upd)

        _t.Thread(target=_run, daemon=True).start()

    # ---- enemy team ----

    def _build_enemy_team(self, parent, c: dict) -> None:
        from src.audio.agent_classifier import ALL_AGENTS
        from src.game.enemy_agents import agent_display

        self._section(parent, "ENEMY TEAM")
        ctk.CTkLabel(
            parent,
            text="  auto-detected on round 1 - manual override if needed",
            text_color=c["dim"], font=("Consolas", 8),
        ).pack(anchor="w", padx=14, pady=(0, 6))

        _NONE = "-"
        options = [_NONE] + [agent_display(a) for a in ALL_AGENTS]
        current = self._master._current_enemy_agents

        self._enemy_combos: List[ctk.CTkComboBox] = []

        # 3 slots per row
        for row_start in range(0, 5, 3):
            row_fr = ctk.CTkFrame(parent, fg_color="transparent")
            row_fr.pack(fill="x", padx=14, pady=(0, 4))
            for slot in range(row_start, min(row_start + 3, 5)):
                saved_name = agent_display(current[slot]) if slot < len(current) else _NONE
                combo = ctk.CTkComboBox(
                    row_fr, values=options, width=106, height=26,
                    fg_color=c["panel"], button_color=c["accent"],
                    button_hover_color=c["safe"], border_color=c["dim"],
                    text_color=c["text"], font=("Consolas", 9),
                    dropdown_fg_color=c["panel"], dropdown_text_color=c["text"],
                    dropdown_hover_color=c["accent"], corner_radius=2,
                )
                combo.set(saved_name if saved_name in options else _NONE)
                combo.pack(side="left", padx=(0, 6))
                self._enemy_combos.append(combo)

        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.pack(anchor="w", padx=14, pady=(2, 4))

        def _clear():
            for cb in self._enemy_combos:
                cb.set(_NONE)

        ctk.CTkButton(
            btn_row, text="CLEAR", width=68, height=24,
            fg_color=c["panel"], text_color=c["dim"],
            hover_color=c["dim"], font=("Consolas", 8, "bold"),
            corner_radius=2, command=_clear,
        ).pack(side="left", padx=(0, 6))

    def _get_enemy_agents(self) -> List[str]:
        """Return raw agent names from the 5 enemy team dropdowns."""
        from src.audio.agent_classifier import ALL_AGENTS
        from src.game.enemy_agents import agent_display

        display_to_raw = {agent_display(a): a for a in ALL_AGENTS}
        result = []
        for cb in getattr(self, "_enemy_combos", []):
            val = cb.get()
            raw = display_to_raw.get(val)
            if raw:
                result.append(raw)
        return result

    # ---- voice ----

    def _build_voice(self, parent, c: dict) -> None:
        self._section(parent, "CALLOUT VOICE")
        voices = self._master._voice_options
        if not voices:
            ctk.CTkLabel(parent, text="  no voices available",
                         text_color=c["dim"], font=("Consolas", 9)).pack(
                anchor="w", padx=14, pady=(0, 8))
            return

        fr = ctk.CTkFrame(parent, fg_color="transparent")
        fr.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(fr, text="FILTER:", text_color=c["dim"],
                     font=("Consolas", 8, "bold")).pack(side="left", padx=(0, 8))
        self._lang_filter = tk.StringVar(value="EN")
        for lbl, val in [("ALL", "ALL"), ("EN", "EN"), ("PL", "PL")]:
            ctk.CTkRadioButton(
                fr, text=lbl, variable=self._lang_filter, value=val,
                text_color=c["text"], fg_color=c["accent"],
                hover_color=c["accent"], border_color=c["dim"],
                font=("Consolas", 8, "bold"), command=self._refresh_voice_list,
            ).pack(side="left", padx=(0, 10))

        cr = ctk.CTkFrame(parent, fg_color="transparent")
        cr.pack(fill="x", padx=14, pady=(0, 8))

        current_id   = self._master._current_voice_id
        filtered     = self._filter_voices(voices, "EN")
        self._filtered_voices = filtered
        names        = [v["name"] for v in filtered]
        current_name = next(
            (v["name"] for v in voices if v["id"] == current_id),
            names[0] if names else "",
        )

        self._voice_combo = ctk.CTkComboBox(
            cr, values=names, width=222, height=28,
            fg_color=c["panel"], button_color=c["accent"],
            button_hover_color=c["safe"], border_color=c["dim"],
            text_color=c["text"], font=("Consolas", 9),
            dropdown_fg_color=c["panel"], dropdown_text_color=c["text"],
            dropdown_hover_color=c["accent"], corner_radius=2,
        )
        if current_name in names:
            self._voice_combo.set(current_name)
        elif names:
            self._voice_combo.set(names[0])
        self._voice_combo.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            cr, text="TEST", width=58, height=28,
            fg_color=c["panel"], text_color=c["accent"],
            hover_color=c["dim"], font=("Consolas", 8, "bold"),
            corner_radius=2, command=self._test_voice,
        ).pack(side="left")

        # Volume slider
        vr = ctk.CTkFrame(parent, fg_color="transparent")
        vr.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(vr, text="VOLUME:", text_color=c["dim"],
                     font=("Consolas", 8, "bold")).pack(side="left", padx=(0, 8))
        saved_vol = load_settings().get("tts_volume", 1.0)
        self._vol_var = tk.DoubleVar(value=saved_vol * 100)
        self._vol_lbl = ctk.CTkLabel(vr, text=f"{int(saved_vol * 100)}%",
                                     text_color=c["text"],
                                     font=("Consolas", 8, "bold"), width=36)
        self._vol_lbl.pack(side="right")
        ctk.CTkSlider(
            vr, from_=0, to=100, variable=self._vol_var, width=140,
            fg_color=c["panel"], button_color=c["accent"],
            progress_color=c["accent"], command=self._on_vol_change,
        ).pack(side="left")

    def _on_vol_change(self, val) -> None:
        pct = int(float(val))
        if hasattr(self, "_vol_lbl"):
            self._vol_lbl.configure(text=f"{pct}%")
        if self._master.on_volume_change:
            self._master.on_volume_change(float(val) / 100.0)

    def _filter_voices(self, voices: list, lang: str) -> list:
        if lang == "ALL":
            return voices
        out = [v for v in voices if v["lang"].lower().startswith(lang.lower())]
        return out or voices

    def _refresh_voice_list(self) -> None:
        if not hasattr(self, "_voice_combo") or not hasattr(self, "_lang_filter"):
            return
        filtered = self._filter_voices(self._master._voice_options,
                                       self._lang_filter.get())
        self._filtered_voices = filtered
        names = [v["name"] for v in filtered]
        self._voice_combo.configure(values=names)
        if names:
            self._voice_combo.set(names[0])

    def _get_selected_voice_id(self) -> str:
        if not self._voice_combo:
            return self._master._current_voice_id
        name = self._voice_combo.get()
        for v in (self._filtered_voices or []):
            if v["name"] == name:
                return v["id"]
        return self._master._current_voice_id

    def _test_voice(self) -> None:
        if self._master.on_voice_preview:
            self._master.on_voice_preview(self._get_selected_voice_id())

    # ---- callout language ----

    def _build_lang(self, parent, c: dict) -> None:
        self._section(parent, "CALLOUT LANGUAGE")
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 8))
        cur = self._master._current_lang
        for lang in ["EN", "PL", "DE", "FR", "ES", "RU"]:
            active = lang == cur
            btn = ctk.CTkButton(
                row, text=lang, width=48, height=28,
                fg_color=c["accent"] if active else c["panel"],
                hover_color=c["accent"],
                text_color=c["bg"] if active else c["text"],
                font=("Consolas", 8, "bold"), corner_radius=2,
                command=lambda l=lang: self._select_lang(l),
            )
            btn.pack(side="left", padx=(0, 4))
            self._lang_btns[lang] = btn

    def _select_lang(self, lang: str) -> None:
        c = self._master._c
        for name, btn in self._lang_btns.items():
            active = name == lang
            btn.configure(
                fg_color=c["accent"] if active else c["panel"],
                text_color=c["bg"] if active else c["text"],
            )
        self._master._current_lang = lang
        if self._master.on_lang_change:
            self._master.on_lang_change(lang)

    # ---- theme presets ----

    def _build_presets(self, parent, c: dict) -> None:
        self._section(parent, "THEME PRESET")
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 8))
        active = _active_preset(c)
        for name in list(THEMES.keys()) + ["CUSTOM"]:
            is_act = name == active
            btn = ctk.CTkButton(
                row, text=name, height=28, width=62,
                fg_color=c["accent"] if is_act else c["panel"],
                hover_color=c["accent"],
                text_color=c["bg"] if is_act else c["text"],
                font=("Consolas", 8, "bold"), corner_radius=2,
                command=lambda n=name: self._apply_preset(n),
            )
            btn.pack(side="left", padx=(0, 4))
            self._preset_btns[name] = btn

    def _highlight_preset(self, name: str) -> None:
        c = self._master._c
        for n, btn in self._preset_btns.items():
            active = n == name
            btn.configure(
                fg_color=c["accent"] if active else c["panel"],
                text_color=c["bg"] if active else c["text"],
            )

    def _apply_preset(self, name: str) -> None:
        if name == "CUSTOM":
            new_c = self._saved_custom or self._master._c
        else:
            new_c = dict(THEMES[name])
        self._master.apply_colors(new_c)
        self._rebuild(new_c, highlight=name)

    def _rebuild(self, colors: dict, highlight: str = "") -> None:
        """Destroy and recreate settings content with new colors."""
        if self._ns_monitor:
            try:
                from AppKit import NSEvent
                NSEvent.removeMonitor_(self._ns_monitor)
                self._ns_monitor = None
            except Exception:
                pass
        for w in self.winfo_children():
            w.destroy()
        self._rows = {}
        self._preset_btns = {}
        self._lang_btns = {}
        self._voice_combo = None
        self._lang_filter = None
        self._filtered_voices = []
        self._vol_var = None
        self._vol_lbl = None
        self.configure(fg_color=colors["bg"])
        self._build()
        if highlight:
            self._highlight_preset(highlight)

    def _apply_and_save(self) -> None:
        new_c = self._collect()
        self._master.apply_colors(new_c)
        self._saved_custom = new_c

        voice_id = self._master._current_voice_id
        if hasattr(self, "_voice_combo"):
            voice_id = self._get_selected_voice_id()
            self._master._current_voice_id = voice_id
            if self._master.on_voice_change:
                self._master.on_voice_change(voice_id)

        api_key = ""
        if hasattr(self, "_api_key_entry"):
            api_key = self._api_key_entry.get().strip()
            if api_key:
                import os as _os
                _os.environ["ANTHROPIC_API_KEY"] = api_key

        tts_volume = float(self._vol_var.get()) / 100.0 if hasattr(self, "_vol_var") else 1.0

        enemy_agents = self._get_enemy_agents()
        self._master._current_enemy_agents = enemy_agents
        if self._master.on_enemy_agents_change:
            self._master.on_enemy_agents_change(enemy_agents)

        map_override = ""
        if hasattr(self, "_map_var"):
            sel = self._map_var.get()
            map_override = "" if sel == "auto-detect" else sel.lower()
            if self._master.on_map_override_change:
                self._master.on_map_override_change(map_override or None)

        save_settings({
            "colors":            new_c,
            "voice_id":          voice_id,
            "callout_lang":      self._master._current_lang,
            "anthropic_api_key": api_key,
            "tts_volume":        tts_volume,
            "enemy_agents":      enemy_agents,
            "map_override":      map_override,
        })
        self.destroy()

    def _collect(self) -> dict:
        result = {}
        for key, row in self._rows.items():
            val = row.get()
            if len(val) == 7 and val.startswith("#"):
                try:
                    int(val[1:], 16)
                    result[key] = val
                    continue
                except ValueError:
                    pass
            result[key] = self._master._c[key]
        return result


# ---------------------------------------------------------------------------
# Main overlay window
# ---------------------------------------------------------------------------
class OverlayWindow(ctk.CTk):
    def __init__(self, fade_after: float = 5.0) -> None:
        super().__init__()
        self._muted  = False
        self.on_mute_change:            Optional[Callable[[bool],      None]] = None
        self.on_voice_change:           Optional[Callable[[str],       None]] = None
        self.on_voice_preview:          Optional[Callable[[str],       None]] = None
        self.on_lang_change:            Optional[Callable[[str],       None]] = None
        self.on_volume_change:          Optional[Callable[[float],     None]] = None
        self.on_enemy_agents_change:    Optional[Callable[[List[str]], None]] = None
        self.on_minimap_region_change:  Optional[Callable[[dict],      None]] = None
        self.on_map_override_change:    Optional[Callable[[Optional[str]], None]] = None
        self._voice_options:   list = []
        self._tracker          = EnemyTracker(fade_after=fade_after)
        self._drag:            dict = {}
        self._visible          = True
        self._abilities:       List[dict] = []
        self._settings_win:    Optional[SettingsWindow] = None
        self._panel_refs:      list = []
        self._sighting_history: List[dict] = []

        saved = load_settings()
        self._c = saved.get("colors", dict(THEMES["VALORANT"]))
        for k, v in THEMES["VALORANT"].items():
            self._c.setdefault(k, v)
        self._current_voice_id   = saved.get("voice_id", "")
        self._current_lang       = saved.get("callout_lang", "EN")
        self._current_enemy_agents: List[str] = saved.get("enemy_agents", [])
        self._callout_text_visible = saved.get("callout_text_visible", True)

        self._setup_window()
        self._build_ui()
        self._schedule_tick()

    def _setup_window(self) -> None:
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.95)
        self.configure(fg_color=self._c["bg"])
        self.geometry("264+50+50")
        self.resizable(False, False)
        self.bind("<F9>", lambda _: self._toggle_visible())

    def _build_ui(self) -> None:
        self._build_titlebar()
        self._build_map_panel()
        self._build_canvas_panel()
        self._build_enemy_panel()
        self._build_utility_panel()
        self._build_callout_panel()
        self._build_defuse_panel()
        self._build_ai_panel()
        self._build_bottombar()

    def _build_titlebar(self) -> None:
        c = self._c
        bar = ctk.CTkFrame(self, fg_color=c["title"], corner_radius=0, height=42)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        self._titlebar = bar

        stripe = ctk.CTkFrame(bar, fg_color=c["accent"], width=4, corner_radius=0)
        stripe.pack(side="left", fill="y")
        self._title_stripe = stripe

        icon = ctk.CTkLabel(bar, text="  ", text_color=c["accent"],
                            font=("Consolas", 18, "bold"))
        icon.pack(side="left", padx=(4, 2))
        self._title_icon = icon

        title = ctk.CTkLabel(bar, text="MINIMAP COACH", text_color=c["text"],
                             font=("Consolas", 10, "bold"))
        title.pack(side="left")
        self._title_lbl = title

        close_btn = FAIconButton(
            bar, char=_FA_XMARK, command=self.destroy,
            fg=c["dim"], bg=c["title"],
            hover_bg=c["enemy"], hover_fg=c["text"],
            size=42, icon_size=16,
        )
        close_btn.pack(side="right")
        self._close_btn = close_btn

        settings_btn = FAIconButton(
            bar, char=_FA_GEAR, command=self._open_settings,
            fg=c["dim"], bg=c["title"],
            hover_bg=c["panel"],
            size=42, icon_size=16,
        )
        settings_btn.pack(side="right")
        self._settings_btn = settings_btn

        self._compact = False
        self._compact_btn = ctk.CTkButton(
            bar, text="—", width=28, height=28,
            fg_color="transparent", text_color=c["dim"],
            hover_color=c["panel"], font=("Consolas", 14, "bold"),
            corner_radius=2, command=self._toggle_compact,
        )
        self._compact_btn.pack(side="right", padx=(0, 2))

        for w in (bar, icon, title):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>",     self._drag_move)

    def _toggle_compact(self) -> None:
        self._compact = not self._compact
        self._compact_btn.configure(text="+" if self._compact else "—")
        for outer, *_ in self._panel_refs:
            if self._compact:
                outer.pack_forget()
            else:
                outer.pack(fill="x", pady=(2, 0))

    def _panel(self, header: str) -> ctk.CTkFrame:
        c = self._c
        outer = ctk.CTkFrame(self, fg_color=c["panel"], corner_radius=0)
        outer.pack(fill="x", pady=(2, 0))
        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(fill="x")
        stripe = ctk.CTkFrame(inner, fg_color=c["accent"], width=4, corner_radius=0)
        stripe.pack(side="left", fill="y")
        content = ctk.CTkFrame(inner, fg_color="transparent")
        content.pack(side="left", fill="both", expand=True)
        hdr_lbl = ctk.CTkLabel(content, text=header,
                               text_color=c["accent"], font=("Consolas", 7, "bold"))
        hdr_lbl.pack(anchor="w", padx=10, pady=(5, 1))
        self._panel_refs.append((outer, stripe, hdr_lbl))
        return content

    def _build_map_panel(self) -> None:
        p = self._panel("MAP")
        self._map_lbl = ctk.CTkLabel(p, text="DETECTING...",
                                     text_color=self._c["text"],
                                     font=("Consolas", 15, "bold"))
        self._map_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _build_canvas_panel(self) -> None:
        p = self._panel("MINIMAP")
        self._canvas = tk.Canvas(p, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg=self._c["canvas_bg"], highlightthickness=1,
                                 highlightbackground=self._c["dim"])
        self._canvas.pack(padx=10, pady=(0, 8), anchor="w")
        self._draw_canvas([])

    def _build_enemy_panel(self) -> None:
        p = self._panel("ENEMIES SPOTTED")
        top_row = ctk.CTkFrame(p, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(0, 4))
        self._enemy_count = ctk.CTkLabel(top_row, text="0",
                                         text_color=self._c["dim"], font=FONT_BIG)
        self._enemy_count.pack(side="left")
        dots_frame = ctk.CTkFrame(top_row, fg_color="transparent")
        dots_frame.pack(side="left", padx=(8, 0))
        self._enemy_dots: List[ctk.CTkLabel] = []
        for _ in range(5):
            d = ctk.CTkLabel(dots_frame, text="-", text_color=self._c["dim"],
                             font=("Consolas", 12, "bold"))
            d.pack(side="left", padx=2)
            self._enemy_dots.append(d)
        self._enemy_div = ctk.CTkFrame(p, fg_color=self._c["dim"],
                                       height=1, corner_radius=0)
        self._enemy_div.pack(fill="x", padx=10, pady=(0, 4))
        self._hist_hdr = ctk.CTkLabel(p, text="LAST SIGHTED",
                                      text_color=self._c["dim"],
                                      font=("Consolas", 7, "bold"))
        self._hist_hdr.pack(anchor="w", padx=10, pady=(0, 2))
        self._hist_rows: List[Tuple[ctk.CTkLabel, ctk.CTkLabel]] = []
        for _ in range(5):
            rf = ctk.CTkFrame(p, fg_color="transparent", height=16)
            rf.pack(fill="x", padx=10, pady=0)
            rf.pack_propagate(False)
            ll = ctk.CTkLabel(rf, text="", text_color=self._c["dim"],
                              font=("Consolas", 9), anchor="w", width=110)
            ll.pack(side="left")
            tl = ctk.CTkLabel(rf, text="", text_color=self._c["dim"],
                              font=("Consolas", 8), anchor="e")
            tl.pack(side="right")
            self._hist_rows.append((ll, tl))
        ctk.CTkLabel(p, text="", height=4).pack()

    def _build_utility_panel(self) -> None:
        p = self._panel("UTILITY")
        self._utility_outer = self._panel_refs[-1][0]
        self._util_rows: List[ctk.CTkLabel] = []
        for _ in range(4):
            lbl = ctk.CTkLabel(p, text="", text_color=self._c["dim"],
                               font=("Consolas", 9), anchor="w")
            lbl.pack(fill="x", padx=10, pady=0)
            self._util_rows.append(lbl)
        ctk.CTkLabel(p, text="", height=2).pack()

    def _build_callout_panel(self) -> None:
        p = self._panel("CALLOUT")
        self._callout_outer = self._panel_refs[-1][0]
        self._callout_lbl = ctk.CTkLabel(p, text="listening...",
                                         text_color=self._c["text"],
                                         font=FONT_MONO, wraplength=218, justify="left")
        self._callout_lbl.pack(anchor="w", padx=10, pady=(0, 8))
        if not self._callout_text_visible:
            self._callout_outer.pack_forget()

    # ── Defuse bar geometry constants ────────────────────────────────────────
    _DB_W   = 220    # canvas width  (px)
    _DB_H   = 20     # canvas height (px)
    _DB_PAD = 10     # horizontal padding inside panel

    def _build_defuse_panel(self) -> None:
        """
        Valorant-style defuse progress bar.
        Hidden until defuse sound is detected.

        Layout (matches Valorant HUD proportions):
          ┌─────────────────────────────┐
          │  DEFUSING          ~43%     │  ← label row
          │ ▐████████▌·········▏        │  ← canvas bar
          │          ↑ 50% tick         │
          └─────────────────────────────┘
        """
        c = self._c
        p = self._panel("DEFUSING")
        self._defuse_panel = p

        # Single-row label: left="DEFUSING", right="~XX%"
        lbl_row = ctk.CTkFrame(p, fg_color="transparent")
        lbl_row.pack(fill="x", padx=self._DB_PAD, pady=(0, 3))

        self._defuse_title = ctk.CTkLabel(
            lbl_row, text="DEFUSING", text_color=c["dim"],
            font=("Consolas", 8, "bold"), anchor="w",
        )
        self._defuse_title.pack(side="left")

        self._defuse_pct_lbl = ctk.CTkLabel(
            lbl_row, text="", text_color=c["safe"],
            font=("Consolas", 9, "bold"), anchor="e",
        )
        self._defuse_pct_lbl.pack(side="right")

        # Canvas bar -- drawn manually for Valorant-style mid-tick
        self._defuse_cv = tk.Canvas(
            p, width=self._DB_W, height=self._DB_H,
            bg=c["bg"], highlightthickness=0,
        )
        self._defuse_cv.pack(padx=self._DB_PAD, pady=(0, 8))

        self._defuse_visible = False
        p.pack_forget()

    def _redraw_defuse_bar(self, pct: float) -> None:
        """Redraw the canvas bar for the given fraction (0.0-1.0)."""
        c   = self._c
        cv  = self._defuse_cv
        W, H = self._DB_W, self._DB_H
        mid  = W // 2
        fill_w = int(pct * W)

        # Bar fill color: yellow-green → amber → red
        if pct >= 0.85:
            bar_color = c["enemy"]          # red
        elif pct >= 0.5:
            bar_color = "#f59e0b"           # amber
        else:
            bar_color = "#b8ca00"           # Valorant yellow-green

        cv.delete("all")

        # Trough (full width)
        tr = H // 3
        cv.create_rectangle(0, tr, W, H - tr,
                            fill="#1c1f27", outline="")

        # Filled portion
        if fill_w > 0:
            cv.create_rectangle(0, tr, fill_w, H - tr,
                                fill=bar_color, outline="")

        # 50% tick -- two-tone: bright line spanning full bar height
        tick_color = "#ffffff" if pct < 0.5 else "#888888"
        cv.create_line(mid, 0, mid, H, fill=tick_color, width=1)

        # Small diamond at 50% mark (Valorant-style mid marker)
        d = 4
        cv.create_polygon(
            mid, H // 2 - d,
            mid + d, H // 2,
            mid, H // 2 + d,
            mid - d, H // 2,
            fill=tick_color, outline="",
        )

    def _build_ai_panel(self) -> None:
        c = self._c
        p = self._panel("AI INSIGHT")
        self._ai_panel_ref = p   # used by update_defuse_progress for pack ordering
        self._ai_lbl = ctk.CTkLabel(p, text="", text_color=c["dim"],
                                    font=FONT_MONO, wraplength=218, justify="left")
        self._ai_lbl.pack(anchor="w", padx=10, pady=(0, 4))

        fb_row = ctk.CTkFrame(p, fg_color="transparent")
        fb_row.pack(anchor="w", padx=10, pady=(0, 6))
        self._fb_up = ctk.CTkButton(
            fb_row, text="▲", width=24, height=16,
            fg_color=c["panel"], hover_color=c["safe"],
            text_color=c["dim"],
            font=("Consolas", 8, "bold"), corner_radius=2,
            command=self._on_feedback_up,
        )
        self._fb_up.pack(side="left", padx=(0, 3))
        self._fb_dn = ctk.CTkButton(
            fb_row, text="▼", width=24, height=16,
            fg_color=c["panel"], hover_color=c["enemy"],
            text_color=c["dim"],
            font=("Consolas", 8, "bold"), corner_radius=2,
            command=self._on_feedback_dn,
        )
        self._fb_dn.pack(side="left")
        self._current_ai_ts: int = 0
        self.on_feedback = None   # Callable[[int, bool], None] set by coach

    def _build_bottombar(self) -> None:
        c = self._c
        bar = ctk.CTkFrame(self, fg_color=c["title"], corner_radius=0, height=40)
        bar.pack(fill="x", pady=(2, 0))
        bar.pack_propagate(False)
        self._bottombar = bar
        self._mute_btn = ctk.CTkButton(
            bar, text="AUDIO ON", width=90, height=26,
            fg_color=c["accent"], text_color=c["bg"],
            hover_color=c["safe"], font=("Consolas", 8, "bold"),
            corner_radius=2, command=self._toggle_mute,
        )
        self._mute_btn.pack(side="left", padx=(8, 0), pady=7)
        txt_on = self._callout_text_visible
        self._callout_btn = ctk.CTkButton(
            bar,
            text="TEXT ON" if txt_on else "TEXT OFF",
            width=72, height=26,
            fg_color=c["accent"] if txt_on else c["panel"],
            text_color=c["bg"] if txt_on else c["dim"],
            hover_color=c["safe"], font=("Consolas", 8, "bold"),
            corner_radius=2, command=self._toggle_callout_panel,
        )
        self._callout_btn.pack(side="left", padx=(4, 0), pady=7)
        ctk.CTkLabel(bar, text="F9  HIDE", text_color=c["dim"],
                     font=("Consolas", 7, "bold")).pack(side="right", padx=10)

    # -----------------------------------------------------------------------
    # Color management
    # -----------------------------------------------------------------------
    def apply_colors(self, new_c: dict) -> None:
        self._c = new_c
        c = new_c
        self.configure(fg_color=c["bg"])
        self._titlebar.configure(fg_color=c["title"])
        self._title_stripe.configure(fg_color=c["accent"])
        self._title_icon.configure(text_color=c["accent"])
        self._title_lbl.configure(text_color=c["text"])
        self._close_btn.recolor(fg=c["dim"], bg=c["title"], hover_bg=c["enemy"], hover_fg=c["text"])
        self._settings_btn.recolor(fg=c["dim"], bg=c["title"], hover_bg=c["panel"], hover_fg=c["dim"])
        for outer, stripe, hdr_lbl in self._panel_refs:
            if not outer.winfo_exists():
                continue
            outer.configure(fg_color=c["panel"])
            stripe.configure(fg_color=c["accent"])
            hdr_lbl.configure(text_color=c["accent"])
        self._map_lbl.configure(text_color=c["text"])
        self._canvas.configure(bg=c["canvas_bg"], highlightbackground=c["dim"])
        self._enemy_count.configure(text_color=c["dim"])
        for d in self._enemy_dots:
            d.configure(text_color=c["dim"])
        self._enemy_div.configure(fg_color=c["dim"])
        self._hist_hdr.configure(text_color=c["dim"])
        for ll, tl in self._hist_rows:
            ll.configure(text_color=c["dim"])
            tl.configure(text_color=c["dim"])
        for lbl in self._util_rows:
            lbl.configure(text_color=c["dim"])
        self._callout_lbl.configure(text_color=c["text"])
        self._ai_lbl.configure(text_color=c["dim"])
        self._fb_up.configure(fg_color=c["panel"], text_color=c["dim"])
        self._fb_dn.configure(fg_color=c["panel"], text_color=c["dim"])
        self._defuse_title.configure(text_color=c["dim"])
        self._defuse_cv.configure(bg=c["canvas_bg"])
        self._bottombar.configure(fg_color=c["title"])
        self._mute_btn.configure(
            fg_color=c["accent"] if not self._muted else c["panel"],
            text_color=c["bg"]   if not self._muted else c["dim"],
            hover_color=c["safe"],
        )
        self._callout_btn.configure(
            fg_color=c["accent"] if self._callout_text_visible else c["panel"],
            text_color=c["bg"]   if self._callout_text_visible else c["dim"],
            hover_color=c["safe"],
        )
        self._draw_canvas(self._tracker.tick(time.time()))

    def set_voice_options(self, voices: list, current_id: str) -> None:
        self._voice_options = voices
        if not self._current_voice_id:
            self._current_voice_id = current_id

    # -----------------------------------------------------------------------
    # Sighting history
    # -----------------------------------------------------------------------
    def _update_sighting_history(self, positions: list) -> None:
        now = time.time()
        for pos in positions:
            best, best_d = None, _SIGHTING_RADIUS
            for e in self._sighting_history:
                d = ((pos[0]-e["pos"][0])**2 + (pos[1]-e["pos"][1])**2)**0.5
                if d < best_d:
                    best_d, best = d, e
            if best:
                best["pos"]       = pos
                best["last_seen"] = now
                best["label"]     = _pos_label(pos[0], pos[1])
            else:
                self._sighting_history.append(
                    {"pos": pos, "last_seen": now,
                     "label": _pos_label(pos[0], pos[1])}
                )
        self._sighting_history.sort(key=lambda e: e["last_seen"], reverse=True)
        self._sighting_history = self._sighting_history[:5]

    def _refresh_history(self) -> None:
        now = time.time()
        c = self._c
        for i, (ll, tl) in enumerate(self._hist_rows):
            if i < len(self._sighting_history):
                e       = self._sighting_history[i]
                elapsed = now - e["last_seen"]
                color   = c["enemy"] if elapsed < 3 else c["dim"]
                ll.configure(text=e['label'], text_color=color)
                tl.configure(text=_ago(e["last_seen"]), text_color=color)
            else:
                ll.configure(text="")
                tl.configure(text="")

    # -----------------------------------------------------------------------
    # Drag / controls
    # -----------------------------------------------------------------------
    def _drag_start(self, event) -> None:
        self._drag = {"x": event.x, "y": event.y}

    def _drag_move(self, event) -> None:
        x = self.winfo_x() + (event.x - self._drag["x"])
        y = self.winfo_y() + (event.y - self._drag["y"])
        self.geometry(f"+{x}+{y}")

    def _toggle_mute(self) -> None:
        c = self._c
        self._muted = not self._muted
        if self._muted:
            self._mute_btn.configure(text="MUTED", fg_color=c["panel"],
                                     text_color=c["dim"])
        else:
            self._mute_btn.configure(text="AUDIO ON", fg_color=c["accent"],
                                     text_color=c["bg"])
        if self.on_mute_change:
            self.on_mute_change(self._muted)

    def _toggle_callout_panel(self) -> None:
        c = self._c
        self._callout_text_visible = not self._callout_text_visible
        if self._callout_text_visible:
            self._callout_outer.pack(fill="x", pady=(2, 0), after=self._utility_outer)
            self._callout_btn.configure(text="TEXT ON", fg_color=c["accent"],
                                        text_color=c["bg"])
        else:
            self._callout_outer.pack_forget()
            self._callout_btn.configure(text="TEXT OFF", fg_color=c["panel"],
                                        text_color=c["dim"])
        save_settings({"callout_text_visible": self._callout_text_visible})

    def _toggle_visible(self) -> None:
        if self._visible:
            self.withdraw()
        else:
            self.deiconify()
        self._visible = not self._visible

    def _open_settings(self) -> None:
        if self._settings_win and self._settings_win.winfo_exists():
            self._settings_win.focus()
            return
        self._settings_win = SettingsWindow(self)

    # -----------------------------------------------------------------------
    # Tick loop
    # -----------------------------------------------------------------------
    def _schedule_tick(self) -> None:
        if not self.winfo_exists():
            return
        try:
            live = self._tracker.tick(time.time())
            self._draw_canvas(live)
            self._refresh_history()
        except Exception:
            pass
        self.after(100, self._schedule_tick)

    def _draw_canvas(self, enemies: List[TrackedEnemy]) -> None:
        cr = self._c
        cv = self._canvas
        s  = CANVAS_SIZE
        cv.delete("all")
        step = s // 4
        for i in range(0, s + 1, step):
            cv.create_line(i, 0, i, s, fill=cr["grid"], width=1)
            cv.create_line(0, i, s, i, fill=cr["grid"], width=1)
        m = s // 2
        cv.create_line(m-10, m, m+10, m, fill=cr["dim"], width=1)
        cv.create_line(m, m-10, m, m+10, fill=cr["dim"], width=1)
        for ab in self._abilities:
            ax    = int(ab["position"][0] * s)
            ay    = int(ab["position"][1] * s)
            color = ab.get("color", "#ffffff")
            cv.create_oval(ax-4, ay-4, ax+4, ay+4, fill=color, outline="")
            cv.create_text(ax+6, ay, text=ab["display"][:3].upper(),
                           fill=color, font=("Consolas", 6), anchor="w")
        cv.create_polygon(m, m-7, m+7, m, m, m+7, m-7, m,
                          fill=cr["accent"], outline="")
        cv.create_oval(m-2, m-2, m+2, m+2, fill=cr["canvas_bg"], outline="")
        for e in enemies:
            ex  = int(e.position[0] * s)
            ey  = int(e.position[1] * s)
            iv  = int(0xe8 * e.alpha)
            dv  = int(0x40 * e.alpha)
            col = f"#{iv:02x}{dv:02x}{dv:02x}"
            r   = 6 if e.alpha > 0.6 else 4
            cv.create_oval(ex-r, ey-r, ex+r, ey+r, fill=col, outline="")

    # -----------------------------------------------------------------------
    # Thread-safe public API
    # -----------------------------------------------------------------------
    def update_map(self, name: str) -> None:
        self._map_lbl.configure(text=name.upper())

    def update_enemies(self, count: int, positions: list) -> None:
        c = self._c
        color = c["enemy"] if count > 0 else c["dim"]
        self._enemy_count.configure(text=str(count), text_color=color)
        for i, dot in enumerate(self._enemy_dots):
            dot.configure(text_color=color if i < count else c["dim"])
        self._tracker.update(positions)
        if positions:
            self._update_sighting_history(positions)

    def update_callout(self, text: str) -> None:
        self._callout_lbl.configure(text=text, text_color=self._c["text"])

    def update_defuse_progress(self, pct: float) -> None:
        """
        Show/update the Valorant-style defuse bar.
        pct: 0.0-1.0  (hypothetical fraction based on detected defuse hum)
        """
        c = self._c
        if not self._defuse_visible:
            self._defuse_panel.pack(fill="x", before=self._ai_panel_ref)
            self._defuse_visible = True

        if pct >= 1.0:
            self.hide_defuse_progress()
            return

        pct_int = int(pct * 100)
        if pct >= 0.85:
            pct_label = f"~{pct_int}%  PEEK!"
            lbl_color = c["enemy"]
            title_color = c["enemy"]
        elif pct >= 0.5:
            pct_label = f"~{pct_int}%"
            lbl_color = "#f59e0b"
            title_color = "#f59e0b"
        else:
            pct_label = f"~{pct_int}%"
            lbl_color = "#b8ca00"
            title_color = c["dim"]

        self._defuse_title.configure(text_color=title_color)
        self._defuse_pct_lbl.configure(text=pct_label, text_color=lbl_color)
        self._redraw_defuse_bar(pct)

    def hide_defuse_progress(self) -> None:
        if self._defuse_visible:
            self._defuse_panel.pack_forget()
            self._defuse_visible = False

    def update_ai(self, text: str, sample_ts: int = 0) -> None:
        self._ai_lbl.configure(text=text)
        self._current_ai_ts = sample_ts
        c = self._c
        self._fb_up.configure(fg_color=c["panel"], state="normal", text_color=c["dim"])
        self._fb_dn.configure(fg_color=c["panel"], state="normal", text_color=c["dim"])

    def _on_feedback_up(self) -> None:
        if self.on_feedback and self._current_ai_ts:
            self.on_feedback(self._current_ai_ts, True)
        self._fb_up.configure(fg_color=self._c["safe"], state="disabled",
                               text_color=self._c["bg"])
        self._fb_dn.configure(state="disabled")

    def _on_feedback_dn(self) -> None:
        if self.on_feedback and self._current_ai_ts:
            self.on_feedback(self._current_ai_ts, False)
        self._fb_dn.configure(fg_color=self._c["enemy"], state="disabled",
                               text_color=self._c["bg"])
        self._fb_up.configure(state="disabled")

    def update_utility(self, abilities: List[dict]) -> None:
        self._abilities = abilities
        for i, row in enumerate(self._util_rows):
            if i < len(abilities):
                ab    = abilities[i]
                color = ab.get("color", self._c["dim"])
                row.configure(text=ab['display'], text_color=color)
            else:
                row.configure(text="")
