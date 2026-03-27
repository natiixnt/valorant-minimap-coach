"""
Esports-style always-on-top overlay with live theme customization,
persistent user settings, in-app color pickers, and enemy sighting history.
"""
import json
import os
import tkinter as tk
import tkinter.colorchooser as colorchooser
import time
from typing import Callable, List, Optional, Tuple

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from src.vision.enemy_tracker import EnemyTracker, TrackedEnemy

ctk.set_appearance_mode("dark")


def _make_app_icon(size: int = 128) -> ImageTk.PhotoImage:
    """Generate a Valorant-style crosshair icon in memory."""
    accent = (232, 64, 87, 255)   # #e84057
    bg     = (13, 15, 20, 255)    # #0d0f14

    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded background
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 6, fill=bg)

    cx = cy = size // 2
    gap    = size // 7
    length = size // 4
    thick  = max(3, size // 22)
    half   = thick // 2

    # Four crosshair lines
    draw.rectangle([cx - half, cy - gap - length, cx + half, cy - gap], fill=accent)
    draw.rectangle([cx - half, cy + gap,          cx + half, cy + gap + length], fill=accent)
    draw.rectangle([cx - gap - length, cy - half, cx - gap, cy + half], fill=accent)
    draw.rectangle([cx + gap, cy - half,          cx + gap + length, cy + half], fill=accent)

    # Small center dot
    dot = half + 1
    draw.rectangle([cx - dot, cy - dot, cx + dot, cy + dot], fill=accent)

    return ImageTk.PhotoImage(img)

# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "user_settings.json")
_SIGHTING_MATCH_RADIUS = 0.15


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Preset themes
# ---------------------------------------------------------------------------
THEMES = {
    "VALORANT": {
        "bg":        "#0d0f14",
        "panel":     "#13161e",
        "title":     "#090b0f",
        "accent":    "#e84057",
        "enemy":     "#e84057",
        "safe":      "#3ec17c",
        "text":      "#f0f0f0",
        "dim":       "#4a5568",
        "canvas_bg": "#07090d",
        "grid":      "#171b27",
    },
    "CYBER": {
        "bg":        "#0f1923",
        "panel":     "#1a2332",
        "title":     "#111827",
        "accent":    "#5bc4e8",
        "enemy":     "#e84057",
        "safe":      "#3ec17c",
        "text":      "#ecf0f1",
        "dim":       "#7f8c9b",
        "canvas_bg": "#0a1018",
        "grid":      "#1c2b3a",
    },
    "MATRIX": {
        "bg":        "#050f05",
        "panel":     "#081408",
        "title":     "#030803",
        "accent":    "#00ff41",
        "enemy":     "#ff4444",
        "safe":      "#00ff41",
        "text":      "#c8ffc8",
        "dim":       "#2e5e2e",
        "canvas_bg": "#020602",
        "grid":      "#081608",
    },
    "PHANTOM": {
        "bg":        "#120e1f",
        "panel":     "#1b1630",
        "title":     "#0d0a17",
        "accent":    "#a855f7",
        "enemy":     "#f43f5e",
        "safe":      "#22d3ee",
        "text":      "#e2e8f0",
        "dim":       "#4a4060",
        "canvas_bg": "#080611",
        "grid":      "#18102e",
    },
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
# Helpers
# ---------------------------------------------------------------------------
def _ago(ts: float) -> str:
    elapsed = max(0, time.time() - ts)
    if elapsed < 60:
        return f"{int(elapsed)}s ago"
    return f"{int(elapsed / 60)}m ago"


def _pos_to_label(x: float, y: float) -> str:
    col = "LEFT" if x < 0.33 else ("RIGHT" if x > 0.66 else "MID")
    row = "TOP"  if y < 0.33 else ("BOT"  if y > 0.66 else "MID")
    if row == "MID" and col == "MID":
        return "CENTER"
    if row == "MID":
        return col
    if col == "MID":
        return row
    return f"{row}-{col}"


def _active_preset(colors: dict) -> str:
    """Return preset name if colors exactly match a theme, else 'CUSTOM'."""
    for name, theme in THEMES.items():
        if all(colors.get(k) == v for k, v in theme.items()):
            return name
    return "CUSTOM"


# ---------------------------------------------------------------------------
# Color picker icon (4-quadrant canvas, no emoji)
# ---------------------------------------------------------------------------
class PickerIcon(tk.Canvas):
    SIZE = 22

    def __init__(self, parent, command: Callable, bg: str):
        s = self.SIZE
        super().__init__(parent, width=s, height=s, highlightthickness=0,
                         cursor="hand2", bg=bg)
        h = s // 2
        self.create_rectangle(0,  0,  h, h, fill="#e84057", outline="")
        self.create_rectangle(h,  0,  s, h, fill="#3ec17c", outline="")
        self.create_rectangle(0,  h,  h, s, fill="#5bc4e8", outline="")
        self.create_rectangle(h,  h,  s, s, fill="#f59e0b", outline="")
        self.bind("<Button-1>", lambda _: command())


# ---------------------------------------------------------------------------
# Color row widget
# ---------------------------------------------------------------------------
class ColorRow(ctk.CTkFrame):
    def __init__(self, parent, key: str, label: str, value: str, colors: dict, **kw):
        super().__init__(parent, fg_color=colors["panel"], corner_radius=4, height=38, **kw)
        self.pack_propagate(False)
        self._key = key
        self._on_change_cb: Optional[Callable] = None

        # Swatch
        self._swatch = tk.Canvas(self, width=22, height=22, highlightthickness=1,
                                  highlightbackground=colors["dim"],
                                  cursor="hand2")
        self._swatch.configure(bg=value)
        self._swatch.pack(side="left", padx=(10, 6), pady=8)
        self._swatch.bind("<Button-1>", self._open_picker)

        # Label
        ctk.CTkLabel(self, text=label, text_color=colors["dim"],
                     font=("Consolas", 8, "bold"), width=82, anchor="w").pack(
            side="left", padx=(0, 4))

        # Picker icon button (right side, packed first so it stays right)
        self._icon = PickerIcon(self, command=self._open_picker, bg=colors["panel"])
        self._icon.pack(side="right", padx=(0, 8), pady=8)

        # Hex entry
        self._entry = ctk.CTkEntry(self, width=86, height=24,
                                   fg_color=colors["bg"],
                                   text_color=colors["text"],
                                   border_color=colors["dim"], border_width=1,
                                   font=("Consolas", 9), corner_radius=3)
        self._entry.insert(0, value)
        self._entry.pack(side="right", padx=(0, 6), pady=7)
        self._entry.bind("<KeyRelease>", self._on_hex_typed)

    def get(self) -> str:
        return self._entry.get().strip()

    def set(self, value: str) -> None:
        self._entry.delete(0, "end")
        self._entry.insert(0, value)
        self._update_swatch(value)

    def set_on_change(self, cb: Callable) -> None:
        self._on_change_cb = cb

    def _open_picker(self, _event=None) -> None:
        result = colorchooser.askcolor(color=self.get(), title=f"Pick — {self._key}")
        if result and result[1]:
            self.set(result[1])
            if self._on_change_cb:
                self._on_change_cb()

    def _on_hex_typed(self, _event=None) -> None:
        val = self.get()
        if len(val) == 7 and val.startswith("#"):
            try:
                int(val[1:], 16)
                self._update_swatch(val)
                if self._on_change_cb:
                    self._on_change_cb()
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
        self.geometry("390x580")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.configure(fg_color=master._c["bg"])
        self._rows: dict = {}
        self._preset_btns: dict = {}
        # Track what the saved "custom" state is
        self._saved_custom = load_settings().get("colors", None)
        self._build()

    def _build(self) -> None:
        c = self._master._c

        # Titlebar
        hdr = ctk.CTkFrame(self, fg_color=c["title"], corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkFrame(hdr, fg_color=c["accent"], width=4, corner_radius=0).pack(
            side="left", fill="y")
        ctk.CTkLabel(hdr, text=" ⚙", text_color=c["accent"],
                     font=("Consolas", 36, "bold")).pack(side="left", padx=(8, 4))
        ctk.CTkLabel(hdr, text="SETTINGS", text_color=c["accent"],
                     font=("Consolas", 12, "bold")).pack(side="left")

        scroll = ctk.CTkScrollableFrame(self, fg_color=c["bg"], corner_radius=0)
        scroll.pack(fill="both", expand=True)

        # -- Voice selection --
        self._build_voice_section(scroll, c)

        ctk.CTkFrame(scroll, fg_color=c["dim"], height=1, corner_radius=0).pack(
            fill="x", padx=14, pady=(4, 12))

        # -- Theme presets --
        ctk.CTkLabel(scroll, text="THEME PRESET", text_color=c["accent"],
                     font=("Consolas", 9, "bold")).pack(anchor="w", padx=14, pady=(14, 6))

        preset_row = ctk.CTkFrame(scroll, fg_color="transparent")
        preset_row.pack(fill="x", padx=14, pady=(0, 12))

        active = _active_preset(c)
        all_names = list(THEMES.keys()) + ["CUSTOM"]

        for name in all_names:
            is_active = (name == active)
            btn = ctk.CTkButton(
                preset_row, text=name, height=30, width=60,
                fg_color=c["accent"] if is_active else c["panel"],
                hover_color=c["accent"],
                text_color=c["bg"] if is_active else c["text"],
                font=("Consolas", 8, "bold"),
                corner_radius=3,
                command=lambda n=name: self._apply_preset(n),
            )
            btn.pack(side="left", padx=(0, 4))
            self._preset_btns[name] = btn

        # Divider
        ctk.CTkFrame(scroll, fg_color=c["dim"], height=1, corner_radius=0).pack(
            fill="x", padx=14, pady=(0, 12))

        # -- Custom colors --
        ctk.CTkLabel(scroll,
                     text="CUSTOM COLORS  — click swatch or icon to pick",
                     text_color=c["accent"], font=("Consolas", 9, "bold")).pack(
            anchor="w", padx=14, pady=(0, 6))

        for key, label in COLOR_FIELDS:
            row = ColorRow(scroll, key=key, label=label, value=c[key], colors=c)
            row.pack(fill="x", padx=14, pady=2)
            row.set_on_change(lambda: self._mark_custom())
            self._rows[key] = row

        ctk.CTkLabel(scroll, text="", height=6).pack()

        # Bottom bar
        bottom = ctk.CTkFrame(self, fg_color=c["title"], corner_radius=0, height=50)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        ctk.CTkButton(
            bottom, text="APPLY & SAVE", width=120, height=30,
            fg_color=c["accent"], text_color=c["bg"],
            hover_color=c["safe"], font=("Consolas", 9, "bold"),
            corner_radius=3, command=self._apply_and_save,
        ).pack(side="right", padx=12, pady=10)

        ctk.CTkButton(
            bottom, text="RESET", width=80, height=30,
            fg_color=c["panel"], text_color=c["dim"],
            hover_color=c["dim"], font=("Consolas", 9, "bold"),
            corner_radius=3, command=lambda: self._apply_preset("VALORANT"),
        ).pack(side="right", padx=(0, 4), pady=10)

    def _build_voice_section(self, parent, c: dict) -> None:
        ctk.CTkLabel(parent, text="CALLOUT VOICE", text_color=c["accent"],
                     font=("Consolas", 10, "bold")).pack(anchor="w", padx=14, pady=(14, 8))

        voices = self._master._voice_options
        if not voices:
            ctk.CTkLabel(parent, text="No voices available", text_color=c["dim"],
                         font=("Consolas", 9)).pack(anchor="w", padx=14, pady=(0, 8))
            return

        # Language filter
        filter_row = ctk.CTkFrame(parent, fg_color="transparent")
        filter_row.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkLabel(filter_row, text="FILTER:", text_color=c["dim"],
                     font=("Consolas", 9, "bold")).pack(side="left", padx=(0, 8))

        self._lang_filter = ctk.StringVar(value="EN")
        for label, val in [("ALL", "ALL"), ("EN", "EN"), ("PL", "PL")]:
            rb = ctk.CTkRadioButton(
                filter_row, text=label, variable=self._lang_filter, value=val,
                text_color=c["text"], fg_color=c["accent"],
                hover_color=c["accent"], border_color=c["dim"],
                font=("Consolas", 9, "bold"),
                command=self._refresh_voice_list,
            )
            rb.pack(side="left", padx=(0, 14))

        # Combobox row
        combo_row = ctk.CTkFrame(parent, fg_color="transparent")
        combo_row.pack(fill="x", padx=14, pady=(0, 10))

        # Build initial filtered list
        current_id = self._master._current_voice_id
        filtered = self._filter_voices(voices, "EN")
        names = [v["name"] for v in filtered]
        self._filtered_voices = filtered

        # Determine which name is currently selected
        current_name = next(
            (v["name"] for v in voices if v["id"] == current_id), names[0] if names else ""
        )

        self._voice_combo = ctk.CTkComboBox(
            combo_row, values=names, width=238, height=30,
            fg_color=c["panel"], button_color=c["accent"],
            button_hover_color=c["safe"], border_color=c["dim"],
            text_color=c["text"], font=("Consolas", 9),
            dropdown_fg_color=c["panel"], dropdown_text_color=c["text"],
            dropdown_hover_color=c["accent"],
            corner_radius=3,
        )
        if current_name in names:
            self._voice_combo.set(current_name)
        elif names:
            self._voice_combo.set(names[0])
        self._voice_combo.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            combo_row, text="TEST", width=66, height=30,
            fg_color=c["panel"], text_color=c["accent"],
            hover_color=c["accent"], text_color_hover=c["bg"],
            font=("Consolas", 9, "bold"),
            corner_radius=3, command=self._test_voice,
        ).pack(side="left")

    _PREFERRED_EN = ["samantha", "alex", "daniel", "karen", "moira"]

    def _filter_voices(self, voices: list, lang: str) -> list:
        if lang == "ALL":
            return voices
        prefix = lang.lower()
        filtered = [v for v in voices if v["lang"].lower().startswith(prefix)] or voices
        if lang == "EN":
            preferred = [v for v in filtered
                         if v["name"].lower() in self._PREFERRED_EN]
            ordered = sorted(preferred,
                             key=lambda v: self._PREFERRED_EN.index(v["name"].lower()))
            return ordered if ordered else filtered[:5]
        return filtered

    def _refresh_voice_list(self) -> None:
        lang = self._lang_filter.get()
        filtered = self._filter_voices(self._master._voice_options, lang)
        self._filtered_voices = filtered
        names = [v["name"] for v in filtered]
        self._voice_combo.configure(values=names)
        if names:
            self._voice_combo.set(names[0])

    def _get_selected_voice_id(self) -> str:
        name = self._voice_combo.get()
        for v in self._filtered_voices:
            if v["name"] == name:
                return v["id"]
        return self._master._current_voice_id

    def _test_voice(self) -> None:
        vid = self._get_selected_voice_id()
        if self._master.on_voice_preview:
            self._master.on_voice_preview(vid)

    def _mark_custom(self) -> None:
        """Highlight CUSTOM button when user edits any color manually."""
        c = self._master._c
        for name, btn in self._preset_btns.items():
            is_active = (name == "CUSTOM")
            btn.configure(
                fg_color=c["accent"] if is_active else c["panel"],
                text_color=c["bg"] if is_active else c["text"],
            )

    def _highlight_preset(self, active_name: str) -> None:
        c = self._master._c
        for name, btn in self._preset_btns.items():
            is_active = (name == active_name)
            btn.configure(
                fg_color=c["accent"] if is_active else c["panel"],
                text_color=c["bg"] if is_active else c["text"],
            )

    def _apply_preset(self, name: str) -> None:
        if name == "CUSTOM":
            # Load saved custom or stay as-is
            saved = self._saved_custom
            if saved:
                for key, row in self._rows.items():
                    row.set(saved.get(key, self._master._c[key]))
            self._highlight_preset("CUSTOM")
            self._push_to_overlay()
        else:
            theme = THEMES[name]
            for key, row in self._rows.items():
                row.set(theme[key])
            self._highlight_preset(name)
            self._push_to_overlay()

    def _push_to_overlay(self) -> None:
        """Apply current row values to the overlay without saving."""
        new_c = self._collect()
        self._master.apply_colors(new_c)

    def _apply_and_save(self) -> None:
        new_c = self._collect()
        self._master.apply_colors(new_c)
        self._saved_custom = new_c

        # Save voice if section was built
        voice_id = None
        if hasattr(self, "_voice_combo"):
            voice_id = self._get_selected_voice_id()
            self._master._current_voice_id = voice_id
            if self._master.on_voice_change:
                self._master.on_voice_change(voice_id)

        data: dict = {"colors": new_c}
        if voice_id:
            data["voice_id"] = voice_id
        save_settings(data)
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
        self._muted = False
        self.on_mute_change: Optional[Callable[[bool], None]] = None
        self.on_voice_change: Optional[Callable[[str], None]] = None
        self.on_voice_preview: Optional[Callable[[str], None]] = None
        self._voice_options: list = []
        self._current_voice_id: str = load_settings().get("voice_id", "")
        self._tracker = EnemyTracker(fade_after=fade_after)
        self._drag: dict = {}
        self._visible = True
        self._abilities: List[dict] = []
        self._settings_win: Optional[SettingsWindow] = None
        self._panel_refs: list = []

        # Sighting history: list of {"pos": (x,y), "last_seen": float, "label": str}
        self._sighting_history: List[dict] = []

        # Load saved colors or default
        saved = load_settings()
        self._c = saved.get("colors", dict(THEMES["VALORANT"]))
        for k, v in THEMES["VALORANT"].items():
            self._c.setdefault(k, v)

        self._setup_window()
        self._build_ui()
        self._schedule_tick()

    # -----------------------------------------------------------------------
    # Window setup
    # -----------------------------------------------------------------------
    def _setup_window(self) -> None:
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.95)
        self.configure(fg_color=self._c["bg"])
        self.geometry("264+50+50")
        self.resizable(False, False)
        self.bind("<F9>", lambda _: self._toggle_visible())
        self._app_icon = _make_app_icon()
        self.iconphoto(True, self._app_icon)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._build_titlebar()
        self._build_map_panel()
        self._build_canvas_panel()
        self._build_enemy_panel()
        self._build_utility_panel()
        self._build_callout_panel()
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

        icon = ctk.CTkLabel(bar, text=" ▣", text_color=c["accent"],
                            font=("Consolas", 17, "bold"))
        icon.pack(side="left", padx=(6, 4))
        self._title_icon = icon

        title = ctk.CTkLabel(bar, text="MINIMAP COACH", text_color=c["text"],
                             font=("Consolas", 10, "bold"))
        title.pack(side="left")
        self._title_lbl = title

        close_btn = ctk.CTkButton(
            bar, text="✕", width=38, height=42,
            fg_color="transparent", hover_color=c["enemy"],
            text_color=c["dim"], font=("Consolas", 14, "bold"),
            corner_radius=0, command=self.destroy,
        )
        close_btn.pack(side="right")
        self._close_btn = close_btn

        settings_btn = ctk.CTkButton(
            bar, text="⚙", width=42, height=42,
            fg_color="transparent", hover_color=c["panel"],
            text_color=c["dim"], font=("Consolas", 20),
            corner_radius=0, command=self._open_settings,
        )
        settings_btn.pack(side="right")
        self._settings_btn = settings_btn

        for w in (bar, icon, title):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

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

        hdr_lbl = ctk.CTkLabel(content, text=header, text_color=c["accent"],
                               font=("Consolas", 7, "bold"))
        hdr_lbl.pack(anchor="w", padx=10, pady=(5, 1))

        self._panel_refs.append((outer, stripe, hdr_lbl))
        return content

    def _build_map_panel(self) -> None:
        p = self._panel("MAP")
        self._map_lbl = ctk.CTkLabel(p, text="DETECTING...", text_color=self._c["text"],
                                     font=("Consolas", 15, "bold"))
        self._map_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _build_canvas_panel(self) -> None:
        p = self._panel("MINIMAP PREVIEW")
        self._canvas = tk.Canvas(p, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg=self._c["canvas_bg"], highlightthickness=1,
                                 highlightbackground=self._c["dim"])
        self._canvas.pack(padx=10, pady=(0, 8), anchor="w")
        self._draw_canvas([])

    def _build_enemy_panel(self) -> None:
        p = self._panel("ENEMIES SPOTTED")

        # Count + dots row
        top_row = ctk.CTkFrame(p, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(0, 4))

        self._enemy_count = ctk.CTkLabel(top_row, text="0",
                                         text_color=self._c["dim"], font=FONT_BIG)
        self._enemy_count.pack(side="left")

        dots_frame = ctk.CTkFrame(top_row, fg_color="transparent")
        dots_frame.pack(side="left", padx=(8, 0), pady=0)
        self._enemy_dots: List[ctk.CTkLabel] = []
        for _ in range(5):
            d = ctk.CTkLabel(dots_frame, text="◆", text_color=self._c["dim"],
                             font=("Consolas", 10))
            d.pack(side="left", padx=2)
            self._enemy_dots.append(d)

        # Thin divider
        self._enemy_div = ctk.CTkFrame(p, fg_color=self._c["dim"], height=1, corner_radius=0)
        self._enemy_div.pack(fill="x", padx=10, pady=(0, 4))

        # Sighting history rows (up to 5)
        hist_hdr = ctk.CTkLabel(p, text="LAST SIGHTED", text_color=self._c["dim"],
                                font=("Consolas", 7, "bold"))
        hist_hdr.pack(anchor="w", padx=10, pady=(0, 2))
        self._hist_hdr = hist_hdr

        self._hist_rows: List[Tuple[ctk.CTkLabel, ctk.CTkLabel]] = []
        for _ in range(5):
            row_frame = ctk.CTkFrame(p, fg_color="transparent", height=16)
            row_frame.pack(fill="x", padx=10, pady=0)
            row_frame.pack_propagate(False)

            label_lbl = ctk.CTkLabel(row_frame, text="", text_color=self._c["dim"],
                                     font=("Consolas", 9), anchor="w", width=110)
            label_lbl.pack(side="left")

            time_lbl = ctk.CTkLabel(row_frame, text="", text_color=self._c["dim"],
                                    font=("Consolas", 8), anchor="e")
            time_lbl.pack(side="right")

            self._hist_rows.append((label_lbl, time_lbl))

        ctk.CTkLabel(p, text="", height=4).pack()

    def _build_utility_panel(self) -> None:
        p = self._panel("UTILITY & AGENTS")
        self._util_rows: List[ctk.CTkLabel] = []
        for _ in range(4):
            lbl = ctk.CTkLabel(p, text="", text_color=self._c["dim"],
                               font=("Consolas", 9), anchor="w")
            lbl.pack(fill="x", padx=10, pady=0)
            self._util_rows.append(lbl)
        ctk.CTkLabel(p, text="", height=2).pack()

    def _build_callout_panel(self) -> None:
        p = self._panel("CALLOUT")
        self._callout_lbl = ctk.CTkLabel(p, text="listening...",
                                         text_color=self._c["text"],
                                         font=FONT_MONO, wraplength=218, justify="left")
        self._callout_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _build_ai_panel(self) -> None:
        p = self._panel("AI INSIGHT")
        self._ai_lbl = ctk.CTkLabel(p, text="", text_color=self._c["dim"],
                                    font=FONT_MONO, wraplength=218, justify="left")
        self._ai_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _build_bottombar(self) -> None:
        c = self._c
        bar = ctk.CTkFrame(self, fg_color=c["title"], corner_radius=0, height=40)
        bar.pack(fill="x", pady=(2, 0))
        bar.pack_propagate(False)
        self._bottombar = bar

        self._mute_btn = ctk.CTkButton(
            bar, text="▶  AUDIO ON", width=110, height=26,
            fg_color=c["accent"], text_color=c["bg"],
            hover_color=c["safe"], font=("Consolas", 8, "bold"),
            corner_radius=2, command=self._toggle_mute,
        )
        self._mute_btn.pack(side="left", padx=10, pady=7)

        ctk.CTkLabel(bar, text="F9 HIDE", text_color=c["dim"],
                     font=("Consolas", 7, "bold")).pack(side="right", padx=10)

    # -----------------------------------------------------------------------
    # Theme / color management
    # -----------------------------------------------------------------------
    def apply_colors(self, new_c: dict) -> None:
        self._c = new_c
        c = new_c

        self.configure(fg_color=c["bg"])
        self._titlebar.configure(fg_color=c["title"])
        self._title_stripe.configure(fg_color=c["accent"])
        self._title_icon.configure(text_color=c["accent"])
        self._title_lbl.configure(text_color=c["text"])
        self._close_btn.configure(hover_color=c["enemy"], text_color=c["dim"])
        self._settings_btn.configure(hover_color=c["panel"], text_color=c["dim"])

        for outer, stripe, hdr_lbl in self._panel_refs:
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
        for lbl, tlbl in self._hist_rows:
            lbl.configure(text_color=c["dim"])
            tlbl.configure(text_color=c["dim"])
        for lbl in self._util_rows:
            lbl.configure(text_color=c["dim"])
        self._callout_lbl.configure(text_color=c["text"])
        self._ai_lbl.configure(text_color=c["dim"])
        self._bottombar.configure(fg_color=c["title"])
        self._mute_btn.configure(
            fg_color=c["accent"] if not self._muted else c["panel"],
            text_color=c["bg"]   if not self._muted else c["dim"],
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
            best, best_dist = None, _SIGHTING_MATCH_RADIUS
            for entry in self._sighting_history:
                d = ((pos[0] - entry["pos"][0]) ** 2 +
                     (pos[1] - entry["pos"][1]) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best = d, entry
            if best is not None:
                best["pos"] = pos
                best["last_seen"] = now
                best["label"] = _pos_to_label(pos[0], pos[1])
            else:
                self._sighting_history.append({
                    "pos":       pos,
                    "last_seen": now,
                    "label":     _pos_to_label(pos[0], pos[1]),
                })

        # Keep only 5 most recent
        self._sighting_history.sort(key=lambda e: e["last_seen"], reverse=True)
        self._sighting_history = self._sighting_history[:5]

    def _refresh_history_display(self) -> None:
        now = time.time()
        c = self._c
        for i, (label_lbl, time_lbl) in enumerate(self._hist_rows):
            if i < len(self._sighting_history):
                e = self._sighting_history[i]
                elapsed = now - e["last_seen"]
                # Fresh sighting (< 3s) = enemy color, fading = dim
                color = c["enemy"] if elapsed < 3 else c["dim"]
                label_lbl.configure(text=f"● {e['label']}", text_color=color)
                time_lbl.configure(text=_ago(e["last_seen"]), text_color=color)
            else:
                label_lbl.configure(text="")
                time_lbl.configure(text="")

    # -----------------------------------------------------------------------
    # Drag
    # -----------------------------------------------------------------------
    def _drag_start(self, event) -> None:
        self._drag = {"x": event.x, "y": event.y}

    def _drag_move(self, event) -> None:
        x = self.winfo_x() + (event.x - self._drag["x"])
        y = self.winfo_y() + (event.y - self._drag["y"])
        self.geometry(f"+{x}+{y}")

    # -----------------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------------
    def _toggle_mute(self) -> None:
        c = self._c
        self._muted = not self._muted
        if self._muted:
            self._mute_btn.configure(text="◼  MUTED", fg_color=c["panel"],
                                     text_color=c["dim"])
        else:
            self._mute_btn.configure(text="▶  AUDIO ON", fg_color=c["accent"],
                                     text_color=c["bg"])
        if self.on_mute_change:
            self.on_mute_change(self._muted)

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
        now = time.time()
        live = self._tracker.tick(now)
        self._draw_canvas(live)
        self._refresh_history_display()
        self.after(100, self._schedule_tick)

    def _draw_canvas(self, enemies: List[TrackedEnemy]) -> None:
        cr = self._c
        cv = self._canvas
        s = CANVAS_SIZE
        cv.delete("all")

        step = s // 4
        for i in range(0, s + 1, step):
            cv.create_line(i, 0, i, s, fill=cr["grid"], width=1)
            cv.create_line(0, i, s, i, fill=cr["grid"], width=1)

        m = s // 2
        cv.create_line(m - 10, m, m + 10, m, fill=cr["dim"], width=1)
        cv.create_line(m, m - 10, m, m + 10, fill=cr["dim"], width=1)

        for ab in self._abilities:
            ax = int(ab["position"][0] * s)
            ay = int(ab["position"][1] * s)
            color = ab.get("color", "#ffffff")
            cv.create_oval(ax - 4, ay - 4, ax + 4, ay + 4, fill=color, outline="")
            cv.create_text(ax + 6, ay, text=ab["display"][:3].upper(),
                           fill=color, font=("Consolas", 6), anchor="w")

        # Player as diamond
        cv.create_polygon(m, m - 7, m + 7, m, m, m + 7, m - 7, m,
                          fill=cr["accent"], outline="")
        cv.create_oval(m - 2, m - 2, m + 2, m + 2, fill=cr["canvas_bg"], outline="")

        for e in enemies:
            ex = int(e.position[0] * s)
            ey = int(e.position[1] * s)
            intensity = int(0xe8 * e.alpha)
            dim_v = int(0x40 * e.alpha)
            color = f"#{intensity:02x}{dim_v:02x}{dim_v:02x}"
            r = 6 if e.alpha > 0.6 else 4
            cv.create_oval(ex - r, ey - r, ex + r, ey + r, fill=color, outline="")

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

    def update_ai(self, text: str) -> None:
        self._ai_lbl.configure(text=text)

    def update_utility(self, abilities: List[dict]) -> None:
        self._abilities = abilities
        for i, row in enumerate(self._util_rows):
            if i < len(abilities):
                ab = abilities[i]
                color = ab.get("color", self._c["dim"])
                row.configure(text=f"◆  {ab['display']}", text_color=color)
            else:
                row.configure(text="")
