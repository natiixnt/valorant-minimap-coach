"""
Tracker.gg-inspired always-on-top overlay.
All public update_*() methods MUST be called via overlay.after(0, fn)
from non-main threads.
"""
import tkinter as tk
import time
from typing import Callable, List, Optional

import customtkinter as ctk

from src.vision.enemy_tracker import EnemyTracker, TrackedEnemy

ctk.set_appearance_mode("dark")

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C_BG = "#0f1923"
C_PANEL = "#1a2332"
C_TITLE = "#111827"
C_ACCENT = "#5bc4e8"
C_ENEMY = "#e84057"
C_SAFE = "#3ec17c"
C_TEXT = "#ecf0f1"
C_DIM = "#7f8c9b"
C_CANVAS_BG = "#0a1018"
C_GRID = "#1c2b3a"

FONT_MONO = ("Consolas", 10)
FONT_MONO_BOLD = ("Consolas", 10, "bold")
FONT_BIG = ("Consolas", 20, "bold")
CANVAS_SIZE = 120


class OverlayWindow(ctk.CTk):
    def __init__(self, fade_after: float = 5.0) -> None:
        super().__init__()
        self._muted = False
        self.on_mute_change: Optional[Callable[[bool], None]] = None
        self._tracker = EnemyTracker(fade_after=fade_after)
        self._drag: dict = {}
        self._visible = True
        # list of {"display": str, "color": hex, "position": (x,y)}
        self._abilities: List[dict] = []

        self._setup_window()
        self._build_ui()
        self._schedule_tick()

    # -----------------------------------------------------------------------
    # Window setup
    # -----------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.93)
        self.configure(fg_color=C_BG)
        self.geometry("+50+50")
        self.resizable(False, False)
        # F9 toggle (works when overlay has focus)
        self.bind("<F9>", lambda _: self._toggle_visible())

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
        bar = ctk.CTkFrame(self, fg_color=C_TITLE, corner_radius=0, height=30)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        dot = ctk.CTkLabel(bar, text="◉", text_color=C_ACCENT, font=FONT_MONO_BOLD)
        dot.pack(side="left", padx=(10, 4))

        lbl = ctk.CTkLabel(bar, text="MINIMAP COACH", text_color=C_TEXT,
                            font=FONT_MONO_BOLD)
        lbl.pack(side="left")

        close = ctk.CTkButton(bar, text="×", width=30, height=30,
                               fg_color="transparent", hover_color=C_ENEMY,
                               text_color=C_DIM, font=("Consolas", 16, "bold"),
                               corner_radius=0, command=self.destroy)
        close.pack(side="right")

        for w in (bar, dot, lbl):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

    def _panel(self, header: str) -> ctk.CTkFrame:
        outer = ctk.CTkFrame(self, fg_color=C_PANEL, corner_radius=6)
        outer.pack(fill="x", padx=8, pady=(5, 0))
        ctk.CTkLabel(outer, text=header, text_color=C_ACCENT,
                     font=("Consolas", 8, "bold")).pack(anchor="w", padx=10, pady=(6, 1))
        return outer

    def _build_map_panel(self) -> None:
        p = self._panel("MAP")
        self._map_lbl = ctk.CTkLabel(p, text="detecting...", text_color=C_TEXT,
                                      font=("Consolas", 14, "bold"))
        self._map_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _build_canvas_panel(self) -> None:
        p = self._panel("MINIMAP PREVIEW")
        self._canvas = tk.Canvas(p, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                  bg=C_CANVAS_BG, highlightthickness=0)
        self._canvas.pack(padx=10, pady=(0, 8), anchor="w")
        self._draw_canvas([])

    def _build_enemy_panel(self) -> None:
        p = self._panel("ENEMIES SPOTTED")
        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(0, 8))
        self._enemy_count = ctk.CTkLabel(row, text="0", text_color=C_DIM,
                                          font=FONT_BIG)
        self._enemy_count.pack(side="left")
        self._enemy_dots = ctk.CTkLabel(row, text="○ ○ ○ ○ ○", text_color=C_DIM,
                                         font=("Consolas", 12))
        self._enemy_dots.pack(side="left", padx=(8, 0), pady=(6, 0))

    def _build_utility_panel(self) -> None:
        p = self._panel("UTILITY & AGENTS")
        # Up to 4 slots, each a colored dot + label
        self._util_rows: List[ctk.CTkLabel] = []
        for _ in range(4):
            lbl = ctk.CTkLabel(p, text="", text_color=C_DIM,
                               font=("Consolas", 9), anchor="w")
            lbl.pack(fill="x", padx=10, pady=0)
            self._util_rows.append(lbl)
        # Spacer to keep panel from collapsing when empty
        ctk.CTkLabel(p, text="", height=2).pack()

    def _build_callout_panel(self) -> None:
        p = self._panel("CALLOUT")
        self._callout_lbl = ctk.CTkLabel(p, text="listening...", text_color=C_TEXT,
                                          font=FONT_MONO, wraplength=220, justify="left")
        self._callout_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _build_ai_panel(self) -> None:
        p = self._panel("AI INSIGHT")
        self._ai_lbl = ctk.CTkLabel(p, text="", text_color=C_DIM,
                                     font=FONT_MONO, wraplength=220, justify="left")
        self._ai_lbl.pack(anchor="w", padx=10, pady=(0, 8))

    def _build_bottombar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=C_TITLE, corner_radius=0, height=36)
        bar.pack(fill="x", pady=(5, 0))
        bar.pack_propagate(False)

        self._mute_btn = ctk.CTkButton(
            bar, text="🔊  ON", width=80, height=24,
            fg_color=C_ACCENT, text_color="#0f1923",
            hover_color="#4aaabf", font=("Consolas", 9, "bold"),
            corner_radius=4, command=self._toggle_mute,
        )
        self._mute_btn.pack(side="left", padx=8, pady=6)

        ctk.CTkLabel(bar, text="F9 hide", text_color=C_DIM,
                     font=("Consolas", 8)).pack(side="right", padx=10)

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
        self._muted = not self._muted
        if self._muted:
            self._mute_btn.configure(text="🔇  MUTED", fg_color="#3d4f61")
        else:
            self._mute_btn.configure(text="🔊  ON", fg_color=C_ACCENT)
        if self.on_mute_change:
            self.on_mute_change(self._muted)

    def _toggle_visible(self) -> None:
        if self._visible:
            self.withdraw()
        else:
            self.deiconify()
        self._visible = not self._visible

    # -----------------------------------------------------------------------
    # Tick loop (UI thread, 10 Hz)
    # -----------------------------------------------------------------------

    def _schedule_tick(self) -> None:
        now = time.time()
        live = self._tracker.tick(now)
        self._draw_canvas(live)
        self.after(100, self._schedule_tick)

    def _draw_canvas(self, enemies: List[TrackedEnemy]) -> None:
        c = self._canvas
        s = CANVAS_SIZE
        c.delete("all")

        # Subtle grid
        for i in range(0, s + 1, s // 4):
            c.create_line(i, 0, i, s, fill=C_GRID, width=1)
            c.create_line(0, i, s, i, fill=C_GRID, width=1)

        # Ability dots (drawn first so enemies/player render on top)
        for ab in self._abilities:
            ax = int(ab["position"][0] * s)
            ay = int(ab["position"][1] * s)
            color = ab.get("color", "#ffffff")
            c.create_oval(ax - 4, ay - 4, ax + 4, ay + 4, fill=color, outline="")
            # Small label tag
            c.create_text(ax + 6, ay, text=ab["display"][:3].upper(),
                          fill=color, font=("Consolas", 6), anchor="w")

        # Player dot (center)
        m = s // 2
        c.create_oval(m - 5, m - 5, m + 5, m + 5, fill=C_ACCENT, outline="")
        c.create_oval(m - 2, m - 2, m + 2, m + 2, fill="#0a1018", outline="")

        # Enemy dots with alpha-based color fade
        for e in enemies:
            ex = int(e.position[0] * s)
            ey = int(e.position[1] * s)
            intensity = int(0xe8 * e.alpha)
            dim = int(0x40 * e.alpha)
            color = f"#{intensity:02x}{dim:02x}{dim:02x}"
            r = 6 if e.alpha > 0.6 else 4
            c.create_oval(ex - r, ey - r, ex + r, ey + r, fill=color, outline="")

    # -----------------------------------------------------------------------
    # Thread-safe public API (always call via overlay.after(0, fn))
    # -----------------------------------------------------------------------

    def update_map(self, name: str) -> None:
        self._map_lbl.configure(text=name.upper())

    def update_enemies(self, count: int, positions: list) -> None:
        color = C_ENEMY if count > 0 else C_DIM
        self._enemy_count.configure(text=str(count), text_color=color)
        filled = "● " * min(count, 5)
        empty = "○ " * max(0, 5 - count)
        self._enemy_dots.configure(text=(filled + empty).strip(), text_color=color)
        self._tracker.update(positions)

    def update_callout(self, text: str) -> None:
        self._callout_lbl.configure(text=text, text_color=C_TEXT)

    def update_ai(self, text: str) -> None:
        self._ai_lbl.configure(text=text)

    def update_utility(self, abilities: List[dict]) -> None:
        """
        abilities: list of {"display": str, "color": hex, "position": (x,y)}
        Shows up to 4 items in the utility panel with color-coded dots.
        """
        self._abilities = abilities
        for i, row in enumerate(self._util_rows):
            if i < len(abilities):
                ab = abilities[i]
                color = ab.get("color", C_DIM)
                row.configure(text=f"◆ {ab['display']}", text_color=color)
            else:
                row.configure(text="")
