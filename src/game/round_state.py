"""
Round state machine for Valorant.

States:
  BUY_PHASE    -- 30 s buy window at round start
  ROUND_ACTIVE -- live round, no spike planted
  POST_PLANT   -- spike has been planted
  ROUND_END    -- round over (win or loss)

Transitions are driven by:
  - Spike detection (POST_PLANT)
  - Enemy count disappearing suddenly (ROUND_END heuristic)
  - Elapsed time (BUY_PHASE -> ROUND_ACTIVE after ~30 s)
  - Explicit reset() call (from audio round_audio.py when round-start sound detected)

The state machine also tracks:
  - Round number (increments on each ROUND_END -> BUY_PHASE)
  - Attackers vs defenders side (alternates every 12 rounds, switches at half)
  - Time-in-state (useful for "30 seconds left" style callouts)

Designed to be updated once per coach loop tick.
"""
from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional


class State(Enum):
    BUY_PHASE    = auto()
    ROUND_ACTIVE = auto()
    POST_PLANT   = auto()
    ROUND_END    = auto()


# Timing constants
_BUY_PHASE_SECONDS    = 30.0
_ROUND_MAX_SECONDS    = 100.0   # max round length before forcing ROUND_END
_CLEAR_CONFIRM_TICKS  = 15      # consecutive ticks with 0 enemies before ROUND_END

# Side alternates at round 12 (MR12) or 13 (MR13)
_HALF_AT_ROUND        = 12


class RoundState:
    def __init__(self) -> None:
        self.state:       State  = State.BUY_PHASE
        self.round_num:   int    = 1
        self.on_attack:   bool   = True    # True = attacking side this half
        self._state_start = time.monotonic()
        self._clear_ticks = 0
        self._prev_enemy_count = 0
        self._finished: bool = False        # guard against double _finish_round()

    # ------------------------------------------------------------------
    # External triggers
    # ------------------------------------------------------------------

    def on_spike_planted(self) -> None:
        if self.state == State.ROUND_ACTIVE:
            self._transition(State.POST_PLANT)

    def on_round_start_sound(self) -> None:
        """Called by round_audio.py when it detects the round-start horn."""
        self.reset()

    def on_round_end_sound(self) -> None:
        """Called by round_audio.py when win/loss jingle detected."""
        self._transition(State.ROUND_END)
        self._finish_round()

    def reset(self) -> None:
        """Force reset to buy phase (called at round start)."""
        self._transition(State.BUY_PHASE)
        self._clear_ticks = 0
        self._finished = False

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def update(self, enemy_count: int, spike_planted: bool) -> Optional[str]:
        """
        Call once per coach loop tick.
        Returns an event string if a transition occurred, else None.
          "round_start" | "spike_planted" | "round_end"
        """
        now = time.monotonic()
        elapsed = now - self._state_start
        event: Optional[str] = None

        if self.state == State.BUY_PHASE:
            if elapsed >= _BUY_PHASE_SECONDS:
                self._transition(State.ROUND_ACTIVE)
                event = "round_start"

        elif self.state == State.ROUND_ACTIVE:
            if spike_planted:
                self.on_spike_planted()
                event = "spike_planted"
            elif elapsed >= _ROUND_MAX_SECONDS:
                self._transition(State.ROUND_END)
                self._finish_round()
                event = "round_end"
            else:
                # Heuristic: enemies suddenly gone after being visible
                if self._prev_enemy_count >= 2 and enemy_count == 0:
                    self._clear_ticks += 1
                    if self._clear_ticks >= _CLEAR_CONFIRM_TICKS:
                        self._transition(State.ROUND_END)
                        self._finish_round()
                        event = "round_end"
                else:
                    self._clear_ticks = 0

        elif self.state == State.POST_PLANT:
            # Post-plant round end: all enemies gone OR timer
            if elapsed >= 45.0:
                self._transition(State.ROUND_END)
                self._finish_round()
                event = "round_end"
            elif self._prev_enemy_count >= 1 and enemy_count == 0:
                self._clear_ticks += 1
                if self._clear_ticks >= _CLEAR_CONFIRM_TICKS:
                    self._transition(State.ROUND_END)
                    self._finish_round()
                    event = "round_end"
            else:
                self._clear_ticks = 0

        elif self.state == State.ROUND_END:
            # Auto-advance to buy phase after a short pause
            if elapsed >= 7.0:
                self._transition(State.BUY_PHASE)

        self._prev_enemy_count = enemy_count
        return event

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def time_in_state(self) -> float:
        return time.monotonic() - self._state_start

    @property
    def is_active(self) -> bool:
        return self.state in (State.ROUND_ACTIVE, State.POST_PLANT)

    @property
    def side(self) -> str:
        return "attack" if self.on_attack else "defense"

    # ------------------------------------------------------------------

    def _transition(self, new_state: State) -> None:
        self.state = new_state
        self._state_start = time.monotonic()
        self._clear_ticks = 0

    def _finish_round(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.round_num += 1
        if self.round_num == _HALF_AT_ROUND + 1:
            self.on_attack = not self.on_attack
