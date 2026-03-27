"""
Spike defuse feasibility analysis from audio.

Valorant spike facts (fixed values):
  - Total timer: 45 seconds
  - Defuse time: 7 seconds (no defuse kits in Valorant -- always 7s)
  - Spike Rush mode: 20 seconds (detected separately)

The spike emits a characteristic tonal beep (~880 Hz center, ~60 ms duration)
with decreasing inter-beep intervals (IBI) as the timer runs down.
IBI ranges from ~2.0 s at plant to ~0.2 s near detonation.

Time estimation model:
  Primary source : wall clock elapsed since _spike_plant_time (already in coach.py)
  Secondary source: audio IBI → power-law estimate: remaining ≈ 26.0 * IBI^0.78
  When both sources are available, wall clock wins; audio used to detect missed plant.

Defuse feasibility:
  remaining_time > DEFUSE_TIME + travel_to_spike  →  "Go!"
  remaining_time <= DEFUSE_TIME + travel_to_spike →  "Too late, hold!"

Advice is spoken at threshold crossings and time milestones (20 s, 10 s, 7 s).

DefuseSoundDetector:
  Detects the sustained defuse interaction hum (~350-1200 Hz, continuous for up to 7 s).
  When detected, records onset_t (time.monotonic()).  onset_t is None when no active defuse.
  Hypothetical defuse progress: pct = min(1.0, (now - onset_t) / 7.0)
  This is NOT assumed to be a real defuse -- it answers "how far along would they be
  if defuse started when we first heard that sound?" -- useful for deciding when to peek.
"""
from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt

from src.audio.capture import SAMPLE_RATE

# ── Spike beep audio characteristics ────────────────────────────────────────
_BEEP_LOW_HZ   = 700    # bandpass low edge  (Hz)
_BEEP_HIGH_HZ  = 2000   # bandpass high edge (Hz)
_BEEP_DUR_S    = 0.06   # beep duration ~60 ms
_BEEP_WIN      = int(_BEEP_DUR_S * SAMPLE_RATE)   # samples per analysis window
_MIN_IBI_S     = 0.12   # minimum allowed IBI (debounce)
_BEEP_THRESH   = 6.0    # short-term RMS / long-term RMS ratio to declare a beep

# ── Spike timer constants ────────────────────────────────────────────────────
_SPIKE_TOTAL   = 45.0   # seconds (standard)
_DEFUSE_TIME   = 7.0    # seconds (no kits in Valorant)
_DEFUSE_MARGIN = 0.5    # safety margin added to DEFUSE_TIME for callouts

# ── IBI → remaining time model ───────────────────────────────────────────────
# Power-law fit: remaining ≈ _IBI_COEFF * IBI ^ _IBI_EXP
# Calibration:  IBI=2.0s → 45s remaining,  IBI=0.2s → ~1s remaining
_IBI_COEFF = 26.0
_IBI_EXP   = 0.78

# ── Defuse advice milestones ─────────────────────────────────────────────────
_ADVICE_THRESHOLDS = [20.0, 10.0, 7.0 + _DEFUSE_MARGIN]   # seconds remaining


# ── Defuse hum audio characteristics ────────────────────────────────────────
_DEFUSE_LOW_HZ   = 350    # bandpass low edge  (Hz) -- defuse interaction hum
_DEFUSE_HIGH_HZ  = 1200   # bandpass high edge (Hz)
_DEFUSE_WIN_S    = 0.02   # analysis window 20 ms
_DEFUSE_WIN      = int(_DEFUSE_WIN_S * SAMPLE_RATE)
_DEFUSE_THRESH   = 3.0    # short RMS / long RMS to declare active defuse hum
_DEFUSE_CONFIRM  = 15     # consecutive windows required to lock onset (~300 ms)
_DEFUSE_DROP     = 6      # consecutive quiet windows to declare abort

# ── Butterworth bandpass (SOS) ───────────────────────────────────────────────
def _make_bandpass(lo: int, hi: int):
    nyq = SAMPLE_RATE / 2.0
    return butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")

_BP_SOS       = _make_bandpass(_BEEP_LOW_HZ, _BEEP_HIGH_HZ)
_DEFUSE_BP_SOS = _make_bandpass(_DEFUSE_LOW_HZ, _DEFUSE_HIGH_HZ)


class SpikeBeepDetector:
    """
    Detects individual spike beep events from a mono audio buffer.

    Call process(mono) each audio tick. Returns a list of detection timestamps
    (time.monotonic()) for each beep found in the buffer.
    """

    def __init__(self) -> None:
        self._long_rms: float = 1e-6        # slowly-updating background level
        self._last_beep_t: float = 0.0      # time.monotonic() of last detected beep
        self._active: bool = False           # True while in spike round

    def reset(self) -> None:
        """Call when round starts (no spike planted)."""
        self._long_rms = 1e-6
        self._last_beep_t = 0.0
        self._active = False

    def arm(self) -> None:
        """Call when spike is planted -- start listening for beeps."""
        self._active = True

    def process(self, mono: np.ndarray) -> List[float]:
        """
        Process one audio chunk.  mono: 1-D float32 array (any length).
        Returns list of beep timestamps (may be empty).
        """
        if not self._active or len(mono) < _BEEP_WIN:
            return []

        filtered = sosfilt(_BP_SOS, mono)
        beeps: List[float] = []
        now = time.monotonic()

        # Slide a window across the buffer in steps of _BEEP_WIN // 2
        step = _BEEP_WIN // 2
        for start in range(0, len(filtered) - _BEEP_WIN + 1, step):
            chunk = filtered[start : start + _BEEP_WIN]
            short_rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)

            # Update long-term background (slow exponential)
            alpha = 0.01
            self._long_rms = (1 - alpha) * self._long_rms + alpha * short_rms

            ratio = short_rms / (self._long_rms + 1e-9)
            if ratio >= _BEEP_THRESH:
                t_beep = now - (len(mono) - start) / SAMPLE_RATE
                if t_beep - self._last_beep_t >= _MIN_IBI_S:
                    beeps.append(t_beep)
                    self._last_beep_t = t_beep

        return beeps


class SpikeTimer:
    """
    Maintains a history of spike beep timestamps and estimates remaining time.

    Two estimation modes:
      1. Wall-clock (authoritative): uses plant_time passed from coach.py
      2. Audio IBI (fallback/validation): power-law estimate from inter-beep interval

    Both estimates are blended only when IBI confidence is high.
    """

    def __init__(self) -> None:
        self._beep_times: deque = deque(maxlen=8)  # last 8 beep timestamps
        self._plant_time: Optional[float] = None    # time.monotonic() of plant
        self._last_ibi: Optional[float] = None
        self._ibi_confidence: float = 0.0

    def reset(self) -> None:
        self._beep_times.clear()
        self._plant_time = None
        self._last_ibi = None
        self._ibi_confidence = 0.0

    def on_spike_planted(self, plant_time: float) -> None:
        """Register plant time (time.monotonic() from coach.py)."""
        self._plant_time = plant_time
        self._beep_times.clear()

    def add_beep(self, timestamp: float) -> None:
        """Register a detected beep event."""
        self._beep_times.append(timestamp)

        if len(self._beep_times) >= 2:
            ibis = [
                self._beep_times[i] - self._beep_times[i - 1]
                for i in range(1, len(self._beep_times))
            ]
            # Median IBI from recent history (robust to outliers)
            self._last_ibi = float(np.median(ibis))
            # Confidence: high when IBI is consistent
            if len(ibis) >= 3:
                cv = float(np.std(ibis) / (np.mean(ibis) + 1e-9))
                self._ibi_confidence = float(np.clip(1.0 - cv * 2.0, 0.0, 1.0))
            else:
                self._ibi_confidence = 0.5

    def remaining(self) -> Optional[float]:
        """
        Returns estimated seconds remaining on the spike.
        Prefers wall-clock; cross-validates with audio IBI.
        Returns None if spike not planted.
        """
        if self._plant_time is None:
            return None

        elapsed = time.monotonic() - self._plant_time
        wall_remaining = max(0.0, _SPIKE_TOTAL - elapsed)

        # If audio IBI estimate is available and confident, blend
        if self._last_ibi is not None and self._ibi_confidence > 0.5:
            ibi_remaining = float(
                np.clip(_IBI_COEFF * (self._last_ibi ** _IBI_EXP), 0.0, _SPIKE_TOTAL)
            )
            # Wall clock weighted heavily (0.85) since it's authoritative
            return 0.85 * wall_remaining + 0.15 * ibi_remaining

        return wall_remaining

    def ibi_estimate(self) -> Optional[float]:
        """Returns raw IBI-based remaining estimate (for diagnostics)."""
        if self._last_ibi is None:
            return None
        return float(
            np.clip(_IBI_COEFF * (self._last_ibi ** _IBI_EXP), 0.0, _SPIKE_TOTAL)
        )


class DefuseSoundDetector:
    """
    Detects the defuse interaction hum (sustained ~350-1200 Hz tone for up to 7 s).

    onset_t is set to time.monotonic() when sustained energy is first confirmed.
    It stays set until the hum disappears (defuse aborted) or reset() is called.

    Thread note: onset_t is a plain float attribute.  CPython's GIL makes float
    reads/writes from another thread safe for this read-only access pattern.
    """

    def __init__(self) -> None:
        self._long_rms: float = 1e-6
        self._above_count: int = 0    # consecutive windows above threshold
        self._below_count: int = 0    # consecutive windows below (for abort detection)
        self._armed: bool = False
        # Public: read by audio_coach → coach.py → overlay
        self.onset_t: Optional[float] = None

    def reset(self) -> None:
        self._long_rms = 1e-6
        self._above_count = 0
        self._below_count = 0
        self._armed = False
        self.onset_t = None

    def arm(self) -> None:
        """Call when spike is planted."""
        self._armed = True

    def process(self, mono: np.ndarray) -> None:
        """
        Update state from one audio chunk.
        Mutates self.onset_t -- no return value needed.
        """
        if not self._armed or len(mono) < _DEFUSE_WIN:
            return

        filtered = sosfilt(_DEFUSE_BP_SOS, mono)

        # Process chunk in _DEFUSE_WIN windows
        for start in range(0, len(filtered) - _DEFUSE_WIN + 1, _DEFUSE_WIN):
            chunk = filtered[start : start + _DEFUSE_WIN]
            short_rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)

            # Slow background update
            alpha = 0.005
            self._long_rms = (1 - alpha) * self._long_rms + alpha * short_rms

            ratio = short_rms / (self._long_rms + 1e-9)

            if ratio >= _DEFUSE_THRESH:
                self._above_count += 1
                self._below_count = 0
                # Lock onset once confirmed
                if self._above_count == _DEFUSE_CONFIRM and self.onset_t is None:
                    # Back-date onset by the confirmation window duration
                    self.onset_t = time.monotonic() - _DEFUSE_WIN_S * _DEFUSE_CONFIRM
            else:
                self._below_count += 1
                if self._below_count >= _DEFUSE_DROP:
                    # Hum stopped -- defuse aborted (or complete)
                    self._above_count = 0
                    self._below_count = 0
                    self.onset_t = None

    def progress(self) -> Optional[float]:
        """
        Returns hypothetical defuse fraction 0.0-1.0 (None if no active defuse hum).
        1.0 means 7 s of continuous hum elapsed -- they would be done.
        """
        if self.onset_t is None:
            return None
        return min(1.0, (time.monotonic() - self.onset_t) / _DEFUSE_TIME)


class DefuseAdvisor:
    """
    Per-tick advisor: is there time to defuse?

    Call update() every tick during POST_PLANT.
    Returns a voice callout string or None.

    Advice logic:
      - When remaining > DEFUSE_TIME + travel + margin: "You can make it!"
      - When remaining drops below DEFUSE_TIME + travel + margin: "No time, hold!"
      - Milestone callouts at 20s, 10s, and 7s remaining (once each)
      - Feasibility flip callout when go/no-go status changes
    """

    def __init__(self) -> None:
        self._last_advice_t: float = 0.0
        self._advice_cooldown = 4.0          # seconds between same-status repeats
        self._announced_thresholds: set = set()
        self._prev_feasible: Optional[bool] = None

    def reset(self) -> None:
        self._last_advice_t = 0.0
        self._announced_thresholds = set()
        self._prev_feasible = None

    def update(
        self,
        remaining: float,
        travel_time: float,
    ) -> Optional[str]:
        """
        remaining   : seconds left on spike (from SpikeTimer.remaining())
        travel_time : estimated seconds to reach spike (from retake_advisor travel data)

        Returns voice callout string or None.
        """
        now = time.monotonic()
        needed = _DEFUSE_TIME + travel_time + _DEFUSE_MARGIN
        feasible = remaining > needed

        callout: Optional[str] = None

        # ── Threshold milestone callouts ─────────────────────────────────────
        for threshold in _ADVICE_THRESHOLDS:
            if remaining <= threshold and threshold not in self._announced_thresholds:
                self._announced_thresholds.add(threshold)
                rem_int = int(remaining)
                if threshold <= _DEFUSE_TIME + _DEFUSE_MARGIN:
                    if feasible:
                        callout = f"{rem_int}s left -- go defuse now!"
                    else:
                        callout = f"{rem_int}s left -- no time to defuse, hold!"
                elif threshold == 10.0:
                    if feasible:
                        callout = f"10 seconds left, defuse possible."
                    else:
                        callout = "10 seconds left, cannot defuse in time."
                else:
                    if feasible:
                        callout = f"{rem_int} seconds, get to the spike."
                    else:
                        callout = f"{rem_int} seconds, retake risky."
                break  # announce only the highest-priority threshold per tick

        # ── Feasibility flip callout ─────────────────────────────────────────
        if callout is None and self._prev_feasible is not None:
            if self._prev_feasible and not feasible:
                callout = "Window closed -- too late to defuse, play for hold."
            elif not self._prev_feasible and feasible:
                callout = "Defuse window open, move now!"

        self._prev_feasible = feasible

        # Apply cooldown to avoid spamming
        if callout is not None:
            if now - self._last_advice_t < self._advice_cooldown:
                # Suppress unless it's a hard warning (critical threshold)
                if remaining > _DEFUSE_TIME + _DEFUSE_MARGIN + 2.0:
                    callout = None
            if callout is not None:
                self._last_advice_t = now

        return callout
