"""
Spike defuse feasibility analysis from audio.

Valorant spike facts (verified):
  - Total timer: 45 seconds (Spike Rush uses the same 45s detonation timer)
  - Plant time: 4 seconds
  - Defuse time: 7 seconds total; 3.5 s saves progress (half-defuse mechanic).
    If defuse is interrupted after 3.5 s, only 3.5 s more are needed on next attempt.
    No defuse kits in Valorant -- always 7 s for full defuse.

The spike beep rate escalates in discrete steps (community-documented, not officially
published by Riot):
  Elapsed 0-25 s  (remaining 45-20 s): ~1 beep/s  → IBI ≈ 1.0 s
  Elapsed 25-35 s (remaining 20-10 s): ~2 beeps/s → IBI ≈ 0.5 s
  Elapsed 35-40 s (remaining 10-5 s):  ~4 beeps/s → IBI ≈ 0.25 s
  Elapsed 40-45 s (remaining 5-0 s):   ~8 beeps/s → IBI ≈ 0.125 s
Exact beep frequency (Hz pitch) is not publicly documented by Riot.

Time estimation model:
  Primary source : wall clock elapsed since _spike_plant_time (already in coach.py)
  Secondary source: audio IBI → power-law estimate: remaining ≈ 45.0 * IBI^1.39
    Calibrated from: IBI=1.0s→45s remaining, IBI=0.125s→~2.5s remaining.
  When both sources are available, wall clock wins; audio used to detect missed plant.

Defuse feasibility:
  remaining_time > DEFUSE_TIME + travel_to_spike  →  "Go!"
  remaining_time <= DEFUSE_TIME + travel_to_spike →  "Too late, hold!"

Advice is spoken at threshold crossings and time milestones (20 s, 10 s, 7 s).

DefuseSoundDetector:
  Detects the defuse START click (short mid-frequency onset when E is pressed on spike).
  Once detected, runs a pure wall-clock 7 s timer.  No abort detection -- if defuse is
  cancelled the bar just counts to 100% and hides.  If E is pressed again after the
  cooldown window, on_spike_resolved() + arm() resets everything for a fresh detection.
"""
from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from src.audio.capture import SAMPLE_RATE

# ── Spike beep audio characteristics ────────────────────────────────────────
_BEEP_LOW_HZ   = 700    # bandpass low edge  (Hz)
_BEEP_HIGH_HZ  = 2000   # bandpass high edge (Hz)
_BEEP_DUR_S    = 0.06   # beep duration ~60 ms
_BEEP_WIN      = int(_BEEP_DUR_S * SAMPLE_RATE)   # samples per analysis window
_MIN_IBI_S     = 0.08   # debounce: half of minimum real IBI (0.125 s at 8 bps)
_BEEP_THRESH   = 6.0    # short-term RMS / long-term RMS ratio to declare a beep

# ── Spike timer constants ────────────────────────────────────────────────────
_SPIKE_TOTAL   = 45.0   # seconds (standard)
_DEFUSE_TIME   = 7.0    # seconds (no kits in Valorant)
_DEFUSE_MARGIN = 0.5    # safety margin added to DEFUSE_TIME for callouts

# ── IBI → remaining time model ───────────────────────────────────────────────
# Power-law fit: remaining ≈ _IBI_COEFF * IBI ^ _IBI_EXP
# Calibration (from community-documented beep rate steps):
#   IBI=1.0s  → 45s remaining  (1 bps phase, elapsed 0-25s)
#   IBI=0.125s → ~2.5s remaining (8 bps phase, elapsed 40-45s)
# Fit: a=45, b=log(2.5/45)/log(0.125/1.0) = log(0.0556)/log(0.125) ≈ 1.39
_IBI_COEFF = 45.0
_IBI_EXP   = 1.39

# ── IBI debounce ─────────────────────────────────────────────────────────────
# Minimum IBI at max beep rate is 0.125 s (8 bps). Debounce at half that.
_LONG_RMS_ALPHA = 0.01    # EMA coefficient for beep detector background level

# ── Defuse advice milestones ─────────────────────────────────────────────────
_ADVICE_THRESHOLDS = [20.0, 10.0, _DEFUSE_TIME + _DEFUSE_MARGIN]   # seconds remaining


# ── Defuse start-sound characteristics ──────────────────────────────────────
# When a player presses E on the spike, Valorant plays a short mid-frequency
# click/activation sound (~500-1500 Hz, ~40-80 ms).  We detect that single
# onset and then run a pure wall-clock 7 s timer.
# No "abort" detection from audio -- if defuse is cancelled the timer just
# reaches 100% and the bar hides.  If defuse restarts (new press of E), reset()
# is called externally and a new onset can be detected.
_DEFUSE_LOW_HZ   = 500    # bandpass low edge  (Hz)
_DEFUSE_HIGH_HZ  = 1500   # bandpass high edge (Hz)
_DEFUSE_WIN_S    = 0.04   # analysis window 40 ms
_DEFUSE_WIN      = int(_DEFUSE_WIN_S * SAMPLE_RATE)
_DEFUSE_THRESH   = 5.0    # short RMS / long RMS ratio to fire onset
_DEFUSE_COOLDOWN  = 8.0    # s -- re-arm only after this much time (prevent re-trigger)
_DEFUSE_BG_ALPHA  = 0.005  # EMA coefficient for defuse detector background level

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
        self._zi_bp = sosfilt_zi(_BP_SOS) * 0.0   # IIR filter state across chunks

    def reset(self) -> None:
        """Call when round starts (no spike planted)."""
        self._long_rms = 1e-6
        self._last_beep_t = 0.0
        self._active = False
        self._zi_bp = sosfilt_zi(_BP_SOS) * 0.0

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

        filtered, self._zi_bp = sosfilt(_BP_SOS, mono, zi=self._zi_bp)
        beeps: List[float] = []
        now = time.monotonic()

        # Slide a window across the buffer in steps of _BEEP_WIN // 2
        step = _BEEP_WIN // 2
        for start in range(0, len(filtered) - _BEEP_WIN + 1, step):
            chunk = filtered[start : start + _BEEP_WIN]
            short_rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)

            # Update long-term background (slow exponential)
            self._long_rms = (1 - _LONG_RMS_ALPHA) * self._long_rms + _LONG_RMS_ALPHA * short_rms

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

    @property
    def has_started(self) -> bool:
        """True once the first spike beep has been detected (reliable plant confirmation)."""
        return len(self._beep_times) > 0

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
    Detects the defuse START sound (short click when E is pressed on the spike).

    On detection, onset_t is set to time.monotonic() and the detector goes deaf
    for _DEFUSE_COOLDOWN seconds (prevents re-triggering on echoes).
    onset_t is cleared only by reset() -- called from coach.py when round ends
    or when on_spike_resolved() fires.

    progress() returns wall-clock fraction (0.0-1.0) of the 7 s defuse window.
    Returns None when no defuse start has been detected this plant phase.

    Thread note: onset_t is a plain float/None attribute; CPython GIL makes
    reads from the main thread safe.
    """

    def __init__(self) -> None:
        self._long_rms: float = 1e-6
        self._armed: bool = False
        self._last_onset_t: float = 0.0   # prevents cooldown re-trigger
        # Public: read by audio_coach → coach.py → overlay
        self.onset_t: Optional[float] = None
        self._zi_defuse = sosfilt_zi(_DEFUSE_BP_SOS) * 0.0  # IIR filter state across chunks

    def reset(self) -> None:
        self._long_rms = 1e-6
        self._armed = False
        self._last_onset_t = 0.0
        self.onset_t = None
        self._zi_defuse = sosfilt_zi(_DEFUSE_BP_SOS) * 0.0

    def arm(self) -> None:
        """Call when spike is planted -- start listening for defuse start."""
        self._armed = True

    def process(self, mono: np.ndarray) -> None:
        """
        Scan one audio chunk for the defuse start onset.
        Sets self.onset_t on first detection; ignores audio for _DEFUSE_COOLDOWN s after.
        """
        if not self._armed or len(mono) < _DEFUSE_WIN:
            return

        # If already within cooldown window, skip (onset already locked)
        now = time.monotonic()
        if now - self._last_onset_t < _DEFUSE_COOLDOWN:
            return

        filtered, self._zi_defuse = sosfilt(_DEFUSE_BP_SOS, mono, zi=self._zi_defuse)

        step = _DEFUSE_WIN // 2
        for start in range(0, len(filtered) - _DEFUSE_WIN + 1, step):
            chunk = filtered[start : start + _DEFUSE_WIN]
            short_rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)

            # Update slow background
            self._long_rms = (1 - _DEFUSE_BG_ALPHA) * self._long_rms + _DEFUSE_BG_ALPHA * short_rms

            ratio = short_rms / (self._long_rms + 1e-9)
            if ratio >= _DEFUSE_THRESH:
                # Onset detected -- record and lock
                self.onset_t = now - (len(mono) - start) / SAMPLE_RATE
                self._last_onset_t = now
                return   # one onset per chunk

    def progress(self) -> Optional[float]:
        """
        Returns wall-clock defuse fraction 0.0-1.0, or None if not detected.
        Caller (coach.py) hides the UI and calls reset() when round ends.
        """
        onset = self.onset_t   # snapshot once; reset() on another thread can set it to None
        if onset is None:
            return None
        return min(1.0, (time.monotonic() - onset) / _DEFUSE_TIME)


# Half-defuse time (verified Valorant mechanic):
# After 3.5 s of defuse, progress is saved. If interrupted, only 3.5 s more needed.
_HALF_DEFUSE_TIME        = _DEFUSE_TIME / 2.0   # 3.5 s
_ADVICE_SUPPRESS_MARGIN  = 2.0   # extra buffer above defuse window before suppressing cooldown


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
      - half_defused flag: if enemy already has half-defuse saved (3.5s done),
        only 3.5s more are needed -- window is larger on restart.
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
        half_defused: bool = False,
    ) -> Optional[str]:
        """
        remaining     : seconds left on spike (from SpikeTimer.remaining())
        travel_time   : estimated seconds to reach spike
        half_defused  : True if defuse bar has passed 50% (3.5 s saved on enemy side)
                        -- reduces required defuse time to 3.5 s on next attempt

        Returns voice callout string or None.
        """
        now = time.monotonic()
        # If half-defuse progress was saved, enemy only needs 3.5s more to complete
        effective_defuse = _HALF_DEFUSE_TIME if half_defused else _DEFUSE_TIME
        needed = effective_defuse + travel_time + _DEFUSE_MARGIN
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
                if remaining > _DEFUSE_TIME + _DEFUSE_MARGIN + _ADVICE_SUPPRESS_MARGIN:
                    callout = None
            if callout is not None:
                self._last_advice_t = now

        return callout
