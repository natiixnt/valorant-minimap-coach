"""
Transient noise gate for footstep detection.

Gunshots, grenade explosions, and ability sounds create loud transient spikes
that trigger false footstep onsets. This gate suppresses audio during transients
so FootstepDetector only processes quiet inter-transient periods.

Algorithm:
  1. Compute short-term RMS over a 5 ms window.
  2. Compare to a slow-moving background RMS (200 ms window).
  3. If short_rms > background_rms * _GATE_RATIO, we're in a transient.
  4. Gate output: zero the signal for the duration of the transient + _RELEASE_MS tail.

Tuning:
  _GATE_RATIO = 8.0  -- trigger 8x above background (gunshot is ~20-40x louder)
  _RELEASE_MS = 80   -- keep gate closed for 80 ms after transient end
                        (gunshot reverb / echo dies out by then)

Returns a gated copy of the audio (same shape, same dtype).
Does NOT modify the input array.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 48000

_SHORT_MS    = 5       # short-term RMS window
_LONG_MS     = 200     # background RMS window
_GATE_RATIO  = 8.0     # trigger threshold multiplier
_RELEASE_MS  = 80      # gate hold after transient

_SHORT_N   = int(_SHORT_MS   * SAMPLE_RATE / 1000)
_LONG_N    = int(_LONG_MS    * SAMPLE_RATE / 1000)
_RELEASE_N = int(_RELEASE_MS * SAMPLE_RATE / 1000)


# Ordering note: GunDetector must run on RAW audio BEFORE this gate because
# gunshots ARE the transients we want to suppress here. Running the gate first
# would zero out the very events the gunshot detector needs to find.
# Conversely, FootstepDetector runs AFTER the gate so that explosion/gunshot
# transients cannot generate spurious footstep onsets.
class NoiseGate:
    def __init__(self) -> None:
        self._short_sq_sum = 0.0
        self._long_sq_sum  = 0.0
        self._short_buf    = np.zeros(_SHORT_N, dtype=np.float32)
        self._long_buf     = np.zeros(_LONG_N,  dtype=np.float32)
        self._short_idx    = 0
        self._long_idx     = 0
        self._release_left = 0   # samples remaining in release phase

    def process(self, mono: np.ndarray) -> np.ndarray:
        """
        mono: float32 array shape (N,)
        Returns gated copy; transient regions zeroed.
        """
        out = mono.copy()
        for i in range(len(mono)):
            s = mono[i]

            # Running squared-sum buffers: subtract the leaving sample, add the new one.
            # This avoids recomputing the whole window sum on every sample (O(1) vs O(N)).
            # Update short-term RMS buffer
            old_s = self._short_buf[self._short_idx]
            self._short_sq_sum += s * s - old_s * old_s
            self._short_buf[self._short_idx] = s
            self._short_idx = (self._short_idx + 1) % _SHORT_N

            # Update long-term RMS buffer (200 ms) - this tracks the ambient background,
            # not instantaneous amplitude, so gunshots don't distort the reference level.
            old_l = self._long_buf[self._long_idx]
            self._long_sq_sum += s * s - old_l * old_l
            self._long_buf[self._long_idx] = s
            self._long_idx = (self._long_idx + 1) % _LONG_N

            short_rms = float(np.sqrt(max(0.0, self._short_sq_sum / _SHORT_N)))
            long_rms  = float(np.sqrt(max(0.0, self._long_sq_sum  / _LONG_N)))

            # Attack: re-arm the release counter every time a new transient peak is
            # detected - this extends the gate if reverb keeps the signal elevated.
            if short_rms > long_rms * _GATE_RATIO and long_rms > 1e-7:
                # Transient detected -- open gate for release period
                self._release_left = _RELEASE_N

            # Release: count down sample by sample; gate stays open for _RELEASE_N
            # samples after the last transient peak to swallow the reverb tail.
            if self._release_left > 0:
                out[i] = 0.0
                self._release_left -= 1

        return out

    def reset(self) -> None:
        self._short_sq_sum = 0.0
        self._long_sq_sum  = 0.0
        self._short_buf[:] = 0.0
        self._long_buf[:]  = 0.0
        self._short_idx    = 0
        self._long_idx     = 0
        self._release_left = 0
