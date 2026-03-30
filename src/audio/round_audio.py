"""
Audio-based round event detection.

Detects round boundaries from the game audio stream:
  - Round start horn    : loud, brief 400-800 Hz burst (buy phase ends)
  - Spike plant sound   : distinct 1000-2000 Hz click/beep sequence
  - Round win jingle    : bright, multi-tone 1-3 kHz signature
  - Round loss jingle   : lower, minor-key 500-1500 Hz signature

All detection uses spectral energy ratios -- no ML needed, just tuned
bandpass energy comparison.

Outputs via callbacks set by the caller:
    detector.on_round_start = round_state.on_round_start_sound
    detector.on_round_end   = round_state.on_round_end_sound

Called from AudioCoach._analysis_loop() once per audio chunk.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 48000

# Energy ratio thresholds (band energy / total energy)
_HORN_LOW_HZ   = 350
_HORN_HIGH_HZ  = 900
_HORN_RATIO    = 0.55     # horn concentrates energy in this band
_HORN_AMP_DB   = -22.0    # minimum loudness (game must be audible)
_HORN_MIN_SEC  = 0.08     # minimum horn duration
_HORN_COOLDOWN = 8.0      # seconds between round-start detections

_WIN_LOW_HZ    = 900
_WIN_HIGH_HZ   = 3000
_WIN_RATIO     = 0.60
_WIN_AMP_DB    = -28.0

_LOSS_LOW_HZ   = 300
_LOSS_HIGH_HZ  = 900
_LOSS_RATIO    = 0.55
_LOSS_AMP_DB   = -28.0

_END_COOLDOWN  = 8.0      # seconds between round-end detections


def _bandpass(lo: int, hi: int) -> object:
    return butter(4, [lo, hi], btype="band", fs=SAMPLE_RATE, output="sos")


class RoundAudioDetector:
    def __init__(self) -> None:
        self._sos_horn = _bandpass(_HORN_LOW_HZ, _HORN_HIGH_HZ)
        self._sos_win  = _bandpass(_WIN_LOW_HZ,  _WIN_HIGH_HZ)
        self._sos_loss = _bandpass(_LOSS_LOW_HZ, _LOSS_HIGH_HZ)

        self._last_start = 0.0
        self._last_end   = 0.0
        self._horn_frames = 0     # consecutive frames detecting the horn

        self.on_round_start: Optional[Callable[[], None]] = None
        self.on_round_end:   Optional[Callable[[], None]] = None

    def process(self, mono: np.ndarray) -> None:
        """
        Call with each mono audio chunk (float32, 48 kHz).
        Fires callbacks when events are detected.
        """
        now = time.monotonic()
        rms_db = float(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9))
        total_energy = float(np.mean(mono ** 2)) + 1e-12

        # --- Round start horn
        if now - self._last_start > _HORN_COOLDOWN:
            horn_filtered = sosfilt(self._sos_horn, mono)
            horn_energy = float(np.mean(horn_filtered ** 2))
            horn_ratio = horn_energy / total_energy
            if horn_ratio > _HORN_RATIO and rms_db > _HORN_AMP_DB:
                self._horn_frames += 1
                if self._horn_frames >= 3:  # ~60 ms of horn
                    self._horn_frames = 0
                    self._last_start = now
                    if self.on_round_start:
                        self.on_round_start()
            else:
                self._horn_frames = 0
        else:
            self._horn_frames = 0

        # --- Round end (win)
        if now - self._last_end > _END_COOLDOWN and rms_db > _WIN_AMP_DB:
            win_filtered = sosfilt(self._sos_win, mono)
            win_energy = float(np.mean(win_filtered ** 2))
            win_ratio = win_energy / total_energy
            if win_ratio > _WIN_RATIO:
                self._last_end = now
                if self.on_round_end:
                    self.on_round_end()
                return

        # --- Round end (loss)
        if now - self._last_end > _END_COOLDOWN and rms_db > _LOSS_AMP_DB:
            loss_filtered = sosfilt(self._sos_loss, mono)
            loss_energy = float(np.mean(loss_filtered ** 2))
            loss_ratio = loss_energy / total_energy
            if loss_ratio > _LOSS_RATIO:
                self._last_end = now
                if self.on_round_end:
                    self.on_round_end()
