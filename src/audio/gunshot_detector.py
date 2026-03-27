"""
Gunshot detection and spatial localization.

Gunshots in Valorant have a distinctive wideband crack (500 Hz-8 kHz) with:
  - Very fast rise time < 1 ms (sharp attack transient)
  - High amplitude (typically > 40x above background RMS)
  - Short duration: transient 10-50 ms, echo/reverb up to ~150 ms
  - Suppressed weapons: much quieter but same spectral shape

ITD + ILD azimuth estimation works well on gunshots because:
  - The initial crack is very time-aligned (sharp impulse)
  - High amplitude gives good L/R ratio measurement

Distance estimation:
  Valorant's flat attenuation means amplitude-distance is unreliable for guns
  just as for footsteps. However, the ratio of direct vs reverb energy (DRR)
  is a weak proxy: close shots have more direct energy, far shots more reverb.
  We use a simplified direct/reverb energy split: first 15 ms = direct, rest = reverb.

Suppressor detection:
  Suppressed weapons have ~12-15 dB lower amplitude and slightly different
  spectral shape (attenuated highs above 4 kHz). We flag shots below a
  threshold as "suppressed -- shooter may be using Vandal/Phantom silenced".

Output: GunEvent dataclass per detected shot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 48000

# Gunshot detection
_GUN_LOW_HZ     = 500
_GUN_HIGH_HZ    = 8000
_RISE_MS        = 2          # samples over which to measure rise time
_RISE_N         = int(_RISE_MS * SAMPLE_RATE / 1000)
_AMPLITUDE_MULT = 30.0       # min ratio of peak to background RMS for trigger
_MIN_GAP_MS     = 150        # minimum ms between separate shots
_MIN_GAP_N      = int(_MIN_GAP_MS * SAMPLE_RATE / 1000)

# Suppressed threshold: if peak is < this factor above background, flag as suppressed
_SUPPRESSED_MULT = 8.0

# Analysis window after onset (ms)
_ANALYSIS_MS = 60
_ANALYSIS_N  = int(_ANALYSIS_MS * SAMPLE_RATE / 1000)
_DIRECT_MS   = 15    # first N ms = direct sound for DRR
_DIRECT_N    = int(_DIRECT_MS * SAMPLE_RATE / 1000)

# ITD max lag (same as footstep estimator)
_MAX_LAG = 32


@dataclass
class GunEvent:
    time_sec: float
    azimuth_deg: float       # -180..+180, right-positive relative to player facing
    suppressed: bool
    amplitude_db: float      # peak level in dB
    distance_hint: str       # "close" | "medium" | "far" (rough from DRR)
    voice: str


class GunDetector:
    def __init__(self) -> None:
        self._sos = butter(4, [_GUN_LOW_HZ, _GUN_HIGH_HZ],
                           btype="band", fs=SAMPLE_RATE, output="sos")
        self._bg_rms_sq   = 0.0
        self._bg_count    = 0
        self._last_onset  = -_MIN_GAP_N
        self._sample_count = 0

    def process(
        self, stereo: np.ndarray
    ) -> List[GunEvent]:
        """
        stereo: float32 (2, N). Returns list of GunEvent for this chunk.
        """
        if stereo.shape[0] < 2 or stereo.shape[1] < _ANALYSIS_N:
            return []

        left, right = stereo[0], stereo[1]
        mono = (left + right) * 0.5
        filtered = sosfilt(self._sos, mono)

        events: List[GunEvent] = []
        n = len(filtered)
        pos = 0
        hop = _RISE_N

        while pos + _ANALYSIS_N <= n:
            window = filtered[pos: pos + _RISE_N]
            peak = float(np.max(np.abs(window)))

            # Update background RMS (long-term, slow)
            rms_sq = float(np.mean(window ** 2))
            self._bg_rms_sq = (self._bg_rms_sq * 0.998 + rms_sq * 0.002)
            bg_rms = float(np.sqrt(self._bg_rms_sq + 1e-10))

            gap_since_last = (self._sample_count + pos) - self._last_onset

            if (peak > bg_rms * _AMPLITUDE_MULT
                    and gap_since_last >= _MIN_GAP_N
                    and bg_rms > 1e-7):
                # Onset confirmed -- analyze window
                analysis = filtered[pos: pos + _ANALYSIS_N]
                l_win = left[pos: pos + _ANALYSIS_N]
                r_win = right[pos: pos + _ANALYSIS_N]

                az = self._azimuth(l_win, r_win)

                # DRR distance hint
                direct_e = float(np.mean(analysis[:_DIRECT_N] ** 2))
                reverb_e = float(np.mean(analysis[_DIRECT_N:] ** 2))
                drr = direct_e / (reverb_e + 1e-12)
                if drr > 3.0:
                    dist_hint = "close"
                elif drr > 1.0:
                    dist_hint = "medium"
                else:
                    dist_hint = "far"

                amp_db = float(20 * np.log10(peak + 1e-9))
                suppressed = peak < bg_rms * _SUPPRESSED_MULT * 3

                voice = self._build_voice(az, dist_hint, suppressed)
                t = (self._sample_count + pos) / SAMPLE_RATE

                events.append(GunEvent(
                    time_sec=t,
                    azimuth_deg=az,
                    suppressed=suppressed,
                    amplitude_db=amp_db,
                    distance_hint=dist_hint,
                    voice=voice,
                ))
                self._last_onset = self._sample_count + pos

            pos += hop

        self._sample_count += n
        return events

    def _azimuth(self, left: np.ndarray, right: np.ndarray) -> float:
        n = min(len(left), len(right), _ANALYSIS_N)
        l = left[:n]
        r = right[:n]

        # ITD via cross-correlation
        L = np.fft.rfft(l, n=2 * n)
        R = np.fft.rfft(r, n=2 * n)
        xcorr = np.fft.irfft(L * np.conj(R))
        lags = np.concatenate([np.arange(0, _MAX_LAG + 1), np.arange(-_MAX_LAG, 0)])
        search = np.concatenate([xcorr[:_MAX_LAG + 1], xcorr[-_MAX_LAG:]])
        lag = int(lags[np.argmax(search)])
        az_itd = float(lag) / _MAX_LAG * 90.0

        # ILD
        l_rms = float(np.sqrt(np.mean(l ** 2)) + 1e-9)
        r_rms = float(np.sqrt(np.mean(r ** 2)) + 1e-9)
        ild_db = 20 * np.log10(r_rms / l_rms)
        az_ild = float(np.clip(ild_db / 6.0, -1.0, 1.0)) * 90.0

        return float(np.clip(0.6 * az_itd + 0.4 * az_ild, -180.0, 180.0))

    @staticmethod
    def _build_voice(az: float, dist: str, suppressed: bool) -> str:
        if -30 < az <= 30:
            direction = "ahead"
        elif 30 < az <= 90:
            direction = "front right"
        elif 90 < az <= 150:
            direction = "right"
        elif az > 150 or az <= -150:
            direction = "behind"
        elif -150 < az <= -90:
            direction = "left"
        else:
            direction = "front left"

        gun_type = "suppressed shot" if suppressed else "shot"
        return f"{gun_type} {direction}, {dist}"
