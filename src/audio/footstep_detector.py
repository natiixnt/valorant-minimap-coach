"""
Footstep onset detection from game audio.

Pipeline:
  1. Band-pass the stereo signal to 200-800 Hz (Valorant footstep core band).
     Community EQ data and competitive guides confirm 200-800 Hz for the "thud/weight",
     with impact transients at 2-4 kHz (critical for surface localization).
  2. Compute spectral flux (frame-by-frame L2 norm of positive spectral diff).
  3. Adaptive threshold: median + k * MAD  (robust to varying game audio levels).
  4. Debounce: minimum 120 ms between onsets.

Valorant-confirmed footstep data (Riot AMA + community EQ analysis):
  - Core detection band:   200-800 Hz
  - Transient/surface cue: 2-4 kHz (where human hearing is most sensitive to direction)
  - Walking max range:     ~15 m
  - Running max range:     ~50 m
  - Crouched max range:    ~12 m
  - Shift-walk:            silent (0 m)
  - Attenuation model:     deliberately FLAT (not inverse-square). Riot confirmed they
                           do not model realistic distance falloff -- footsteps stay
                           audible clearly even at max range.

Surface classification via spectral centroid of the full spectrum (not bandpassed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 48000         # Valorant native sample rate; match system to 48 kHz
FRAME_SIZE = 1024           # FFT frame
HOP_SIZE = 512              # hop between frames

# Bandpass for footstep energy (Valorant core detection band)
_BP_LOW_HZ = 200
_BP_HIGH_HZ = 800

# Onset detection
_FLUX_THRESHOLD_K = 2.5     # threshold = median + k * MAD
_MIN_ONSET_SEC = 0.12       # 120 ms minimum gap between onsets
_MIN_ONSET_SAMPLES = int(_MIN_ONSET_SEC * SAMPLE_RATE / HOP_SIZE)  # in frames (recalculated at 48 kHz)

# Surface spectral centroid breakpoints (Hz)
_SURFACE_THRESHOLDS = [
    (350,  "carpet"),    # < 350 Hz
    (650,  "wood"),      # 350-650 Hz
    (1100, "concrete"),  # 650-1100 Hz
    (float("inf"), "metal"),   # > 1100 Hz
]


@dataclass
class FootstepEvent:
    sample_idx: int          # sample index in the captured audio
    time_sec: float          # seconds since capture started
    surface: str             # "metal" | "wood" | "concrete" | "carpet" | "unknown"
    centroid_hz: float       # spectral centroid for reference
    amplitude_db: float      # RMS in dB (proxy for distance — quieter = farther)
    stereo_balance: float    # L-R energy ratio, -1.0 left … +1.0 right


class FootstepDetector:
    def __init__(self) -> None:
        self._sos = butter(4, [_BP_LOW_HZ, _BP_HIGH_HZ],
                           btype="band", fs=SAMPLE_RATE, output="sos")
        self._prev_spectrum: Optional[np.ndarray] = None
        self._flux_history: List[float] = []
        self._last_onset_frame = -_MIN_ONSET_SAMPLES - 1
        self._frame_count = 0
        self._sample_counter = 0

    # ------------------------------------------------------------------
    def process(self, stereo: np.ndarray) -> List[FootstepEvent]:
        """
        stereo: float32 array shaped (2, N) — left/right channels.
        Returns list of FootstepEvent detected in this chunk.
        """
        if stereo.shape[0] < 2 or stereo.shape[1] < FRAME_SIZE:
            return []

        left, right = stereo[0], stereo[1]
        mono = (left + right) * 0.5

        # Band-pass filter
        mono_bp = sosfilt(self._sos, mono)

        events: List[FootstepEvent] = []
        n = len(mono_bp)
        pos = 0

        while pos + FRAME_SIZE <= n:
            frame = mono_bp[pos: pos + FRAME_SIZE]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(FRAME_SIZE)))

            # Spectral flux (positive diff only)
            if self._prev_spectrum is not None and len(self._prev_spectrum) == len(spectrum):
                diff = spectrum - self._prev_spectrum
                flux = float(np.sum(np.maximum(diff, 0) ** 2))
            else:
                flux = 0.0
            self._prev_spectrum = spectrum
            self._flux_history.append(flux)

            # Keep history trimmed (last 5 s worth of frames)
            max_history = int(5 * SAMPLE_RATE / HOP_SIZE)
            if len(self._flux_history) > max_history:
                self._flux_history = self._flux_history[-max_history:]

            # Adaptive threshold
            if len(self._flux_history) >= 10:
                arr = np.array(self._flux_history)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                threshold = med + _FLUX_THRESHOLD_K * mad * 1.4826
            else:
                threshold = 1e9  # not enough history yet

            frames_since_last = self._frame_count - self._last_onset_frame

            if flux > threshold and frames_since_last >= _MIN_ONSET_SAMPLES and threshold > 0:
                # Onset confirmed
                sample_abs = self._sample_counter + pos
                time_sec = sample_abs / SAMPLE_RATE

                # Classify surface from full-spectrum centroid
                full_spec = np.abs(np.fft.rfft(mono[pos: pos + FRAME_SIZE]
                                                * np.hanning(FRAME_SIZE)))
                centroid_hz = _spectral_centroid(full_spec, SAMPLE_RATE)
                surface = _classify_surface(centroid_hz)

                # Amplitude
                rms = float(np.sqrt(np.mean(frame ** 2)) + 1e-9)
                amp_db = float(20 * np.log10(rms))

                # Stereo balance over this frame
                l_rms = float(np.sqrt(np.mean(left[pos: pos + FRAME_SIZE] ** 2)) + 1e-9)
                r_rms = float(np.sqrt(np.mean(right[pos: pos + FRAME_SIZE] ** 2)) + 1e-9)
                balance = float((r_rms - l_rms) / (r_rms + l_rms))

                events.append(FootstepEvent(
                    sample_idx=sample_abs,
                    time_sec=time_sec,
                    surface=surface,
                    centroid_hz=centroid_hz,
                    amplitude_db=amp_db,
                    stereo_balance=balance,
                ))
                self._last_onset_frame = self._frame_count

            pos += HOP_SIZE
            self._frame_count += 1

        self._sample_counter += n
        return events


# ------------------------------------------------------------------
def _spectral_centroid(spectrum: np.ndarray, sr: int) -> float:
    freqs = np.fft.rfftfreq(2 * (len(spectrum) - 1), d=1.0 / sr)
    total = float(np.sum(spectrum))
    if total < 1e-12:
        return 0.0
    return float(np.dot(freqs, spectrum) / total)


def _classify_surface(centroid_hz: float) -> str:
    for threshold, label in _SURFACE_THRESHOLDS:
        if centroid_hz < threshold:
            return label
    return "metal"
