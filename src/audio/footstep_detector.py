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

from collections import deque
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

# Multi-feature surface classification thresholds.
# Using three features (centroid, rolloff, ZCR) instead of centroid alone resolves
# cases where e.g. a hollow wood thump and a concrete step share similar centroids
# but differ in how broadly their energy is distributed (rolloff) and how noisy
# the waveform is (ZCR -- metal rings with rapid sign changes, carpet is smooth).
_CARPET_MAX_CENTROID  = 450    # Hz: carpet absorbs high frequencies, centroid sits very low
_CARPET_MAX_ROLLOFF   = 1500   # Hz: 85% of carpet energy is below 1.5 kHz
_METAL_MIN_CENTROID   = 1100   # Hz: metallic ring shifts spectral weight above 1 kHz
_METAL_MIN_ZCR        = 0.12   # ratio: rapid waveform oscillations (ring) produce many sign flips
_WOOD_MAX_CENTROID    = 950    # Hz: resonant hollow thump stays mid-range (above carpet, below metal)
_WOOD_MAX_ROLLOFF     = 4000   # Hz: broader than carpet but narrower than concrete/metal spread


@dataclass
class FootstepEvent:
    sample_idx: int          # sample index in the captured audio
    time_sec: float          # seconds since capture started
    surface: str             # "metal" | "wood" | "concrete" | "carpet" | "unknown"
    centroid_hz: float       # spectral centroid for reference
    amplitude_db: float      # RMS in dB (proxy for distance — quieter = farther)
    stereo_balance: float    # L-R energy ratio, -1.0 left … +1.0 right


_HANNING_WINDOW = np.hanning(FRAME_SIZE)
_FREQ_BINS      = np.fft.rfftfreq(FRAME_SIZE, d=1.0 / SAMPLE_RATE)


class FootstepDetector:
    def __init__(self) -> None:
        self._sos = butter(4, [_BP_LOW_HZ, _BP_HIGH_HZ],
                           btype="band", fs=SAMPLE_RATE, output="sos")
        _max_history = int(5 * SAMPLE_RATE / HOP_SIZE)
        self._prev_spectrum: Optional[np.ndarray] = None
        self._flux_history: deque = deque(maxlen=_max_history)
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

        # sosfilt without zi: stateless per call - each 700 ms chunk is processed
        # independently so there is no cross-chunk signal to preserve filter state for.
        # Band-pass filter
        mono_bp = sosfilt(self._sos, mono)

        events: List[FootstepEvent] = []
        n = len(mono_bp)
        pos = 0

        while pos + FRAME_SIZE <= n:
            frame = mono_bp[pos: pos + FRAME_SIZE]
            spectrum = np.abs(np.fft.rfft(frame * _HANNING_WINDOW))

            # Spectral flux (positive diff only)
            if self._prev_spectrum is not None and len(self._prev_spectrum) == len(spectrum):
                diff = spectrum - self._prev_spectrum
                flux = float(np.sum(np.maximum(diff, 0) ** 2))
            else:
                flux = 0.0
            self._prev_spectrum = spectrum
            self._flux_history.append(flux)

            # Adaptive threshold using MAD (Median Absolute Deviation).
            # MAD = median(|x_i - median(x)|); it is the median analog of std deviation
            # and is far more resistant to outliers (like a loud gunshot frame) than mean/std.
            # The 1.4826 factor is the consistency constant that makes MAD a statistically
            # consistent estimator of sigma for a normal distribution (1/qnorm(0.75) ≈ 1.4826),
            # so that (k * MAD * 1.4826) behaves like k standard deviations above the median.
            if len(self._flux_history) >= 10:
                arr = np.array(self._flux_history)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                threshold = med + _FLUX_THRESHOLD_K * mad * 1.4826
            else:
                threshold = 1e9  # not enough history yet

            # Use frame count rather than wall time for the debounce gap: frame count is
            # deterministic regardless of system load or sleep jitter, so the 120 ms
            # minimum onset gap stays exact even if the analysis thread runs behind.
            frames_since_last = self._frame_count - self._last_onset_frame

            if flux > threshold and frames_since_last >= _MIN_ONSET_SAMPLES and threshold > 0:
                # Onset confirmed
                sample_abs = self._sample_counter + pos
                time_sec = sample_abs / SAMPLE_RATE

                # Classify surface using centroid + rolloff + zero-crossing rate
                full_frame = mono[pos: pos + FRAME_SIZE]
                full_spec  = np.abs(np.fft.rfft(full_frame * _HANNING_WINDOW))
                surface = _classify_surface_multi(full_spec, full_frame)

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

    def reset(self) -> None:
        """Call at round start to clear accumulated flux history and debounce state."""
        self._prev_spectrum = None
        self._flux_history.clear()
        self._last_onset_frame = -_MIN_ONSET_SAMPLES - 1
        self._frame_count = 0
        self._sample_counter = 0


# ------------------------------------------------------------------
def _spectral_centroid(spectrum: np.ndarray, sr: int) -> float:
    total = float(np.sum(spectrum))
    if total < 1e-12:
        return 0.0
    return float(np.dot(_FREQ_BINS, spectrum) / total)


def _spectral_rolloff(spectrum: np.ndarray, sr: int, roll_percent: float = 0.85) -> float:
    """Frequency below which roll_percent of total spectral energy lies.

    Low rolloff: energy concentrated in low frequencies (carpet, heavy thud).
    High rolloff: energy spread broadly across spectrum (concrete, metal).
    """
    total = float(np.sum(spectrum))
    if total < 1e-12:
        return 0.0
    cumsum = np.cumsum(spectrum)
    idx = int(np.searchsorted(cumsum, total * roll_percent))
    idx = min(idx, len(_FREQ_BINS) - 1)
    return float(_FREQ_BINS[idx])


def _zero_crossing_rate(frame: np.ndarray) -> float:
    """Normalized rate of sign changes in the time-domain signal.

    High ZCR: noisy, clangy waveform (metal surfaces ring with rapid oscillations).
    Low ZCR: smooth, tonal waveform (carpet absorbs high-frequency oscillations).
    """
    if len(frame) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * len(frame)))


def _classify_surface_multi(spectrum: np.ndarray, frame: np.ndarray) -> str:
    """Multi-feature surface classification using centroid, rolloff, and ZCR.

    Three features used because centroid alone is ambiguous:
    - A hollow wood thump and a concrete impact can share similar centroids
      but differ in rolloff (wood resonates broadly, concrete is more impulsive)
    - Metal and concrete both have high centroids but metal has a much higher ZCR
      due to the metallic ring producing rapid waveform oscillations
    """
    centroid_hz = _spectral_centroid(spectrum, SAMPLE_RATE)
    rolloff_hz  = _spectral_rolloff(spectrum, SAMPLE_RATE)
    zcr         = _zero_crossing_rate(frame)

    # Carpet: heavily absorbed surface, very little high-frequency content
    if centroid_hz < _CARPET_MAX_CENTROID and rolloff_hz < _CARPET_MAX_ROLLOFF:
        return "carpet"

    # Metal: hard ringing surface, bright spectrum, many waveform sign changes
    if centroid_hz > _METAL_MIN_CENTROID and zcr > _METAL_MIN_ZCR:
        return "metal"

    # Wood: resonant thump, moderate centroid, broader-than-carpet rolloff
    if centroid_hz < _WOOD_MAX_CENTROID and rolloff_hz < _WOOD_MAX_ROLLOFF:
        return "wood"

    return "concrete"
