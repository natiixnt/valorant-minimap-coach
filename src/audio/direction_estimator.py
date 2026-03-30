"""
Stereo direction estimation for footstep events.

Combines two cues:
  ITD  — Interaural Time Delay.   Cross-correlation peak lag between L/R channels.
         Good at low frequencies (< 1.5 kHz).  Max lag at human head: ~0.65 ms
         at 48 kHz = ±31 samples.

  ILD  — Interaural Level Difference.  Log ratio of L/R RMS energy.
         Good at high frequencies (> 1.5 kHz).

Valorant audio engine (confirmed via Riot AMA + community):
  - Wwise + THX Spatial Audio HRTF (introduced Patch 2.06).
  - Fixed single-profile HRTF (not personalized).
  - Pre-HRTF: sounds at 45° front-left and 45° rear-left were indistinguishable.
  - Post-HRTF: full 3D sphere, elevation cues available.
  - Running footsteps: audible up to ~50 m.
  - Walking: ~15 m.  Crouched: ~12 m.  Shift-walk: silent.

IMPORTANT -- Valorant uses a FLAT attenuation model by design (confirmed by Riot).
Volume does NOT fall off realistically with distance. A footstep at 5 m and at 40 m
can have similar amplitude in the mix. Amplitude-based distance estimates are therefore
unreliable and should be treated as a rough lower-bound (quieter = farther, but not
accurate). The _estimate_distance() method provides a best-effort value only.

Output azimuth is in degrees:
  0° = straight ahead (same direction player is facing)
  +90° = right
  -90° = left
  ±180° = directly behind
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 48000   # Valorant native sample rate

# Max ITD lag for cross-correlation search (samples)
# Human head ~0.65 ms → 48000 * 0.00065 ≈ 31 samples
_MAX_LAG = 32

# Mixing weights for ITD vs ILD azimuth
_ITD_WEIGHT = 0.6
_ILD_WEIGHT = 0.4

# Distance calibration (unreliable due to Valorant flat attenuation -- use as rough estimate only)
_REF_AMP_DB = -18.0   # dB RMS expected at ~5 m reference distance
_REF_DIST_M = 5.0
_MAX_DIST_M = 50.0    # running footsteps audible up to ~50 m per Riot

# Multi-band ILD: the acoustic head shadow grows with frequency, so different bands
# contribute different amounts of directional information.
# Below 1 kHz: head is too small relative to wavelength, ILD near zero for all angles.
# 1-2 kHz: shadow starts emerging.  2-4 kHz: strongest, most consistent shadow.
# 4-8 kHz: strong shadow but HRTF pinna notches can distort the raw level reading.
_ILD_BANDS = [
    (1000, 2000, 0.25),   # emerging shadow, moderate reliability
    (2000, 4000, 0.45),   # strongest shadow, most consistent cue - highest weight
    (4000, 8000, 0.30),   # strong shadow, but HRTF colorization reduces reliability
]

# Front-back disambiguation via HRTF spectral notch cues.
# Valorant's THX Spatial Audio HRTF encodes elevation/front-back as spectral coloration:
# sounds from behind produce a notch (energy dip) in the 4-10 kHz band caused by
# pinna (outer ear) reflections. We compare energy in the notch band vs a reference
# band to detect when a "directly ahead" estimate should actually be "directly behind".
_HRTF_NOTCH_LOW_HZ   = 4000
_HRTF_NOTCH_HIGH_HZ  = 10000
_HRTF_REF_LOW_HZ     = 2000
_HRTF_REF_HIGH_HZ    = 4000
_FRONT_BACK_NOTCH_DB = -3.0   # if notch band is >3 dB below reference: rear sound


class DirectionEstimator:
    """Per-event direction estimator. Filter coefficients precomputed in __init__."""

    def __init__(self) -> None:
        # One filter per ILD band (computed once, applied stateless per call)
        self._ild_sos_bands = [
            butter(4, [lo, hi], btype="band", fs=SAMPLE_RATE, output="sos")
            for lo, hi, _ in _ILD_BANDS
        ]
        # Bandpass filters for HRTF front-back notch detection
        self._hrtf_sos = butter(4, [_HRTF_NOTCH_LOW_HZ, _HRTF_NOTCH_HIGH_HZ],
                                btype="band", fs=SAMPLE_RATE, output="sos")
        self._ref_sos  = butter(4, [_HRTF_REF_LOW_HZ, _HRTF_REF_HIGH_HZ],
                                btype="band", fs=SAMPLE_RATE, output="sos")

    def estimate(
        self,
        stereo: np.ndarray,      # (2, N) float32 around the onset
        amplitude_db: float,
    ) -> tuple[float, float]:
        """
        Returns (azimuth_deg, distance_m).

        azimuth_deg: -180..+180, positive = right side of player
        distance_m:  rough estimate 1-25 m
        """
        left = stereo[0]
        right = stereo[1]

        az_itd, itd_conf = self._itd_azimuth(left, right)
        az_ild = self._ild_azimuth(left, right)

        # Confidence-weighted fusion: when the GCC-PHAT peak is sharp and unambiguous,
        # lean on ITD (weight up to 0.6); when the peak is flat or noisy (reverb, poor
        # stereo separation), fall back to ILD which is more robust in those conditions.
        # The two weights always sum to 1.0, so the result stays in [-180, +180].
        itd_w = _ITD_WEIGHT * itd_conf
        ild_w = 1.0 - itd_w
        azimuth = itd_w * az_itd + ild_w * az_ild
        azimuth = max(-180.0, min(180.0, azimuth))

        # Disambiguate front vs back using HRTF spectral notch cues
        azimuth = self._front_back(left, right, azimuth)

        distance = self._estimate_distance(amplitude_db)
        return azimuth, distance

    # ------------------------------------------------------------------
    def _itd_azimuth(self, left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
        """Frequency-weighted GCC-PHAT -> (azimuth_deg, confidence).

        Two improvements over plain GCC-PHAT:
        1. Frequency weighting: after whitening, upweight the 1-4 kHz band where
           footstep transients carry the most reliable timing information. This is a
           hybrid between pure whitening (all freqs equal) and a band-limited correlator.
        2. Sub-sample parabolic interpolation: the integer-sample lag gives ~2.9 deg
           resolution per sample. Fitting a parabola through the peak and its two
           neighbors finds the fractional lag, improving to ~0.5-1 deg effective precision.

        Also returns a confidence score (0-1) based on the peak-to-noise ratio of the
        correlation output, used by estimate() to weight ITD vs ILD adaptively.
        """
        n = min(len(left), len(right))
        if n < _MAX_LAG * 2:
            return 0.0, 0.0

        l = left[:n] - left[:n].mean()
        r = right[:n] - right[:n].mean()

        L = np.fft.rfft(l, n=2 * n)
        R = np.fft.rfft(r, n=2 * n)
        cross = L * np.conj(R)
        # Whiten the cross-spectrum (GCC-PHAT), then additionally upweight 1-4 kHz --
        # the band where footstep impacts produce their sharpest timing cues.
        cross_phat = cross / (np.abs(cross) + 1e-10)
        freqs = np.fft.rfftfreq(2 * n, d=1.0 / SAMPLE_RATE)
        freq_w = np.where((freqs >= 1000) & (freqs <= 4000), 2.0, 1.0)
        xcorr = np.fft.irfft(cross_phat * freq_w)

        search = np.concatenate([xcorr[: _MAX_LAG + 1], xcorr[-_MAX_LAG:]])
        lags   = np.concatenate([np.arange(0, _MAX_LAG + 1), np.arange(-_MAX_LAG, 0)])
        peak_idx = int(np.argmax(search))
        lag_float = float(lags[peak_idx])

        # Parabolic sub-sample interpolation: skip at the wrap boundary where lags
        # jump discontinuously from +MAX_LAG to -MAX_LAG (neighbors are not adjacent).
        at_wrap = (peak_idx == 0 or peak_idx >= len(search) - 1
                   or peak_idx == _MAX_LAG or peak_idx == _MAX_LAG + 1)
        if not at_wrap:
            y1, y2, y3 = search[peak_idx - 1], search[peak_idx], search[peak_idx + 1]
            denom = y1 - 2.0 * y2 + y3
            if abs(denom) > 1e-10:
                delta = float(np.clip(0.5 * (y1 - y3) / denom, -0.5, 0.5))
                lag_float += delta

        az = lag_float / _MAX_LAG * 90.0

        # Confidence: how many times above the noise floor is the correlation peak.
        # A clean, sharp peak (single source, good stereo separation) gives confidence
        # near 1.0; a flat or noisy correlation (reverb, multiple sources) gives ~0.
        noise_floor = float(np.mean(np.abs(search)))
        confidence = float(np.clip(search[peak_idx] / (noise_floor * 4.0 + 1e-10), 0.0, 1.0))

        return az, confidence

    def _ild_azimuth(self, left: np.ndarray, right: np.ndarray) -> float:
        """Multi-band ILD -> azimuth degrees.

        Computes the level difference separately in three frequency bands and combines
        them with frequency-dependent weights. A single wide-band measurement is pulled
        toward zero by the low frequencies where ILD carries no directional information;
        splitting into bands lets each band contribute proportionally to its reliability.
        """
        az_total = 0.0
        for (lo, hi, weight), sos in zip(_ILD_BANDS, self._ild_sos_bands):
            l_bp = sosfilt(sos, left)
            r_bp = sosfilt(sos, right)
            l_rms = float(np.sqrt(np.mean(l_bp ** 2)) + 1e-9)
            r_rms = float(np.sqrt(np.mean(r_bp ** 2)) + 1e-9)
            ild_db = 20 * np.log10(r_rms / l_rms)
            az_band = float(np.clip(ild_db / 6.0, -1.0, 1.0) * 90.0)
            az_total += weight * az_band
        return az_total

    def _front_back(self, left: np.ndarray, right: np.ndarray, az: float) -> float:
        """Disambiguate front vs rear using HRTF spectral notch cues.

        When a sound is directly ahead or behind, ITD and ILD are both near zero --
        pure timing/level cues cannot distinguish 0° from ±180°. Valorant's THX Spatial
        Audio HRTF encodes front vs back via spectral coloration: the outer ear (pinna)
        reflects and diffracts the 4-10 kHz band differently for rear sounds, creating
        a spectral notch (energy dip) that frontal sounds do not have.

        We only attempt correction within ±45° of the front axis, where the ambiguity
        actually exists. Lateral sounds (45°-135°) have reliable ITD/ILD and are skipped.
        """
        if abs(az) >= 45.0:
            return az   # clearly lateral or already rear: no ambiguity

        mono   = (left + right) * 0.5
        # Use 2-4 kHz as the reference band - high enough to be in the HRTF-affected
        # range yet below the notch band, so it represents the "unnotched" baseline.
        ref_e  = float(np.mean(sosfilt(self._ref_sos,  mono) ** 2))
        if ref_e < 1e-8:
            return az   # not enough high-frequency energy for reliable spectral analysis

        # 4-10 kHz energy relative to the 2-4 kHz baseline: rear pinna reflections
        # create a dip in this band. If the ratio drops below -3 dB the signature
        # matches a rear-origin sound in Valorant's fixed THX HRTF profile.
        notch_e  = float(np.mean(sosfilt(self._hrtf_sos, mono) ** 2))
        ratio_db = 10.0 * np.log10(notch_e / (ref_e + 1e-12))

        # Notch detected: rear HRTF signature present, flip to behind
        if ratio_db < _FRONT_BACK_NOTCH_DB:
            az = 180.0 if az >= 0.0 else -180.0

        return az

    def _estimate_distance(self, amplitude_db: float) -> float:
        """Inverse-square law distance estimate."""
        delta_db = _REF_AMP_DB - amplitude_db   # positive = quieter = farther
        distance = _REF_DIST_M * (10 ** (delta_db / 20.0))
        return float(np.clip(distance, 1.0, _MAX_DIST_M))


def audio_az_to_map_direction(audio_azimuth: float, player_facing_deg: float) -> float:
    """
    Convert audio azimuth (relative to player) to map compass direction.

    audio_azimuth: -180..+180, right-positive
    player_facing_deg: 0=up/north on map, clockwise

    Returns map_direction 0-360.
    """
    map_dir = (player_facing_deg + audio_azimuth) % 360
    return map_dir


def direction_to_map_pos(
    player_pos: tuple[float, float],
    map_direction_deg: float,
    distance_m: float,
    scale_m_per_unit: float = 1.0,
) -> tuple[float, float]:
    """
    Project estimated enemy position onto normalized map coordinates.

    player_pos: (x, y) normalized 0-1 on the minimap
    map_direction_deg: 0=up, clockwise
    distance_m: estimated distance
    scale_m_per_unit: how many meters per normalized map unit (calibrate per map)

    Returns estimated (x, y) normalized position.
    """
    rad = np.deg2rad(map_direction_deg)
    # map: 0° = up = -y direction, 90° = right = +x
    dx = np.sin(rad) * distance_m / scale_m_per_unit
    dy = -np.cos(rad) * distance_m / scale_m_per_unit
    ex = float(np.clip(player_pos[0] + dx, 0.0, 1.0))
    ey = float(np.clip(player_pos[1] + dy, 0.0, 1.0))
    return ex, ey
