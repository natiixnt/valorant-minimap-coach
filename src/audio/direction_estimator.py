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


class DirectionEstimator:
    """Stateless: call estimate() per footstep event + surrounding audio window."""

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

        az_itd = self._itd_azimuth(left, right)
        az_ild = self._ild_azimuth(left, right)

        # Weighted combination
        azimuth = _ITD_WEIGHT * az_itd + _ILD_WEIGHT * az_ild
        # Clamp to [-180, 180]
        azimuth = max(-180.0, min(180.0, azimuth))

        distance = self._estimate_distance(amplitude_db)
        return azimuth, distance

    # ------------------------------------------------------------------
    def _itd_azimuth(self, left: np.ndarray, right: np.ndarray) -> float:
        """Cross-correlation ITD -> azimuth degrees."""
        n = min(len(left), len(right))
        if n < _MAX_LAG * 2:
            return 0.0

        l = left[:n] - left[:n].mean()
        r = right[:n] - right[:n].mean()

        # Cross-correlation using FFT
        L = np.fft.rfft(l, n=2 * n)
        R = np.fft.rfft(r, n=2 * n)
        xcorr = np.fft.irfft(L * np.conj(R))

        # Search in ±MAX_LAG range
        center = 0
        search = np.concatenate([xcorr[center: center + _MAX_LAG + 1],
                                  xcorr[-(  _MAX_LAG):]])
        # Build lag array: [0, 1, ..., MAX_LAG, -MAX_LAG, ..., -1]
        lags = np.concatenate([np.arange(0, _MAX_LAG + 1),
                                np.arange(-_MAX_LAG, 0)])
        peak_lag = int(lags[np.argmax(search)])   # samples; + = right first

        # Map lag to azimuth: max_lag samples -> ±90°
        az = float(peak_lag) / _MAX_LAG * 90.0
        return az

    def _ild_azimuth(self, left: np.ndarray, right: np.ndarray) -> float:
        """Level difference -> azimuth degrees."""
        l_rms = float(np.sqrt(np.mean(left ** 2)) + 1e-9)
        r_rms = float(np.sqrt(np.mean(right ** 2)) + 1e-9)
        ild_db = 20 * np.log10(r_rms / l_rms)
        # Empirical: ~6 dB ILD at 90° for Valorant-style panning
        az = np.clip(ild_db / 6.0, -1.0, 1.0) * 90.0
        return float(az)

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
