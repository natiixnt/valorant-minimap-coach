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

# High-frequency band for ILD -- below ~1 kHz, sound diffracts around the head and
# ILD approaches zero for all azimuths; only above 1 kHz does the head cast an
# "acoustic shadow" that produces a meaningful level difference between the ears
_ILD_LOW_HZ  = 1000
_ILD_HIGH_HZ = 8000

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
        # Bandpass for high-frequency ILD (computed once, applied stateless per call)
        self._ild_sos  = butter(4, [_ILD_LOW_HZ, _ILD_HIGH_HZ],
                                btype="band", fs=SAMPLE_RATE, output="sos")
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

        az_itd = self._itd_azimuth(left, right)
        az_ild = self._ild_azimuth(left, right)

        # Weighted combination
        azimuth = _ITD_WEIGHT * az_itd + _ILD_WEIGHT * az_ild
        azimuth = max(-180.0, min(180.0, azimuth))

        # Disambiguate front vs back using HRTF spectral notch cues
        azimuth = self._front_back(left, right, azimuth)

        distance = self._estimate_distance(amplitude_db)
        return azimuth, distance

    # ------------------------------------------------------------------
    def _itd_azimuth(self, left: np.ndarray, right: np.ndarray) -> float:
        """GCC-PHAT interaural time delay -> azimuth degrees.

        GCC-PHAT (Generalized Cross-Correlation with Phase Transform) divides the
        cross-spectrum by its magnitude before computing the IFFT. This whitens the
        spectrum so no single frequency dominates the correlation peak -- critical for
        game audio where the loud 200 Hz footstep thud would otherwise drown out the
        timing information in the 1-4 kHz transient band.
        Result: a much sharper, more reliable peak even in noisy conditions.
        """
        n = min(len(left), len(right))
        if n < _MAX_LAG * 2:
            return 0.0

        l = left[:n] - left[:n].mean()
        r = right[:n] - right[:n].mean()

        L = np.fft.rfft(l, n=2 * n)
        R = np.fft.rfft(r, n=2 * n)
        cross = L * np.conj(R)
        # Phase transform: normalize cross-spectrum to unit magnitude per frequency bin
        # GCC-PHAT normalization: dividing each bin by its own magnitude (+ epsilon to
        # avoid /0) forces every frequency to contribute equally to the correlation peak.
        # Without this, the loud low-frequency thud (200-800 Hz) would dominate and
        # smear the peak, making the lag estimate unreliable.
        cross_phat = cross / (np.abs(cross) + 1e-10)
        xcorr = np.fft.irfft(cross_phat)

        search = np.concatenate([xcorr[: _MAX_LAG + 1], xcorr[-_MAX_LAG:]])
        lags   = np.concatenate([np.arange(0, _MAX_LAG + 1), np.arange(-_MAX_LAG, 0)])
        peak_lag = int(lags[np.argmax(search)])   # positive = right ear first

        az = float(peak_lag) / _MAX_LAG * 90.0
        return az

    def _ild_azimuth(self, left: np.ndarray, right: np.ndarray) -> float:
        """High-band ILD (1-8 kHz) -> azimuth degrees.

        The full-spectrum ILD is dominated by the low-frequency footstep thud (200-800 Hz)
        where the head is too small relative to the wavelength to cast an acoustic shadow --
        ILD at these frequencies is near zero regardless of direction. Restricting to
        1-8 kHz captures the frequencies where the head actually creates a meaningful
        level difference between the ears.
        """
        # sosfilt without zi (initial conditions) is intentional: each call is stateless.
        # We process one short onset window per event; there is no continuous signal to
        # preserve state across - starting from zero is correct and avoids stale filter
        # memory from a previous (possibly very different) event leaking into this one.
        l_bp = sosfilt(self._ild_sos, left)
        r_bp = sosfilt(self._ild_sos, right)
        l_rms = float(np.sqrt(np.mean(l_bp ** 2)) + 1e-9)
        r_rms = float(np.sqrt(np.mean(r_bp ** 2)) + 1e-9)
        ild_db = 20 * np.log10(r_rms / l_rms)
        az = np.clip(ild_db / 6.0, -1.0, 1.0) * 90.0
        return float(az)

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
