"""
Agent footstep classifier using MFCCs + spectral features.

Training:
  Collect short (~0.3 s) audio clips of each agent's footstep using
  tools/collect_footsteps.py.  Then call:
      classifier = AgentClassifier()
      classifier.train(samples_dir="data/footsteps/")
      classifier.save("data/footstep_model.pkl")

Inference (real-time):
      classifier = AgentClassifier()
      classifier.load("data/footstep_model.pkl")
      agent, confidence = classifier.predict(audio_chunk)

Feature vector (per clip, 21 features):
  - 13 MFCCs (mean over time)
  - 13 MFCC deltas (mean)
  - Spectral centroid (mean)
  - Spectral rolloff 85% (mean)
  - Zero-crossing rate (mean)
  - RMS energy

Known Valorant footstep characteristics (community + empirical):
  Heavy agents (Sage, KJ, Reyna, Breach, Sova, Cypher, Deadlock):
    - Deeper thud, more low-frequency content
    - Spectral centroid ~300-600 Hz
  Light agents (Jett, Neon):
    - Lighter tap, less bass
    - Spectral centroid ~500-900 Hz
  Neon (sprint): fast cadence, distinct cadence rhythm
  Jett (glide): almost silent, very low amplitude

Agents are grouped into broad acoustic classes to avoid over-fitting
on limited training data. A "role" output (heavy/medium/light) is
also provided alongside the per-agent prediction.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 44100
CLIP_SECONDS = 0.35         # clip length for feature extraction
N_MFCC = 13

# Agent groups for hierarchical classification
AGENT_ROLES: Dict[str, str] = {
    "jett": "light", "neon": "light", "yoru": "light", "iso": "light",
    "reyna": "medium", "phoenix": "medium", "clove": "medium",
    "sage": "heavy", "killjoy": "heavy", "cypher": "heavy", "deadlock": "heavy",
    "breach": "heavy", "sova": "heavy", "skye": "heavy", "kayo": "heavy",
    "omen": "heavy", "brimstone": "heavy", "viper": "heavy", "harbor": "heavy",
    "astra": "heavy", "fade": "medium", "gekko": "medium", "chamber": "medium",
    "unknown": "unknown",
}

ALL_AGENTS = sorted([a for a in AGENT_ROLES if a != "unknown"])


class AgentClassifier:
    def __init__(self) -> None:
        self._model = None          # sklearn RandomForestClassifier
        self._label_map: List[str] = []
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, samples_dir: str) -> None:
        """
        Train from a directory tree:
          samples_dir/
            jett/
              001.npy   (float32 mono array at 44100 Hz)
              002.npy
            sage/
              001.npy
            ...

        Accepts .npy (numpy) or .wav files.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            print("[AgentClassifier] pip install scikit-learn")
            return

        X, y_raw = [], []
        base = Path(samples_dir)
        for agent_dir in sorted(base.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent_name = agent_dir.name.lower()
            clips = list(agent_dir.glob("*.npy")) + list(agent_dir.glob("*.wav"))
            for clip_path in clips:
                audio = self._load_clip(clip_path)
                if audio is None:
                    continue
                features = extract_features(audio)
                X.append(features)
                y_raw.append(agent_name)

        if not X:
            print("[AgentClassifier] No training samples found.")
            return

        X_arr = np.array(X)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        self._label_map = list(le.classes_)

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_arr, y)
        self._model = clf
        self._trained = True
        print(f"[AgentClassifier] Trained on {len(X)} samples, {len(self._label_map)} agents.")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "labels": self._label_map}, f)
        print(f"[AgentClassifier] Saved to {path}")

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._label_map = data["labels"]
        self._trained = True
        print(f"[AgentClassifier] Loaded from {path}")
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, audio: np.ndarray) -> Tuple[str, float, str]:
        """
        audio: float32 mono array at 44100 Hz (at least CLIP_SECONDS long).

        Returns (agent_name, confidence, role).
        Falls back to ("unknown", 0.0, "unknown") if not trained.
        """
        if not self._trained or self._model is None:
            return "unknown", 0.0, "unknown"

        clip_len = int(CLIP_SECONDS * SAMPLE_RATE)
        if len(audio) > clip_len:
            # Take the loudest window
            audio = _loudest_window(audio, clip_len)
        elif len(audio) < clip_len:
            audio = np.pad(audio, (0, clip_len - len(audio)))

        features = extract_features(audio).reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        best_idx = int(np.argmax(proba))
        agent = self._label_map[best_idx] if best_idx < len(self._label_map) else "unknown"
        confidence = float(proba[best_idx])
        role = AGENT_ROLES.get(agent, "unknown")
        return agent, confidence, role

    # ------------------------------------------------------------------
    def _load_clip(self, path: Path) -> Optional[np.ndarray]:
        if path.suffix == ".npy":
            try:
                return np.load(str(path)).astype(np.float32)
            except Exception:
                return None
        if path.suffix == ".wav":
            try:
                import wave, struct
                with wave.open(str(path)) as wf:
                    n = wf.getnframes()
                    raw = wf.readframes(n)
                    ch = wf.getnchannels()
                    sw = wf.getsampwidth()
                    if sw == 2:
                        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sw == 4:
                        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        return None
                    if ch > 1:
                        data = data.reshape(-1, ch).mean(axis=1)
                    return data
            except Exception:
                return None
        return None


# ------------------------------------------------------------------
# Feature extraction (used by both train and predict)
# ------------------------------------------------------------------

def extract_features(audio: np.ndarray) -> np.ndarray:
    """Return 1D feature vector of length 43."""
    sr = SAMPLE_RATE
    clip_len = int(CLIP_SECONDS * sr)
    if len(audio) < clip_len:
        audio = np.pad(audio, (0, clip_len - len(audio)))
    audio = audio[:clip_len]

    # -- MFCCs
    mfcc = _mfcc(audio, sr, N_MFCC)          # (N_MFCC, T)
    mfcc_mean = mfcc.mean(axis=1)             # (N_MFCC,)
    # Delta (first order difference across time)
    if mfcc.shape[1] > 2:
        delta = np.diff(mfcc, axis=1)
        delta_mean = delta.mean(axis=1)
    else:
        delta_mean = np.zeros(N_MFCC)

    # -- Spectral features
    stft = np.abs(_stft(audio, n_fft=512, hop=256))  # (freq_bins, T)
    freqs = np.fft.rfftfreq(512, d=1.0 / sr)
    mag_sum = stft.sum(axis=0) + 1e-9           # (T,)

    # Spectral centroid
    centroid = (freqs[:, None] * stft).sum(axis=0) / mag_sum   # (T,)
    centroid_mean = float(centroid.mean())

    # Spectral rolloff 85%
    cumsum = np.cumsum(stft, axis=0)
    total = cumsum[-1, :] + 1e-9
    rolloff_idx = np.argmax(cumsum >= 0.85 * total, axis=0)
    rolloff_hz = freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)].mean()

    # Zero crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) * 0.5)

    # RMS
    rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)

    features = np.concatenate([
        mfcc_mean,           # 13
        delta_mean,          # 13
        [centroid_mean],     # 1
        [rolloff_hz],        # 1
        [zcr],               # 1
        [rms],               # 1
    ])
    return features.astype(np.float32)   # 30 features total


# ------------------------------------------------------------------
# Minimal DSP helpers (avoid librosa dependency for inference)
# ------------------------------------------------------------------

def _stft(audio: np.ndarray, n_fft: int = 512, hop: int = 256) -> np.ndarray:
    window = np.hanning(n_fft)
    frames = []
    for start in range(0, len(audio) - n_fft + 1, hop):
        frame = audio[start: start + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)))
    if not frames:
        return np.zeros((n_fft // 2 + 1, 1))
    return np.array(frames).T   # (freq_bins, T)


def _mfcc(audio: np.ndarray, sr: int, n_mfcc: int) -> np.ndarray:
    """Compute MFCCs without librosa."""
    n_fft = 512
    hop = 256
    n_mels = 40

    stft_mag = _stft(audio, n_fft, hop)    # (freq_bins, T)
    mel_fb = _mel_filterbank(sr, n_fft, n_mels)   # (n_mels, freq_bins)
    mel_spec = np.dot(mel_fb, stft_mag)    # (n_mels, T)
    log_mel = np.log(mel_spec + 1e-9)

    # DCT-II to get MFCCs
    T = log_mel.shape[1]
    mfcc = np.zeros((n_mfcc, T))
    for n in range(n_mfcc):
        mfcc[n] = np.sum(
            log_mel * np.cos(np.pi * n / n_mels * (np.arange(n_mels)[:, None] + 0.5)),
            axis=0,
        )
    return mfcc


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Simple mel filterbank."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    low_mel = hz_to_mel(80)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / (c - l + 1e-9)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / (r - c + 1e-9)
    return fb


def _loudest_window(audio: np.ndarray, window: int) -> np.ndarray:
    """Return the window of length `window` with highest RMS."""
    best_rms = -1.0
    best_start = 0
    step = window // 4
    for start in range(0, len(audio) - window, step):
        rms = float(np.sqrt(np.mean(audio[start: start + window] ** 2)))
        if rms > best_rms:
            best_rms = rms
            best_start = start
    return audio[best_start: best_start + window]
