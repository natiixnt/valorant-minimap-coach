"""
Real-time stereo audio capture via system loopback (what-you-hear).

Uses the `soundcard` library which supports:
  - Windows: WASAPI loopback
  - macOS: BlackHole / Loopback virtual device
  - Linux: PulseAudio monitor source

The ring buffer holds ~2 s of stereo audio at 44.1 kHz and is read from
the analysis pipeline in a non-blocking fashion.

Usage:
    cap = AudioCapture()
    cap.start()
    ...
    chunk = cap.read(n_samples=2048)   # returns (2, N) float32 or None
    cap.stop()
"""
import threading
from collections import deque
from typing import Optional

import numpy as np

SAMPLE_RATE = 44100
CHANNELS = 2
CHUNK_FRAMES = 1024          # frames per loopback read
RING_SECONDS = 2.0           # ring buffer length
_RING_CHUNKS = int(RING_SECONDS * SAMPLE_RATE / CHUNK_FRAMES) + 1


class AudioCapture:
    def __init__(self, device_name: Optional[str] = None) -> None:
        """
        device_name: partial name of loopback/monitor device, or None for default.
        On Windows leave as None to auto-select WASAPI loopback output.
        On macOS set to 'BlackHole' or 'Loopback'.
        """
        self._device_name = device_name
        self._ring: deque = deque(maxlen=_RING_CHUNKS)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._mic: object = None   # soundcard loopback object

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="AudioCapture"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------
    def read(self, n_samples: int = 2048) -> Optional[np.ndarray]:
        """
        Return the most recent n_samples as float32 array shaped (2, n_samples).
        Channels: [0] = left, [1] = right.
        Returns None if not enough data yet.
        """
        with self._lock:
            chunks = list(self._ring)
        if not chunks:
            return None
        flat = np.concatenate(chunks, axis=0)   # (total_frames, 2)
        if flat.shape[0] < n_samples:
            return None
        recent = flat[-n_samples:]               # (n_samples, 2)
        return recent.T.astype(np.float32)       # (2, n_samples)

    def read_mono(self, n_samples: int = 2048) -> Optional[np.ndarray]:
        """Convenience: sum channels to mono, shape (n_samples,)."""
        stereo = self.read(n_samples)
        if stereo is None:
            return None
        return (stereo[0] + stereo[1]) * 0.5

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        try:
            import soundcard as sc
        except ImportError:
            print("[AudioCapture] 'soundcard' not installed. Run: pip install soundcard")
            return

        mic = self._get_loopback(sc)
        if mic is None:
            print("[AudioCapture] No loopback device found. Audio analysis disabled.")
            return

        print(f"[AudioCapture] Recording from: {mic.name}")
        with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as rec:
            while self._running:
                try:
                    data = rec.record(numframes=CHUNK_FRAMES)   # (frames, channels)
                    if data.shape[1] < CHANNELS:
                        # Mono device — duplicate channel
                        data = np.repeat(data, CHANNELS, axis=1)
                    with self._lock:
                        self._ring.append(data[:, :CHANNELS].astype(np.float32))
                except Exception as e:
                    print(f"[AudioCapture] Record error: {e}")
                    break

    def _get_loopback(self, sc) -> object:
        """Return best matching loopback/monitor microphone."""
        try:
            mics = sc.all_microphones(include_loopback=True)
        except Exception:
            mics = sc.all_microphones()

        if self._device_name:
            name_lower = self._device_name.lower()
            for m in mics:
                if name_lower in m.name.lower():
                    return m

        # Auto-select: prefer loopback/monitor sources
        keywords = ["loopback", "monitor", "stereo mix", "what u hear",
                    "blackhole", "virtual", "output"]
        for m in mics:
            ml = m.name.lower()
            if any(k in ml for k in keywords):
                return m

        # Fallback: default microphone (will capture mic audio instead of game)
        try:
            return sc.default_microphone()
        except Exception:
            return mics[0] if mics else None
