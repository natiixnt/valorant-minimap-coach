#!/usr/bin/env python3
"""
Training data collection tool for the footstep shoe-type classifier.

IMPORTANT: Valorant agents do NOT have unique per-agent footstep sounds (confirmed by
Riot AMA). They use shoe type categories. Collect samples by shoe type:

    python tools/collect_footsteps.py --agent heavy   --output data/footsteps/
    python tools/collect_footsteps.py --agent medium  --output data/footsteps/
    python tools/collect_footsteps.py --agent light   --output data/footsteps/

Shoe type categories:
  heavy  - Brimstone, Breach, Sage, Killjoy, Cypher, Deadlock, Omen, Viper, Astra, Sova
  medium - Skye, Phoenix, Fade, Gekko, Clove, Harbor, KAYO
  light  - Jett, Neon, Yoru, Reyna, ISO, Chamber

You can also use per-agent names (e.g. --agent jett) and they will be auto-mapped
to shoe type during training.

While recording:
  - Have Valorant running or use a gameplay recording
  - Press SPACE when you hear a footstep to capture it
  - Press Q to finish the session

Each captured clip is saved as:
    data/footsteps/<shoe_type_or_agent>/<N>.npy   (float32 mono, 48000 Hz, 0.35 s)

Collect at least 20 samples per category, ideally 50+.
After collecting, train the model:
    python tools/train_classifier.py --samples data/footsteps/ --output data/footstep_model.pkl
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_RATE = 48000   # Valorant outputs at 48 kHz; match agent_classifier.py
CLIP_SECONDS = 0.35
PRE_TRIGGER_SEC = 0.08   # capture this many seconds before key press
CLIP_SAMPLES = int(CLIP_SECONDS * SAMPLE_RATE)
PRE_SAMPLES = int(PRE_TRIGGER_SEC * SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect footstep audio samples")
    parser.add_argument("--agent", required=True, help="Agent name (e.g. jett)")
    parser.add_argument("--output", default="data/footsteps", help="Output directory")
    parser.add_argument("--device", default=None, help="Audio device name (default: auto loopback)")
    args = parser.parse_args()

    agent = args.agent.lower()
    out_dir = Path(args.output) / agent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing samples
    existing = len(list(out_dir.glob("*.npy")))
    print(f"[Collect] Agent: {agent}  Output: {out_dir}  Existing: {existing} samples")
    print("[Collect] Starting audio capture... (SPACE=capture, Q=quit)")

    from src.audio.capture import AudioCapture
    cap = AudioCapture(device_name=args.device)
    cap.start()
    time.sleep(0.5)   # let ring buffer fill

    count = existing

    # Use pynput for keyboard if available, fallback to input()
    try:
        from pynput import keyboard as kb

        captured = []
        done = [False]

        def on_press(key):
            try:
                if key.char == 'q':
                    done[0] = True
                    return False
            except AttributeError:
                if key == kb.Key.space:
                    captured.append(time.time())

        with kb.Listener(on_press=on_press) as listener:
            while not done[0]:
                if captured:
                    ts = captured.pop(0)
                    _save_clip(cap, out_dir, count, agent)
                    count += 1
                time.sleep(0.01)

    except ImportError:
        print("[Collect] pynput not installed. Using basic input mode.")
        print("          Type ENTER to capture, q+ENTER to quit.")
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().lower() == 'q':
                break
            _save_clip(cap, out_dir, count, agent)
            count += 1

    cap.stop()
    total = len(list(out_dir.glob("*.npy")))
    print(f"[Collect] Done. {total} total samples saved for {agent}.")
    if total >= 20:
        print(f"[Collect] Ready to train. Run: python tools/train_classifier.py")
    else:
        print(f"[Collect] Collect at least 20 samples per agent for reliable results.")


def _save_clip(cap, out_dir: Path, count: int, agent: str) -> None:
    stereo = cap.read(n_samples=CLIP_SAMPLES + PRE_SAMPLES)
    if stereo is None:
        print(f"  [!] No audio data yet")
        return
    # Take pre-trigger window
    mono = ((stereo[0] + stereo[1]) * 0.5)[-CLIP_SAMPLES:]
    path = out_dir / f"{count:04d}.npy"
    np.save(str(path), mono.astype(np.float32))
    rms_db = 20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9)
    print(f"  Saved {path.name}  RMS: {rms_db:.1f} dB")


if __name__ == "__main__":
    main()
