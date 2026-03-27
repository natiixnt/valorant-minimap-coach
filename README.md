# valorant-minimap-coach

Screen-reading overlay that watches the Valorant minimap and gives voice callouts like a teammate.

**How it works:** reads pixels from your screen (like OBS does), never touches game memory. Vanguard watches for memory injection and kernel hooks, not screen capture APIs.

## Stack

- **mss** - fast screen capture (~30 fps)
- **OpenCV** - color-based enemy/teammate blob detection on the minimap
- **Claude Vision API** - periodic deeper analysis for rotation/flank advice
- **pyttsx3** - offline text-to-speech

Hybrid logic: CV fires instantly when new enemies appear, AI adds context every few seconds. Map is detected automatically from your screen on startup.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
```

## Calibrate minimap position (once)

Run once to teach the tool where your minimap is on screen:

```bash
python calibrate.py
```

Alt-tab to Valorant, wait 3 s, then click the top-left and bottom-right corners of the minimap. Saves to `config.yaml` automatically.

## Run

```bash
python coach.py
```

That's it. On startup the coach detects which map is loaded from your screen and announces it. If you're in the lobby it retries every 10 s until the game starts. It re-checks the map every 5 minutes so it picks up new games without a restart.

If auto-detection misbehaves for a specific map, set `map_override` in `config.yaml`:

```yaml
map_override: "ascent"  # null = auto-detect
```

## Config

| Key | Default | Description |
|---|---|---|
| `minimap.region` | auto after calibrate | screen coordinates of minimap |
| `detection.min_contour_area` | 5 | noise filter for blob detection |
| `ai.enabled` | true | toggle Claude Vision analysis |
| `ai.analyze_interval` | 3.0 | seconds between AI calls |
| `audio.cooldown` | 2.0 | seconds before same callout repeats |
| `map_override` | null | force a map name, skips auto-detect |
| `map_detection.recheck_interval` | 300 | seconds between map re-checks |
| `map_detection.startup_retry_interval` | 10 | seconds between retries before game starts |

## Tuning detection

If enemies aren't being detected (or getting false positives), run:

```python
import cv2
import numpy as np
from src.capture.screen import ScreenCapture
import yaml

config = yaml.safe_load(open("config.yaml"))
cap = ScreenCapture(config)
frame = cap.capture()
cv2.imwrite("minimap_debug.png", frame.data)
```

Open `minimap_debug.png` and use a color picker to check the HSV values of enemy dots. Update `detection.enemy_color_lower/upper` in `config.yaml` accordingly.
