# valorant-minimap-coach

Screen-reading overlay that watches the Valorant minimap and gives voice callouts like a teammate.

**How it works:** reads pixels from your screen (like OBS does), never touches game memory. Vanguard watches for memory injection and kernel hooks, not screen capture APIs.

## Stack

- **mss** - fast screen capture (~30 fps)
- **OpenCV** - color-based enemy/teammate blob detection on the minimap
- **Claude Vision API** - periodic deeper analysis for rotation/flank advice
- **pyttsx3** - offline text-to-speech

Hybrid logic: CV fires instantly when new enemies appear, AI adds context every few seconds.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
```

## Calibrate minimap position

Run once to teach the tool where your minimap is on screen:

```bash
python calibrate.py
```

Alt-tab to Valorant, wait 3 s, then click the top-left and bottom-right corners of the minimap. Saves to `config.yaml` automatically.

## Run

```bash
python coach.py
```

Change the active map in `config.yaml` before each session:

```yaml
map: "ascent"  # ascent | bind | haven | split | icebox | lotus
```

## Config

| Key | Default | Description |
|---|---|---|
| `minimap.region` | auto after calibrate | screen coordinates of minimap |
| `detection.min_contour_area` | 5 | noise filter for blob detection |
| `ai.enabled` | true | toggle Claude Vision analysis |
| `ai.analyze_interval` | 3.0 | seconds between AI calls |
| `audio.cooldown` | 2.0 | seconds before same callout repeats |

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
