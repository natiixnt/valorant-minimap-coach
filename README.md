# valorant-minimap-coach

Screen-reading + audio-listening overlay that acts as a real-time coaching assistant during Valorant matches. Reads pixels from your screen and audio from your speakers -- never touches game memory. Vanguard watches for memory injection and kernel hooks, not screen capture or audio loopback APIs.

---

## Features

### Vision / minimap analysis

| Feature | What it does |
|---|---|
| **Enemy detection** | Tracks red dots on the minimap, normalizes to named map zones |
| **Team detection** | Tracks cyan teammate dots; estimates player position from average |
| **Player angle** | Reads the directional triangle near minimap center to determine which way you're facing |
| **Ability detection** | Announces active utility seen on minimap: Reyna Eye, Viper Wall, Sage Wall, Killjoy setup, Sova Bolt, Phoenix Fire, Omen/Brimstone Smoke, Skye Trailblazer, Cypher Camera, KAY/O Knife |
| **Spike detection** | Detects the planted spike icon (yellow-orange), confirmed over 4 frames to avoid false positives |
| **Enemy stack** | Warns when 3+ enemies cluster in the same zone for 3 consecutive frames |
| **Zone transitions** | Announces when an enemy moves from one named zone to another ("Enemy moving from B Long to B Site") |
| **Trajectory prediction** | Extrapolates enemy movement 1.5 s ahead using linear velocity fit; warns before they cross a zone boundary |
| **Play pattern detection** | Classifies enemy formations as RUSH / EXECUTE / SPLIT / LURK / MID_CTRL based on clustering and velocity |
| **Site clear** | Announces when enemies that were visible all disappear |
| **Map auto-detection** | Identifies map on startup via Claude Vision, then uses local HSV histogram matching -- one API call per map ever seen |

### Audio analysis

All audio processing runs locally at 48 kHz via system loopback (WASAPI on Windows, BlackHole on macOS, PulseAudio on Linux). No API calls.

| Feature | What it does |
|---|---|
| **Footstep detection** | Bandpass spectral flux onset detection (200-800 Hz thud band). Announces zone + distance |
| **3D direction estimation** | ITD (interaural time delay via cross-correlation) + ILD (level ratio) fused 60/40 to estimate azimuth. Zone name used as callout instead of "left/right" |
| **Distance estimation** | Amplitude-based rough estimate. Note: Riot uses flat attenuation by design, so amplitude is unreliable at long range |
| **Shoe-type classification** | RandomForest on 30-dim MFCC features classifies footsteps as heavy / medium / light shoe type. Per-agent audio does not exist in Valorant (confirmed by Riot) |
| **Surface detection** | Spectral centroid of footstep classifies surface: carpet / wood / concrete / metal |
| **Gunshot detection** | Wideband onset (500-8 kHz, >30x amplitude spike). Estimates direction and suppressor status (suppressed = peak <24x threshold) |
| **Noise gate** | Suppresses gunshot/explosion transients before footstep detection to eliminate false positives |
| **Round audio events** | Detects round-start horn and win/loss jingles without reading HUD |
| **Spike beep tracking** | Detects spike beep events; estimates remaining time via power-law IBI model (secondary to wall clock) |
| **Defuse start detection** | Detects the defuse initiation click; shows Valorant-style 7 s progress bar. Re-arms after each attempt |

### Game intelligence

| Feature | What it does |
|---|---|
| **Round state machine** | Tracks BUY_PHASE → ROUND_ACTIVE → POST_PLANT → ROUND_END; side flips at round 12 |
| **Heatmap** | Tracks which zones enemies appear in across rounds; score decays 0.55× per round boundary; announces 2 hottest zones at round end |
| **Economy tracker** | Estimates enemy credits using verified loss bonus (1900/2400/2900, caps at 3rd loss), win bonus (3000), kill credits (200), plant bonus (300). Recommends save/force/full buy |
| **Retake advisor** | When spike is planted, ranks teammates by pre-computed zone travel time per map. Speaks one prioritized rotation callout |
| **Defuse feasibility** | Checks remaining time vs 7 s defuse + travel time each tick. Accounts for the half-defuse mechanic: if enemy has reached 3.5 s, only 3.5 s more are needed. Speaks at 20 s, 10 s, and ~7.5 s milestones |
| **Defuse progress bar** | Valorant-style UI bar with 50% diamond marker when defuse sound detected. Green → amber → red. Key question answered: "are they past halfway?" |
| **Ultimate tracker** | Warns at round 4 (earliest realistic ult charge) then every 4 rounds. Ult points from kills, deaths, orbs (2 per map per half), plant, and defuse completion |

### AI analysis

| Feature | What it does |
|---|---|
| **Claude Vision analysis** | Sends minimap JPEG + game state to Claude Sonnet every 3 s (when enemies visible). Returns one tactical callout ≤ 10 words |
| **Scene-change dedup** | Skips API call if enemy count, spike state, and active abilities are identical to last call (saves cost when scene is static) |
| **Feedback loop** | Thumbs up/down buttons in overlay. Rating stored with the minimap sample for future training data |
| **Local model support** | Optional: route analysis to a local LLM instead of the API |

### Overlay UI

| Element | Detail |
|---|---|
| **Minimap canvas** | Live enemy dot positions with fade-out on loss-of-sight; sighting history rings; grid overlay |
| **Enemy panel** | Count + dot indicators colored by threat |
| **Callout panel** | Last spoken callout |
| **Defuse tracker** | Valorant-style progress bar, hidden when spike not planted |
| **AI insight panel** | Claude's tactical read with thumbs up/down feedback buttons |
| **Utility panel** | Active enemy abilities |
| **Settings** | Voice selector with TEST preview, theme presets (VALORANT/CYBER/MATRIX/PHANTOM), custom color picker for all UI elements, callout language selector |
| **F9** | Toggle overlay visibility |
| **Mute** | Toggle TTS without closing |

---

## API cost

Most features run entirely locally (audio processing, CV detection, game intelligence, TTS). The only API usage is Claude Vision for tactical minimap analysis.

### Per-session cost breakdown

**Minimap analysis** (`claude-sonnet-4-6`, fires every 3 s when enemies visible, deduplicated on static scenes)

| Token type | Per call | 150 calls / 30-min game |
|---|---|---|
| Input (minimap JPEG + prompt) | ~430 tokens | ~64 500 tokens |
| Output (callout ≤ 10 words) | ~12 tokens | ~1 800 tokens |

**Map detection** (`claude-haiku-4-5-20251001`)
- First time each map is seen: 1 call (~800 input, ~5 output tokens) → template saved locally
- All subsequent detections: local HSV histogram matching, ~1 ms, $0
- After seeing all 7 active maps once: API never called for map detection again

### Monthly estimate (40 games, 30 min each)

| Component | Model | Cost per game | 40 games/month |
|---|---|---|---|
| Minimap analysis | `claude-sonnet-4-6` ($3 / $15 per MTok) | ~$0.22 | **~$8.80** |
| Map detection (first sight) | `claude-haiku-4-5-20251001` ($0.80 / $4 per MTok) | ~$0.001 once | **~$0** after learning |
| Audio + CV + game logic | local | $0 | **$0** |
| **Total (steady state)** | | **~$0.22** | **~$8.80** |

Current model prices: [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing)

### Reducing cost

| Method | Effect |
|---|---|
| Increase `ai.analyze_interval` from 3 s to 6 s | Halves call count → ~$4.40/month |
| Switch `ai.model` to `claude-haiku-4-5-20251001` | ~10x cheaper → ~$1/month |
| Set `ai.enabled: false` | $0 -- all CV/audio/game-logic callouts still work |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# paste your ANTHROPIC_API_KEY into .env
```

### Audio loopback (required for audio features)

| OS | Driver needed |
|---|---|
| Windows | WASAPI loopback -- built-in, no setup required |
| macOS | [BlackHole](https://github.com/ExistentialAudio/BlackHole) or Loopback -- free/paid virtual audio device |
| Linux | PulseAudio monitor source -- usually available by default |

Set Valorant's audio output to the loopback device, or use a virtual cable to split your speakers and the loopback.

### Calibrate minimap position (once per resolution)

```bash
python calibrate.py
```

Alt-tab to Valorant, wait 3 s, click the top-left and bottom-right corners of the minimap. Saves to `config.yaml`.

### Run

```bash
python coach_app.py
```

---

## Config

| Key | Default | Description |
|---|---|---|
| `minimap.region` | set by calibrate | screen pixel coordinates of minimap |
| `detection.min_contour_area` | 5 | noise filter for blob detection |
| `ai.enabled` | true | toggle Claude Vision analysis |
| `ai.model` | `claude-sonnet-4-6` | model for minimap analysis |
| `ai.analyze_interval` | 3.0 | seconds between AI calls |
| `audio_coach.enabled` | true | toggle all audio analysis |
| `audio_coach.device` | null | loopback device name; null = auto-detect |
| `map_override` | null | force a map name, skip auto-detect |
| `map_detection.recheck_interval` | 300 | seconds between map re-checks |
| `data_collection.enabled` | false | opt-in telemetry upload |
| `data_collection.endpoint` | `""` | your collection server URL |

---

## Windows build (no Python)

Download `ValorantCoach.exe` from [Releases](https://github.com/natiixnt/valorant-minimap-coach/releases).

Create `.env` next to the exe:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Run `ValorantCoach.exe`. On first launch it writes `config.yaml` next to itself.

---

## Data collection (optional, opt-in)

Enable in `config.yaml`:

```yaml
data_collection:
  enabled: true
  endpoint: "http://your-server:8000"
  api_key: "changeme"
```

**What gets uploaded:**
- Minimap frame (200×200 JPEG) + Claude callout as label -- on every AI analysis call
- Footstep audio clip (48 kHz mono .npy) + zone + shoe type -- on confident footstep detections

Full-screen captures are not uploaded. No usernames or chat text.

### Running the collection server

```bash
pip install -r server/requirements.txt
API_KEY=changeme DATA_DIR=data/collected uvicorn server.collect_server:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /stats`, `GET /review` (label UI), `GET /download/<type>`.

---

## Tuning detection

If enemies aren't detected or you're getting false positives:

```python
from src.capture.screen import ScreenCapture
import yaml, cv2

config = yaml.safe_load(open("config.yaml"))
frame = ScreenCapture(config).capture()
cv2.imwrite("minimap_debug.png", frame.data)
```

Open `minimap_debug.png`, use a color picker on enemy dots, update `detection.enemy_color_lower/upper` in `config.yaml`.

### Training the shoe-type classifier (optional)

```bash
python tools/collect_footsteps.py   # record labeled samples
python tools/train_classifier.py    # trains RandomForest, saves data/footstep_model.pkl
```

Without a trained model, shoe-type classification is disabled; zone + distance callouts still work.
