"""
Training data collection server.

Receives labeled screenshots and audio clips from client instances and stores
them in an organized directory structure ready for model training.

Directory layout:
    data/collected/
      minimap_callout/
        <sha1_of_callout>/
          meta.json          # { label, map, enemies, spike_active, recent_callouts,
                             #   feedback: {positive: N, negative: N} }
          1700000000_v1.0_conf1.00.jpg
      footstep_audio/
        heavy/
          1700000001_v1.0_conf0.00.npy
      map_detection/
        ascent/
          1700000002_v1.0_conf0.61.jpg

Run:
    pip install -r server/requirements.txt
    API_KEY=changeme DATA_DIR=data/collected uvicorn server.collect_server:app --host 0.0.0.0 --port 8000
"""

import hashlib
import json
import os
import time
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR      = Path(os.getenv("DATA_DIR", "data/collected"))
API_KEY       = os.getenv("API_KEY", "")
MAX_MB        = float(os.getenv("MAX_IMAGE_MB", "4"))
MAX_LABEL_LEN = 300

app = FastAPI(title="Valorant Coach Data Collector", version="2.0")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_key(key: str) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _safe(s: str, max_len: int = 60) -> str:
    return "".join(c for c in s if c.isalnum() or c in "_-")[:max_len]


def _label_dir(function: str, label: str) -> Path:
    """
    For free-text labels (minimap_callout): hash the text to get a stable dir name.
    For categorical labels (map name, shoe type): use the label directly.
    """
    if function in ("minimap_callout",):
        key = hashlib.sha1(label.encode()).hexdigest()[:12]
    else:
        key = _safe(label)
    return DATA_DIR / _safe(function) / key


def _update_meta(directory: Path, update: dict) -> None:
    path = directory / "meta.json"
    meta: dict = {}
    if path.exists():
        try:
            meta = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            meta = {}
    meta.update(update)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/collect")
async def collect(
    file:            UploadFile = File(...),
    label:           str        = Form(...),
    function:        str        = Form(...),
    map:             str        = Form(default=""),
    enemies:         str        = Form(default=""),
    spike_active:    str        = Form(default="0"),
    recent_callouts: str        = Form(default=""),
    conf:            float      = Form(default=1.0),
    app_version:     str        = Form(default="unknown"),
    ts:              int        = Form(default_factory=lambda: int(time.time())),
    x_api_key:       str        = Header(default=""),
) -> JSONResponse:
    _check_key(x_api_key)

    if len(label) > MAX_LABEL_LEN:
        raise HTTPException(status_code=422, detail="label too long")

    data = await file.read()
    if len(data) > MAX_MB * 1_000_000:
        raise HTTPException(status_code=413, detail="file too large")

    ext = Path(file.filename or "sample.jpg").suffix.lstrip(".")
    if ext not in ("jpg", "jpeg", "npy"):
        raise HTTPException(status_code=415, detail="unsupported file type")

    out_dir = _label_dir(function, label)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{ts}_v{_safe(app_version)}_conf{conf:.2f}.{ext}"
    (out_dir / filename).write_bytes(data)

    _update_meta(out_dir, {
        "label":           label,
        "map":             map,
        "enemies_example": enemies[:200] if enemies else "",
        "spike_active":    spike_active == "1",
        "recent_callouts": recent_callouts.split("|", 9) if recent_callouts else [],
    })

    return JSONResponse({"status": "ok", "saved": filename})


@app.post("/feedback")
async def feedback(
    ts:        int  = Body(...),
    positive:  bool = Body(...),
    x_api_key: str  = Header(default=""),
) -> JSONResponse:
    _check_key(x_api_key)

    # Search all label dirs for a file matching this timestamp
    for fn_dir in DATA_DIR.iterdir():
        if not fn_dir.is_dir():
            continue
        for label_dir in fn_dir.iterdir():
            if not label_dir.is_dir():
                continue
            matches = list(label_dir.glob(f"{ts}_*.jpg")) + list(label_dir.glob(f"{ts}_*.npy"))
            if not matches:
                continue
            meta = json.loads((label_dir / "meta.json").read_text()) if (label_dir / "meta.json").exists() else {}
            fb   = meta.get("feedback", {"positive": 0, "negative": 0})
            if positive:
                fb["positive"] = fb.get("positive", 0) + 1
            else:
                fb["negative"] = fb.get("negative", 0) + 1
            meta["feedback"] = fb
            _update_meta(label_dir, meta)
            return JSONResponse({"status": "ok"})

    return JSONResponse({"status": "not_found"}, status_code=404)


@app.get("/stats")
async def stats(x_api_key: str = Header(default="")) -> JSONResponse:
    _check_key(x_api_key)

    result: dict = {}
    if not DATA_DIR.exists():
        return JSONResponse(result)

    for fn_dir in sorted(DATA_DIR.iterdir()):
        if not fn_dir.is_dir():
            continue
        fn_stats: dict = {"total": 0, "labels": {}}
        for label_dir in sorted(fn_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            count = len([f for f in label_dir.iterdir() if f.suffix in (".jpg", ".npy")])
            fn_stats["total"] += count
            meta = {}
            if (label_dir / "meta.json").exists():
                meta = json.loads((label_dir / "meta.json").read_text())
            display  = meta.get("label", label_dir.name)[:60]
            feedback = meta.get("feedback", {})
            fn_stats["labels"][display] = {
                "count":    count,
                "positive": feedback.get("positive", 0),
                "negative": feedback.get("negative", 0),
            }
        result[fn_dir.name] = fn_stats

    return JSONResponse(result)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
