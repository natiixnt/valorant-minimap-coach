#!/usr/bin/env python3
"""
Training data collection server.

Receives samples from the coach app and stores them for dataset curation.
Also provides a web review UI for validating footstep audio clips.

Usage:
    pip install fastapi uvicorn python-multipart
    python tools/server.py --port 8000 --key mysecretkey --data ./server_data/

Endpoints:
    POST /collect          -- receive a sample (image or audio clip)
    GET  /stats            -- collection statistics JSON
    GET  /review           -- browser review UI (footstep audio clips)
    POST /label/<ts>       -- submit corrected label for a sample
    GET  /download/<type>  -- download all samples of a type as a zip

Client config (config.yaml):
    data_collection:
      enabled: true
      endpoint: "http://localhost:8000"
      api_key:  "mysecretkey"
"""
import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

API_KEY: str = ""
DATA_DIR: Path = Path("server_data")


def create_app():
    try:
        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError:
        print("pip install fastapi uvicorn python-multipart")
        raise

    app = FastAPI(title="Valorant Coach Data Server")

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------
    def _check_auth(api_key: Optional[str]) -> None:
        if API_KEY and api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

    # ------------------------------------------------------------------
    # POST /collect
    # ------------------------------------------------------------------
    @app.post("/collect")
    async def collect(
        file:        UploadFile = File(...),
        type:        str        = Form(...),
        label:       str        = Form(""),
        map:         str        = Form(""),
        conf:        float      = Form(0.0),
        ts:          int        = Form(0),
        zone:        str        = Form(""),
        surface:     str        = Form(""),
        enemies:     str        = Form(""),
        app_version: str        = Form(""),
        x_api_key:   Optional[str] = Header(None, alias="X-API-Key"),
    ):
        _check_auth(x_api_key)

        sample_ts = ts or int(time.time())
        sample_dir = DATA_DIR / type / (map or "unknown")
        sample_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(file.filename or "sample").suffix or ".jpg"
        fname = f"{sample_ts}_{label[:40].replace('/', '_')}{ext}"
        fpath = sample_dir / fname
        contents = await file.read()
        fpath.write_bytes(contents)

        # Store metadata alongside
        meta = {
            "ts": sample_ts, "type": type, "label": label,
            "map": map, "conf": conf, "zone": zone,
            "surface": surface, "enemies": enemies,
            "app_version": app_version, "file": fname,
        }
        (sample_dir / f"{sample_ts}.json").write_text(json.dumps(meta))

        return {"ok": True, "ts": sample_ts}

    # ------------------------------------------------------------------
    # POST /feedback
    # ------------------------------------------------------------------
    @app.post("/feedback")
    async def feedback(
        ts:        int  = Form(...),
        positive:  str  = Form("1"),
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ):
        _check_auth(x_api_key)
        fb_path = DATA_DIR / "feedback" / f"{ts}.json"
        fb_path.parent.mkdir(parents=True, exist_ok=True)
        fb_path.write_text(json.dumps({"ts": ts, "positive": positive == "1"}))
        return {"ok": True}

    # ------------------------------------------------------------------
    # GET /stats
    # ------------------------------------------------------------------
    @app.get("/stats")
    async def stats(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        _check_auth(x_api_key)
        result = {}
        for sample_type in ("minimap_callout", "footstep_audio", "map_detection"):
            type_dir = DATA_DIR / sample_type
            if not type_dir.exists():
                result[sample_type] = 0
                continue
            count = sum(
                1 for f in type_dir.rglob("*")
                if f.suffix in (".jpg", ".npy")
            )
            result[sample_type] = count

        feedback_dir = DATA_DIR / "feedback"
        if feedback_dir.exists():
            feedbacks = list(feedback_dir.glob("*.json"))
            positives = sum(
                1 for f in feedbacks
                if json.loads(f.read_text()).get("positive")
            )
            result["feedback_total"] = len(feedbacks)
            result["feedback_positive"] = positives
        return result

    # ------------------------------------------------------------------
    # GET /review -- footstep audio review UI
    # ------------------------------------------------------------------
    @app.get("/review", response_class=HTMLResponse)
    async def review(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
        _check_auth(x_api_key)

        # Find all footstep audio samples
        audio_dir = DATA_DIR / "footstep_audio"
        samples = []
        if audio_dir.exists():
            for meta_file in sorted(audio_dir.rglob("*.json"))[:100]:
                try:
                    meta = json.loads(meta_file.read_text())
                    samples.append(meta)
                except Exception:
                    pass

        rows = ""
        for s in samples:
            rows += (
                f"<tr>"
                f"<td>{s.get('ts','')}</td>"
                f"<td>{s.get('map','')}</td>"
                f"<td>{s.get('zone','')}</td>"
                f"<td>{s.get('label','')}</td>"
                f"<td>{s.get('surface','')}</td>"
                f"<td>"
                f"<button onclick=\"relabel({s.get('ts',0)}, 'heavy')\">Heavy</button> "
                f"<button onclick=\"relabel({s.get('ts',0)}, 'medium')\">Medium</button> "
                f"<button onclick=\"relabel({s.get('ts',0)}, 'light')\">Light</button> "
                f"<button onclick=\"relabel({s.get('ts',0)}, 'discard')\">Discard</button>"
                f"</td>"
                f"</tr>"
            )

        html = f"""<!DOCTYPE html>
<html>
<head><title>Footstep Review</title>
<style>
  body {{ font-family: monospace; background: #111; color: #eee; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
  th {{ background: #222; }}
  tr:hover {{ background: #1a1a1a; }}
  button {{ margin: 2px; padding: 4px 10px; cursor: pointer; }}
  h2 {{ color: #ff4655; }}
</style>
</head>
<body>
<h2>Footstep Audio Review -- {len(samples)} samples</h2>
<table>
<thead><tr>
  <th>Timestamp</th><th>Map</th><th>Zone</th><th>Current Label</th>
  <th>Surface</th><th>Actions</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
<script>
async function relabel(ts, newLabel) {{
  const fd = new FormData();
  fd.append('label', newLabel);
  const r = await fetch('/label/' + ts, {{method: 'POST', body: fd}});
  if (r.ok) {{
    const row = document.querySelector(`tr[data-ts="${{ts}}"]`);
    location.reload();
  }}
}}
</script>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------
    # POST /label/<ts>
    # ------------------------------------------------------------------
    @app.post("/label/{ts}")
    async def relabel(
        ts: int,
        label: str = Form(...),
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ):
        _check_auth(x_api_key)
        # Find and update the metadata
        for meta_file in (DATA_DIR / "footstep_audio").rglob(f"{ts}.json"):
            meta = json.loads(meta_file.read_text())
            meta["label"] = label
            meta["relabeled"] = True
            meta_file.write_text(json.dumps(meta))
            return {"ok": True}
        raise HTTPException(status_code=404, detail="Sample not found")

    # ------------------------------------------------------------------
    # GET /download/<type>
    # ------------------------------------------------------------------
    @app.get("/download/{sample_type}")
    async def download(
        sample_type: str,
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ):
        _check_auth(x_api_key)
        type_dir = DATA_DIR / sample_type
        if not type_dir.exists():
            raise HTTPException(status_code=404, detail="No data for this type")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in type_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(DATA_DIR))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={sample_type}.zip"},
        )

    return app


# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Coach data collection server")
    parser.add_argument("--port",  type=int,  default=8000)
    parser.add_argument("--host",  type=str,  default="0.0.0.0")
    parser.add_argument("--key",   type=str,  default="", help="API key (leave blank to disable auth)")
    parser.add_argument("--data",  type=str,  default="server_data", help="Data storage directory")
    args = parser.parse_args()

    global API_KEY, DATA_DIR
    API_KEY  = args.key
    DATA_DIR = Path(args.data)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import uvicorn
    except ImportError:
        print("pip install uvicorn")
        return

    print(f"[Server] Data dir : {DATA_DIR.resolve()}")
    print(f"[Server] Auth     : {'enabled' if API_KEY else 'disabled'}")
    print(f"[Server] Review UI: http://{args.host}:{args.port}/review")
    print(f"[Server] Stats    : http://{args.host}:{args.port}/stats")

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
