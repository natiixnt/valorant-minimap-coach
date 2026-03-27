#!/usr/bin/env python3
"""
One-time calibration tool.

Run this while Valorant is open (it waits 3 s so you can alt-tab).
Click the top-left corner of the minimap, then the bottom-right corner.
The minimap region is saved to config.yaml automatically.
"""
import time

import cv2
import mss
import numpy as np
import yaml


def take_screenshot() -> np.ndarray:
    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[0])  # monitors[0] = virtual desktop spanning all displays
    return np.array(raw)[:, :, :3]


def draw_grid(img: np.ndarray, step: int = 50) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (80, 80, 80), 1)
        if x % 200 == 0:
            cv2.putText(out, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (80, 80, 80), 1)
        if y % 200 == 0:
            cv2.putText(out, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return out


def select_minimap(img: np.ndarray):
    points = []
    display = draw_grid(img)
    title = "Click: top-left then bottom-right of minimap | press any key when done"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            print(f"  Point {len(points)}: ({x}, {y})")
            if len(points) == 2:
                cv2.rectangle(display, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow(title, display)

    cv2.imshow(title, display)
    cv2.setMouseCallback(title, on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points


def save_region(region: dict, path: str = "config.yaml") -> None:
    try:
        with open(path) as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[Calibrate] {path} not found. Create it before running calibrate.")
        return
    config.setdefault("minimap", {})["region"] = region
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[Calibrate] Saved to {path}: {region}")


def main() -> None:
    print("[Calibrate] Switch to Valorant. Capturing in 3 s...")
    time.sleep(3)

    img = take_screenshot()
    cv2.imwrite("calibrate_screenshot.png", img)
    print("[Calibrate] Screenshot saved: calibrate_screenshot.png")

    points = select_minimap(img)

    if len(points) < 2:
        print("[Calibrate] Need 2 points. Update config.yaml manually.")
        return

    x1, y1 = points[0]
    x2, y2 = points[1]
    region = {
        "top": min(y1, y2),
        "left": min(x1, x2),
        "width": abs(x2 - x1),
        "height": abs(y2 - y1),
    }
    save_region(region)


if __name__ == "__main__":
    main()
