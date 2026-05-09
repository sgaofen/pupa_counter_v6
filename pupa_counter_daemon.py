"""Long-lived Python worker for the desktop app.

Loads the CNN + classifier ONCE at startup, then reads one JSON command
per line on stdin and writes one JSON response per line on stdout.

Eliminates the ~2s cold-start (torch import + model load + classifier
pickle) that a subprocess-per-request approach paid on every click.

Protocol (one JSON object per line, on stdin / stdout):

    request : {"id": <int>, "cmd": "detect", "imagePath": "<path>"}
    response: {"id": <int>, "ok": true,  "result": {<detection>}}

    request : {"id": <int>, "cmd": "ping"}
    response: {"id": <int>, "ok": true,  "result": "pong"}

    request : {"id": <int>, "cmd": "quit"}
    response: {"id": <int>, "ok": true}   then exit(0)

The <detection> payload matches the --json-out format emitted by
pupa_counter.py's CLI path.

Startup is announced with a sentinel line so the caller knows the
worker is ready:

    {"ready": true, "model": "<filename>", "classifier": "<filename>"}

Usage (typically invoked by Electron main):

    $ python pupa_counter_daemon.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# These pieces are imported from the existing pupa_counter.py so any
# future detection tweak stays in one place.
from pupa_counter import (  # type: ignore
    TinyUNet,
    predict_heatmap,
    extract_peaks,
    filter_false_positives,
    render_annotated,
    pick_device,
    device_description,
    CLASSIFIER_PATH,
)


def _resolve(env_name: str, default: Path) -> Path:
    """Env-var override → absolute path; else default."""
    raw = os.environ.get(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def _heat_bbox(heatmap: np.ndarray, blob_thr: float, pad: int):
    """Bounding box of high-confidence blob (used to crop out blank
    scanner background on LiDE-style scans where pupae fill < half
    the bed). Returns (x0, y0, x1, y1) or None if heatmap is empty."""
    above = heatmap > blob_thr
    rows = np.any(above, axis=1)
    if not rows.any():
        return None
    cols = np.any(above, axis=0)
    H, W = heatmap.shape
    y0 = int(rows.argmax())
    y1_inv = int(rows[::-1].argmax())
    x0 = int(cols.argmax())
    x1_inv = int(cols[::-1].argmax())
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(W, W - x1_inv + pad),
        min(H, H - y1_inv + pad),
    )


# --- Env-var configurable inference parameters --------------------------------
# Documented in pupa_counter_desktop's electron/main.js where the daemon is
# spawned. Defaults match the LiDE 300 ship config so a CLI run of the
# daemon gives the same answers as the desktop app.

_PEAK_THR = float(os.environ.get("PUPA_PEAK_THR", "0.50"))
_MIN_DIST = int(os.environ.get("PUPA_MIN_DIST", "3"))
_BBOX_CROP = os.environ.get("PUPA_BBOX_CROP", "1") == "1"
_BBOX_HEAT_THR = float(os.environ.get("PUPA_BBOX_HEAT_THR", "0.40"))
_BBOX_PAD = int(os.environ.get("PUPA_BBOX_PAD", "50"))
_CLF_PROB_THR = float(os.environ.get("PUPA_CLF_PROB_THR", "0.50"))


def main() -> None:
    # Wrap startup in a guard so missing/corrupt model files produce a
    # machine-readable error line before we exit — otherwise the desktop
    # app sees a silent EOF and has no way to distinguish "worker is
    # still loading" from "worker died".
    try:
        # Default to LiDE 300 if those weights are present (current ship);
        # otherwise fall back to the legacy v12 1200 dpi recipe.
        lide = HERE / "model" / "pupa_counter_lide300.pt"
        lide_clf = HERE / "model" / "peak_filter_clf_lide300.pkl"
        model_path = _resolve("PUPA_MODEL_PATH",
                              lide if lide.exists() else HERE / "model" / "pupa_counter_v12.pt")
        clf_path = _resolve("PUPA_CLF_PATH",
                            lide_clf if lide_clf.exists() else CLASSIFIER_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"model weights not found: {model_path}")
        device = pick_device()
        model = TinyUNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        classifier = None
        if clf_path.exists():
            with open(clf_path, "rb") as f:
                classifier = pickle.load(f)
    except Exception as exc:
        sys.stdout.write(json.dumps({
            "ready": False,
            "stage": "startup",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }) + "\n")
        sys.stdout.flush()
        sys.exit(2)

    sys.stdout.write(json.dumps({
        "ready": True,
        "device": str(device),
        "deviceName": device_description(device),
        "model": model_path.name,
        "classifier": clf_path.name if classifier is not None else None,
        "config": {
            "peak_thr": _PEAK_THR, "min_dist": _MIN_DIST,
            "bbox_crop": _BBOX_CROP, "clf_prob_thr": _CLF_PROB_THR,
        },
    }) + "\n")
    sys.stdout.flush()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _respond({"id": None, "ok": False, "error": f"bad JSON: {exc}"})
            continue

        rid = req.get("id")
        cmd = req.get("cmd")

        try:
            if cmd == "ping":
                _respond({"id": rid, "ok": True, "result": "pong"})
            elif cmd == "detect":
                result = _detect(req["imagePath"], model, device, classifier, model_path.name)
                _respond({"id": rid, "ok": True, "result": result})
            elif cmd == "quit":
                _respond({"id": rid, "ok": True})
                return
            else:
                _respond({"id": rid, "ok": False, "error": f"unknown cmd: {cmd!r}"})
        except Exception as exc:
            _respond({
                "id": rid, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })


def _respond(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _detect(image_path: str, model, device, classifier, model_name: str) -> dict:
    path = Path(image_path)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    heatmap = predict_heatmap(model, img_rgb, device)
    raw_peaks = extract_peaks(heatmap, threshold=_PEAK_THR, min_dist=_MIN_DIST)

    # Optional bbox crop: drop peaks outside the high-confidence blob
    # (used on LiDE 300 scans where pupae occupy only part of the bed).
    if _BBOX_CROP:
        bbox = _heat_bbox(heatmap, _BBOX_HEAT_THR, _BBOX_PAD)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            raw_peaks = [(x, y) for x, y in raw_peaks
                         if x0 <= x < x1 and y0 <= y < y1]

    peaks = (
        filter_false_positives(raw_peaks, heatmap, img_rgb, classifier,
                               threshold=_CLF_PROB_THR)
        if classifier is not None else raw_peaks
    )
    _, counts, per_pupa = render_annotated(img_bgr, peaks, path.name)
    h, w = img_bgr.shape[:2]
    return {
        "imagePath": str(path),
        "imageWidth": w,
        "imageHeight": h,
        "modelVersion": model_name,
        "pupae": [
            {
                "index": p["pupa_idx"],
                "x": p["x_pixel"],
                "y": p["y_pixel"],
                "rankPct": p["rank_pct"],
                "band": p["band"],
                "source": "cnn",
            }
            for p in per_pupa
        ],
        "counts": {
            "total": len(peaks),
            "top5Pct": counts["top_5_pct"],
            "rank5To25": counts["rank_5_to_25_pct"],
            "middle50": counts["middle_50_pct"],
            "bottom25": counts["bottom_25_pct"],
        },
        "yMin": counts["y_min_of_pupae"],
        "yMax": counts["y_max_of_pupae"],
    }


if __name__ == "__main__":
    main()
