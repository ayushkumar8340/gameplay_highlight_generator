# --- src/tests/test_crop.py ---
from __future__ import annotations
from pathlib import Path
import time
import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "vids" / "1.MP4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"   # change if your file name differs

# Show all ROIs (classes) in one pass. Press 'q' to quit.
EVERY_N = 1  # preview every Nth frame
WINDOW   = "ROI Preview"

def seconds_to_timecode(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"

def draw_label(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.5, thickness=1):
    # Draw filled bg + then the text (keeps text readable)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    pad = 4
    cv2.rectangle(img, (x, y - th - baseline - pad), (x + tw + pad*2, y + pad), (0, 128, 0), -1)
    cv2.putText(img, text, (x + pad, y - baseline), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main():
    # Load ROIs from YAML (name, x, y, w, h) + base resolution
    with open(YAML, "r") as f:
        ycfg = yaml.safe_load(f)

    # Backward compatibility for base resolution keys
    if "base_resolution" in ycfg and isinstance(ycfg["base_resolution"], (list, tuple)) and len(ycfg["base_resolution"]) == 2:
        base_w, base_h = int(ycfg["base_resolution"][0]), int(ycfg["base_resolution"][1])
    else:
        base_w = int(ycfg.get("base_width", 1920))
        base_h = int(ycfg.get("base_height", 1080))

    areas = ycfg.get("areas", [])
    if not areas:
        raise RuntimeError("No 'areas' found in YAML. Expected: areas: [{name,x,y,w,h}, ...]")

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    max_w = 1000  # cap preview width

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % EVERY_N != 0:
            continue

        H, W = frame.shape[:2]
        sx = W / float(base_w)
        sy = H / float(base_h)

        t_sec = frame_idx / fps
        tc = seconds_to_timecode(t_sec)

        # Draw each ROI as a green rectangle with label
        for a in areas:
            try:
                name = str(a["name"])
                x = int(a["x"]); y = int(a["y"]); w = int(a["w"]); h = int(a["h"])
            except Exception:
                # Skip malformed entries
                continue

            X = int(round(x * sx))
            Y = int(round(y * sy))
            Wb = int(round(w * sx))
            Hb = int(round(h * sy))

            # Box
            cv2.rectangle(frame, (X, Y), (X + Wb, Y + Hb), (0, 255, 0), 2)

            # Label text: class, time, frame index
            label = f"class={name}  t={tc}  frame={frame_idx}"
            draw_label(frame, label, (X, max(Y, 18)))  # keep label inside the frame top

        # Keep window manageable
        if frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(300) & 0xFF
        if key == ord('q'):
            break
        # Optional slow-down to visually inspect
        # time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
