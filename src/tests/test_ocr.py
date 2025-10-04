# --- src/tests/test_ocr.py ---
# Preview HSV-red Tesseract OCR on a chosen ROI (e.g., "awards")
# Controls:
#   q : quit
#   ⎵ : pause/resume
#   s : save current composite to runs/tess_ocr_<ROI>/

from __future__ import annotations
from pathlib import Path
import time
import cv2
import numpy as np

from processing.roi_cropper import ROICropper
from processing.ocr import DeepTextDetector, DeepOCRConfig

REPO_ROOT = Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "vids" / "1.MP4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"

# Only process/show this ROI (must match the YAML 'name')
TARGET_ROI_NAME = "kills"  # change if needed

EVERY_N = 60
MAX_W = 1000  # preview width cap

# --- HSV-red + Tesseract config (tweak as needed) ---
cfg = DeepOCRConfig(
    hsv_low1=(0, 80, 70),     # lower red
    hsv_high1=(10, 255, 255),
    hsv_low2=(170, 80, 70),   # upper red
    hsv_high2=(180, 255, 255),
    dilate_ksize=(2, 2),
    dilate_iters=1,           # increase to 2 if strokes are too thin
    upscale=3.0,              # 2.0–3.0 is good for small HUD text
    tesseract_oem=3,
    tesseract_psm=7,          # single line (HUD ribbons)
    whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ",
    blacklist=None,
    min_conf_keep=40.0,
    tesseract_cmd=None,       # set to a path if tesseract isn't on PATH
)

detector = DeepTextDetector(cfg)

def side_by_side(a: np.ndarray, b: np.ndarray, max_w: int) -> np.ndarray:
    """Hstack raw and annotated views and scale down if too wide."""
    h = min(a.shape[0], b.shape[0])
    def rh(img):
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*s), h), interpolation=cv2.INTER_AREA)
    cat = cv2.hconcat([rh(a), rh(b)])
    if cat.shape[1] > max_w:
        s = max_w / cat.shape[1]
        cat = cv2.resize(cat, (int(cat.shape[1]*s), int(cat.shape[0]*s)), interpolation=cv2.INTER_AREA)
    return cat

def main():
    assert VIDEO.exists(), f"Video not found: {VIDEO}"
    assert YAML.exists(), f"ROI YAML not found: {YAML}"

    cropper = ROICropper(str(VIDEO), str(YAML))
    cv2.namedWindow("HSV-Red Tesseract OCR", cv2.WINDOW_NORMAL)
    paused = False

    print("Starting OCR preview... (q quit, space pause, s save)")

    for batch in cropper.iter_all_crops(every_n=EVERY_N):
        for s in batch:
            # Only process frames whose ROI name matches TARGET_ROI_NAME
            if s.roi_name.lower() != TARGET_ROI_NAME.lower():
                continue

            raw = s.crop.copy()
            result = detector.detect(s)
            ann = detector.draw_annotations(s, result, font_scale=0.55, thickness=1)

            timecode = getattr(s, "timecode", None) or cropper.seconds_to_timecode(s.t_sec)
            avg_conf = float(np.mean([sp.conf for sp in result.spans])) if result.spans else 0.0
            header = f"{s.roi_name}  t={timecode}  frame={s.frame_idx}  conf~{avg_conf:.1f}"
            text_line = (result.all_text[:80] + "…") if len(result.all_text) > 80 else result.all_text

            viz = side_by_side(raw, ann, MAX_W)
            # Overlay header + text
            y0 = 24
            # cv2.rectangle(viz, (0, 0), (viz.shape[1], y0 + 28), (0, 0, 0), -1)
            # cv2.putText(viz, header, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(viz, f"text: {text_line}", (8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

            cv2.imshow("HSV-Red Tesseract OCR", viz)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                out_dir = REPO_ROOT / f"runs/tess_ocr_{TARGET_ROI_NAME.lower()}"
                out_dir.mkdir(parents=True, exist_ok=True)
                safe_tc = timecode.replace(":", "-")
                out_path = out_dir / f"{s.frame_idx:08d}_{s.roi_name}_{safe_tc}.jpg"
                # cv2.imwrite(str(out_path), viz)
                print(f"Saved: {out_path}")

            # Slow slightly for readability (adjust/remove as needed)
            time.sleep(0.10)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
