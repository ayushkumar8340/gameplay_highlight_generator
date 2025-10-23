# src/tests/test_ocr_hud_text.py
from pathlib import Path
import time
import cv2

from processing.roi_cropper import ROICropper
from processing.awards import WhiteHudTextDetector, WhiteTextOCRConfig

REPO_ROOT = Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "vids" / "1.MP4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"   # change if your file name differs

# Set this to your ROI name from YAML (e.g., "streak" or "legendary")
TEXT_ROI_NAME = "awards"

EVERY_N = 15  # preview every Nth frame
MAX_W = 1000

# ---- init
cfg = WhiteTextOCRConfig(
    # Add digits if your HUD words can contain numbers:
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    gpu=False,
    min_conf_keep=0.40,
)
detector = WhiteHudTextDetector(cfg)
cropper = ROICropper(str(VIDEO), str(YAML))

cv2.namedWindow("HUD Text ROI", cv2.WINDOW_NORMAL)

for batch in cropper.iter_all_crops(every_n=EVERY_N):
    for s in batch:
        frame = s.crop.copy()

        if s.roi_name == TEXT_ROI_NAME:
            res = detector.detect(s)
            vis = detector.draw_annotations(s, res)
            label = f"class={s.roi_name}  t={cropper.seconds_to_timecode(s.t_sec)}  frame={s.frame_idx}"
            if res.all_text:
                label += f" | text='{res.all_text}'"
                print(label)
            else:
                label += " | <no text>"
                print(label)
            disp = vis

            cv2.imshow("HUD Text ROI", disp)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit



cv2.destroyAllWindows()
