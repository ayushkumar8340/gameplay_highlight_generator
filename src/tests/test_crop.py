# --- add to src/tests/test_crop.py (or replace its main loop) ---
from pathlib import Path
import cv2
from processing.roi_cropper import ROICropper
import time

REPO_ROOT =  Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "test.mp4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"   # change if your file name differs

cropper = ROICropper(str(VIDEO), str(YAML))

def draw_label(img, text, org=(8, 24)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    pad = 4
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# Show all ROIs (classes) in one pass. Press 'q' to quit.
EVERY_N = 10  # preview every Nth frame
cv2.namedWindow("ROI Preview", cv2.WINDOW_NORMAL)

for batch in cropper.iter_all_crops(every_n=EVERY_N):
    for s in batch:
        frame = s.crop.copy()
        label = f"class={s.roi_name}  t={cropper.seconds_to_timecode(s.t_sec)}  frame={s.frame_idx}"
        draw_label(frame, label)

        # Optional: keep window manageable
        max_w = 900
        if frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

        cv2.imshow("ROI Preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit
        time.sleep(1)

cv2.destroyAllWindows()
