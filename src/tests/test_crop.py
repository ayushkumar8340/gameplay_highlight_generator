# --- add to src/tests/test_crop.py (or replace its main loop) ---
from pathlib import Path
import cv2
from processing.roi_cropper import ROICropper
import time

REPO_ROOT =  Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "vids" /"1.MP4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"   # change if your file name differs

cropper = ROICropper(str(VIDEO), str(YAML))

# Show all ROIs (classes) in one pass. Press 'q' to quit.
EVERY_N = 60  # preview every Nth frame
cv2.namedWindow("ROI Preview", cv2.WINDOW_NORMAL)

for batch in cropper.iter_all_crops(every_n=EVERY_N):
    for s in batch:
        frame = s.crop.copy()
        label = f"class={s.roi_name}  t={cropper.seconds_to_timecode(s.t_sec)}  frame={s.frame_idx}"

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
