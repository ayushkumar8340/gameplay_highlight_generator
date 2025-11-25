# --- add to src/tests/test_crop_health.py ---
from pathlib import Path
import time
import cv2

from processing.roi_cropper import ROICropper
from processing.health_bar import HealthBarColorAlertDetector

REPO_ROOT = Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "vids" / "1.mp4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"   # change if your file name differs

# Set this to your health ROI name from YAML (e.g., "nhealth")
HEALTH_ROI_NAME = "health"

EVERY_N = 60  # preview every Nth frame

# init
cropper = ROICropper(str(VIDEO), str(YAML))
detector = HealthBarColorAlertDetector()  # tweak thresholds if needed

cv2.namedWindow("ROI Preview", cv2.WINDOW_NORMAL)

for batch in cropper.iter_all_crops(every_n=EVERY_N):
    for s in batch:
        frame = s.crop.copy()

        # Only run detection on the health bar ROI (still show others with labels)
        label = ""
        if s.roi_name == HEALTH_ROI_NAME:
            det = detector.detect_one(s)
            if det.state == "low_health":
                label += f" | LOW ({det.tone}, sev={det.severity:.0f})"
            else:
                label += " | OK (gray)"
            
            # health_text = f" | health={det.percent:.1f}% ({det.state})"
            print(label)
            cv2.imshow("ROI Preview", frame)

        # label = f"class={s.roi_name}  t={cropper.seconds_to_timecode(s.t_sec)}  frame={s.frame_idx}{health_text}"

        # Draw label
        # y0 = 22
        # cv2.rectangle(frame, (0, 0), (frame.shape[1], y0 + 8), (0, 0, 0), -1)
        # cv2.putText(frame, label, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Optional: keep window manageable
        max_w = 900
        if frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit

        # Slow it a bit so you can read the overlay (adjust or remove)
        time.sleep(0.25)

cv2.destroyAllWindows()
