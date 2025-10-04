# --- src/tests/test_ocr.py ---
# Pretrained DeepOCR (digits-only, red) on a chosen ROI (e.g., "kills")
# Controls:
#   q : quit
#   ⎵ : pause/resume
#   s : save current composite to runs/deepocr_<ROI>/

from pathlib import Path
import time
import cv2
import numpy as np

from processing.roi_cropper import ROICropper
from processing.ocr import DeepTextDetector, DeepOCRConfig, MultiCropSample

REPO_ROOT = Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "vids" / "1.MP4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"   # change if your file name differs

# Only this ROI name will be OCR'd (must match YAML 'name')
TARGET_ROI_NAME = "kills"

EVERY_N = 60  # preview every Nth frame
MAX_W = 1000  # optional resize for the preview window

# --- pretrained DeepOCR config (digits-only, red) ---
cfg = DeepOCRConfig(
    hsv_low1=(0, 80, 70),
    hsv_high1=(10, 255, 255),
    hsv_low2=(170, 80, 70),
    hsv_high2=(180, 255, 255),
    close_ksize=(2, 2),
    close_iters=1,
    dilate_ksize=(2, 2),
    dilate_iters=1,
    upscale=2.0,            # bump to 3.0 for tiny HUD text
    languages=("en",),
    gpu=False,
    allowlist="0123456789",
    text_threshold=0.4,
    low_text=0.3,
    link_threshold=0.4,
    min_conf_keep=0.40,
    sort_left_to_right=True,
)

detector = DeepTextDetector(cfg)

def side_by_side(a: np.ndarray, b: np.ndarray, max_w: int) -> np.ndarray:
    h = min(a.shape[0], b.shape[0])
    def rh(img):
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*s), h), interpolation=cv2.INTER_AREA)
    cat = cv2.hconcat([rh(a), rh(b)])
    if cat.shape[1] > max_w:
        s = max_w / cat.shape[1]
        cat = cv2.resize(cat, (int(cat.shape[1]*s), int(cat.shape[0]*s)), interpolation=cv2.INTER_AREA)
    return cat

# init
assert VIDEO.exists(), f"Video not found: {VIDEO}"
assert YAML.exists(), f"ROI YAML not found: {YAML}"

cropper = ROICropper(str(VIDEO), str(YAML))

cv2.namedWindow("OCR Preview", cv2.WINDOW_NORMAL)
paused = False

print("Starting OCR preview... (q quit, space pause, s save)")

for batch in cropper.iter_all_crops(every_n=EVERY_N):
    for s in batch:
        if s.roi_name != TARGET_ROI_NAME:
            continue

        # use crop directly (no rotation)
        sample = MultiCropSample(
            roi_name=s.roi_name, frame_idx=s.frame_idx, t_sec=s.t_sec, timecode=s.timecode, crop=s.crop.copy()
        )

        t0 = time.time()
        result = detector.detect(sample)
        ms = (time.time() - t0) * 1000.0

        ann = detector.draw_annotations(sample, result, font_scale=0.55, thickness=1)
        viz = side_by_side(sample.crop, ann, MAX_W)

        # label line (printed)
        avg_conf = float(np.mean([sp.conf for sp in result.spans])) if result.spans else 0.0
        tc = s.timecode
        print(f"{s.frame_idx:6d} {tc}  {s.roi_name:<12}  '{result.all_text}'  ~{avg_conf:4.1f}%  {ms:5.1f}ms")

        cv2.imshow("OCR Preview", viz)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            out_dir = REPO_ROOT / f"runs/deepocr_{TARGET_ROI_NAME.lower()}"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_tc = tc.replace(":", "-")
            out_path = out_dir / f"{s.frame_idx:08d}_{s.roi_name}_{safe_tc}.jpg"
            cv2.imwrite(str(out_path), viz)
            print(f"Saved: {out_path}")

        time.sleep(0.10)

cv2.destroyAllWindows()
