# --- src/tests/test_rules_pipeline.py ---
from pathlib import Path
import cv2
from tqdm import tqdm  # <-- for progress bar

from processing.roi_cropper import ROICropper
from processing.ocr import DeepTextDetector, DeepOCRConfig, MultiCropSample
from processing.health_bar import HealthBarColorAlertDetector
from processing.awards import WhiteHudTextDetector, WhiteTextOCRConfig
from processing.rules import HUDRulesEngine, RulesConfig

REPO_ROOT = Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "vids" / "1.MP4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"

# ROI names (must match YAML)
KILLS_ROI_NAME  = "kills"
HEALTH_ROI_NAME = "health"
TEXT_ROI_NAME   = "awards"

EVERY_N = 60
PREVIEW = True  # set True if you want to see frames when events occur

# --- detectors ---
ocr_cfg = DeepOCRConfig(
    hsv_low1=(0, 80, 70), hsv_high1=(10, 255, 255),
    hsv_low2=(170, 80, 70), hsv_high2=(180, 255, 255),
    close_ksize=(2, 2), close_iters=1,
    dilate_ksize=(2, 2), dilate_iters=1,
    upscale=2.0, languages=("en",), gpu=False,
    allowlist="0123456789",
    text_threshold=0.4, low_text=0.3, link_threshold=0.4,
    min_conf_keep=0.40, sort_left_to_right=True,
)
hud_cfg = WhiteTextOCRConfig(
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    gpu=False, min_conf_keep=0.40,
)

ocr  = DeepTextDetector(ocr_cfg)
hdet = HealthBarColorAlertDetector()
hud  = WhiteHudTextDetector(hud_cfg)

# --- rules engine ---
rules = HUDRulesEngine(RulesConfig(
    max_kill_jump=3,
    min_conf_accept=0.45,
    reset_if_zero_after_s=30.0,
    award_words=("streak", "treak", "armor", "armor breaker"),
))

# --- pipeline ---
assert VIDEO.exists(), f"Video not found: {VIDEO}"
assert YAML.exists(),  f"ROI YAML not found: {YAML}"

cropper = ROICropper(str(VIDEO), str(YAML))
total_frames = cropper.frame_count

if PREVIEW:
    cv2.namedWindow("FRAME", cv2.WINDOW_NORMAL)

print("Running rules pipeline... (q to quit preview, if enabled)")

# --- progress bar setup ---
num_samples = total_frames // EVERY_N if EVERY_N > 0 else total_frames
progress = tqdm(total=num_samples, desc="Processing frames", unit="batch")

# --- main loop ---
for batch in cropper.iter_all_crops(every_n=EVERY_N):
    progress.update(1)

    frame_idx = None
    t_sec = None
    timecode = None

    kills_raw_text = ""
    kills_avg_conf = 0.0

    health_state = "unknown"
    health_low = False
    health_sev = 0.0

    awards_text = ""

    # --- process all ROI samples in this batch ---
    for s in batch:
        frame_idx = s.frame_idx
        t_sec     = s.t_sec
        timecode  = s.timecode

        if s.roi_name == KILLS_ROI_NAME:
            sample = MultiCropSample(
                roi_name=s.roi_name, frame_idx=s.frame_idx, t_sec=s.t_sec,
                timecode=s.timecode, crop=s.crop.copy()
            )
            res = ocr.detect(sample)
            kills_raw_text = res.all_text.strip()
            if res.spans:
                kills_avg_conf = sum(sp.conf for sp in res.spans) / len(res.spans)

        elif s.roi_name == HEALTH_ROI_NAME:
            det = hdet.detect_one(s)
            health_state = det.state
            health_low   = (det.state == "low_health")
            health_sev   = float(getattr(det, "severity", 0.0) or 0.0)

        elif s.roi_name == TEXT_ROI_NAME:
            res = hud.detect(s)
            awards_text = res.all_text.strip()

    # --- run rules for this frame ---
    if frame_idx is not None:
        events = rules.process(
            frame_idx=frame_idx,
            t_sec=float(t_sec or 0.0),
            timecode=timecode or "",
            kills_raw_text=kills_raw_text,
            kills_avg_conf=float(kills_avg_conf or 0.0),
            health_state=health_state,
            health_low=bool(health_low),
            health_severity=float(health_sev or 0.0),
            awards_text=awards_text,
        )

        # --- show only when events are detected ---
        if PREVIEW and events:
            full = cropper.get_original_frame_by_index(frame_idx)
            frame_vis = full.frame.copy()

            # overlay event info
            y = 32
            for ev in events:
                msg = ev.type.replace("_", " ").upper()
                cv2.putText(frame_vis, f"[{full.timecode}] {msg}",
                            (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                y += 32

            cv2.imshow("FRAME", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # --- console log of detected events ---
        for ev in events:
            if ev.type == "kill_while_low_health":
                print(f"[{ev.timecode}] --- LOW HEALTH KILLS EVENT ---")
            elif ev.type == "false_kill_while_low_health":
                print(f"[{ev.timecode}] --- LOW HEALTH KILLS EVENT ---")
            elif ev.type == "award_word_detected":
                print(f"[{ev.timecode}] --- EVENT AWARDS ---")

progress.close()
cv2.destroyAllWindows()
print("Done.")
