# --- src/tests/test_merge_kills_health_awards.py ---
from pathlib import Path
import re
import time
import csv

import cv2
import numpy as np

from processing.roi_cropper import ROICropper
from processing.ocr import DeepTextDetector, DeepOCRConfig, MultiCropSample
from processing.health_bar import HealthBarColorAlertDetector
from processing.awards import WhiteHudTextDetector, WhiteTextOCRConfig

REPO_ROOT = Path(__file__).resolve().parents[1].parent

VIDEO = REPO_ROOT / "vids" / "1.MP4"
YAML  = REPO_ROOT / "rois" / "rois.yaml"

# ROI names (must match YAML)
KILLS_ROI_NAME   = "kills"
HEALTH_ROI_NAME  = "health"
TEXT_ROI_NAME    = "awards"

EVERY_N = 60
MAX_W   = 1100
PREVIEW = True

# --- DeepOCR config (digits-only, red) ---
ocr_cfg = DeepOCRConfig(
    hsv_low1=(0, 80, 70),
    hsv_high1=(10, 255, 255),
    hsv_low2=(170, 80, 70),
    hsv_high2=(180, 255, 255),
    close_ksize=(2, 2),
    close_iters=1,
    dilate_ksize=(2, 2),
    dilate_iters=1,
    upscale=2.0,
    languages=("en",),
    gpu=False,
    allowlist="0123456789",
    text_threshold=0.4,
    low_text=0.3,
    link_threshold=0.4,
    min_conf_keep=0.40,
    sort_left_to_right=True,
)

# --- White HUD text OCR config ---
hud_cfg = WhiteTextOCRConfig(
    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    gpu=False,
    min_conf_keep=0.40,
)

# --- kill detection heuristics ---
MAX_KILL_JUMP = 3
MIN_CONF_ACCEPT = 0.45
RESET_IF_ZERO_AFTER_S = 30

RUNS_DIR = REPO_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RUNS_DIR / f"timeline_{VIDEO.stem}_kills_health_awards.csv"

# ---------------- helpers ----------------
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

def extract_first_int(s: str) -> int | None:
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None

# Exact-match list (case-insensitive)
AWARD_WORDS = {"streak", "treak", "armor", "armor breaker"}

# --------------- init ---------------
assert VIDEO.exists(), f"Video not found: {VIDEO}"
assert YAML.exists(),  f"ROI YAML not found: {YAML}"

cropper = ROICropper(str(VIDEO), str(YAML))
ocr     = DeepTextDetector(ocr_cfg)
hdet    = HealthBarColorAlertDetector()
hud     = WhiteHudTextDetector(hud_cfg)

if PREVIEW:
    cv2.namedWindow("Merge Preview", cv2.WINDOW_NORMAL)

paused = False
last_good_kills: int | None = None
last_accept_t: float | None = None

# CSV header
with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "frame_idx","timecode","t_sec",
        "kills_raw_text","kills_avg_conf","kills_accepted",
        "kill_event","false_kill_event",
        "health_state","health_severity","health_tone",
        "health_low","kill_while_low","false_kill_while_low",
        "awards_text","streak_like"
    ])

# --------------- main loop ---------------
for batch in cropper.iter_all_crops(every_n=EVERY_N):
    frame_idx = None
    t_sec = None
    timecode = None

    # per-frame accumulators
    kills_raw_text = ""
    kills_avg_conf = 0.0
    kills_int_read = None
    kills_accepted = None
    kill_event = False
    false_kill_event = False

    health_state = "unknown"
    health_sev = 0.0
    health_tone = "unknown"
    health_low = False

    awards_text = ""
    streak_like = False

    kills_viz = None
    health_viz = None
    awards_viz = None

    for s in batch:
        frame_idx = s.frame_idx
        t_sec     = s.t_sec
        timecode  = s.timecode

        if s.roi_name == KILLS_ROI_NAME:
            sample = MultiCropSample(
                roi_name=s.roi_name, frame_idx=s.frame_idx, t_sec=s.t_sec,
                timecode=s.timecode, crop=s.crop.copy()
            )
            ocr_res = ocr.detect(sample)
            kills_raw_text = ocr_res.all_text.strip()
            kills_avg_conf = float(np.mean([sp.conf for sp in ocr_res.spans])) if ocr_res.spans else 0.0
            kills_int_read = extract_first_int(kills_raw_text)
            kills_viz = ocr.draw_annotations(sample, ocr_res, font_scale=0.55, thickness=1)

            now = t_sec if t_sec is not None else 0.0
            accept = False
            if kills_int_read is not None:
                if kills_avg_conf >= MIN_CONF_ACCEPT:
                    if last_good_kills is None:
                        accept = True
                    else:
                        diff = kills_int_read - last_good_kills
                        if 0 <= diff <= MAX_KILL_JUMP:
                            accept = True
                        elif kills_int_read == 0 and last_good_kills > 0:
                            if last_accept_t is None or (now - last_accept_t) >= RESET_IF_ZERO_AFTER_S:
                                accept = True

            if accept:
                prev = last_good_kills
                last_good_kills = kills_int_read
                last_accept_t = now
                kills_accepted = kills_int_read
                if prev is not None and kills_int_read is not None and kills_int_read > prev:
                    kill_event = True
            else:
                kills_accepted = last_good_kills
                if kills_int_read is not None:
                    false_kill_event = True

        elif s.roi_name == HEALTH_ROI_NAME:
            det = hdet.detect_one(s)
            health_state = det.state
            health_sev   = float(getattr(det, "severity", 0.0) or 0.0)
            health_tone  = getattr(det, "tone", "unknown") or "unknown"
            health_low   = (det.state == "low_health")

            health_viz = s.crop.copy()
            y0 = 22
            label = f"health: {health_state} (sev={health_sev:.0f}, tone={health_tone})"
            cv2.rectangle(health_viz, (0,0), (health_viz.shape[1], y0+8), (0,0,0), -1)
            cv2.putText(health_viz, label, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        elif s.roi_name == TEXT_ROI_NAME:
            res = hud.detect(s)
            awards_text = res.all_text.strip()
            streak_like = awards_text.lower() in AWARD_WORDS
            awards_viz = hud.draw_annotations(s, res)

    # co-occurrence checks
    kill_while_low = bool(kill_event and health_low)
    false_kill_while_low = bool(false_kill_event and health_low)

    # --- selective prints ---
    if kill_while_low:
        print(f"[{timecode}] *** KILL while LOW HEALTH ***  kills={kills_accepted}")
    if false_kill_while_low:
        print(f"[{timecode}] !!! FALSE KILL while LOW HEALTH !!!")
    if streak_like:
        print(f"[{timecode}] --- WORD DETECTED: '{awards_text}' ---")

    # write to CSV
    if frame_idx is not None:
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                frame_idx, timecode, t_sec if t_sec is not None else "",
                kills_raw_text, f"{kills_avg_conf:.3f}", kills_accepted if kills_accepted is not None else "",
                int(kill_event), int(false_kill_event),
                health_state, f"{health_sev:.1f}", health_tone,
                int(health_low), int(kill_while_low), int(false_kill_while_low),
                awards_text, int(streak_like),
            ])

    # preview window
    if PREVIEW and any(v is not None for v in (kills_viz, health_viz, awards_viz)):
        panes = [p for p in (kills_viz, health_viz, awards_viz) if p is not None]
        mosaic = panes[0]
        for p in panes[1:]:
            mosaic = side_by_side(mosaic, p, MAX_W)
        cv2.imshow("Merge Preview", mosaic)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            out_dir = RUNS_DIR / "merge_mosaics"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_tc = (timecode or "t").replace(":", "-")
            out_path = out_dir / f"{(frame_idx or 0):08d}_{safe_tc}.jpg"
            cv2.imwrite(str(out_path), mosaic)

        time.sleep(0.10)

cv2.destroyAllWindows()
print(f"\nTimeline written to: {CSV_PATH}")
