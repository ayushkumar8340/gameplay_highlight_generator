from tkinter import messagebox
import time
from pathlib import Path
import cv2
from tqdm import tqdm  

from processing.roi_cropper import ROICropper
from processing.ocr import DeepTextDetector, DeepOCRConfig, MultiCropSample
from processing.health_bar import HealthBarColorAlertDetector
from processing.awards import WhiteHudTextDetector, WhiteTextOCRConfig
from processing.rules import HUDRulesEngine, RulesConfig
from processing.caption_generator import ChatGPTImageUploader
from processing.caption_gen import TTSGenerator
from processing.editing_module import HighlightVideoCreator

REPO_ROOT = Path(__file__).resolve().parents[1].parent

EVENT_IMG_DIR = REPO_ROOT / "saved"
EVENT_IMG_DIR.mkdir(parents=True, exist_ok=True)

YAML  = REPO_ROOT / "rois" / "rois.yaml"

COMM_PATH = REPO_ROOT / "saved" / "gameplay_commentary.yaml"
OUT_PATH = REPO_ROOT / "highlights_output.mp4"
AUDIO_DIR = REPO_ROOT / "generated_voice"

# ROI names (must match YAML)
KILLS_ROI_NAME  = "kills"
HEALTH_ROI_NAME = "health"
TEXT_ROI_NAME   = "awards"

BUFFER_L_SECONDS = 4
BUFFER_R_SECONDS = 1

BUFFER = BUFFER_L_SECONDS + BUFFER_R_SECONDS

EVERY_N = 60
PREVIEW = False  # set True if you want to see frames when events occur

timestamps = []


def checkDiff(prev_ts: str, curr_ts: str) -> bool:
    if(abs(HighlightVideoCreator._to_seconds(prev_ts) - abs(HighlightVideoCreator._to_seconds(curr_ts))) < BUFFER):
        return False
    else:
        return True

def draw_bbox(img, x, y, w, h, color=(0, 255, 0), thickness=2):
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, color, thickness)
    return img

def run_entire_pipeline(video_path, log, set_progress, ui_ref, update_image):

    log("✔ Loaded video")
    log("✔ Initializing pipeline")

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
    assert YAML.exists(),  f"ROI YAML not found: {YAML}"

    cropper = ROICropper(str(video_path), str(YAML))
    total_frames = cropper.frame_count
    rois = cropper._getParsedData()

    prev_timestep = "0:00"
    i = 0
    count = 0

    total_frames = total_frames / EVERY_N

    # --- main loop ---
    for batch in cropper.iter_all_crops(every_n=EVERY_N):
        set_progress((i/total_frames) * 80)
        i+=1
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
                out_path = EVENT_IMG_DIR / f"{count}_img.png"
                cv2.imwrite(str(out_path),frame_vis)
                count=count+1
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            
            # --- console log of detected events ---
            for ev in events:
                full = cropper.get_original_frame_by_index(frame_idx)
                frame_vis = full.frame.copy()
                if ev.type == "kill_while_low_health":
                    if checkDiff(prev_timestep , ev.timecode):
                        timestamps.append(ev.timecode)
                        print(f"[{ev.timecode}] --- LOW HEALTH KILLS EVENT ---")
                        prev_timestep = ev.timecode
                        for roi in rois:
                            if roi.name == "health":
                                frame_vis = draw_bbox(frame_vis,roi.x,roi.y,roi.w,roi.h,(255,0,0))
                            elif roi.name == "kills":
                                frame_vis = draw_bbox(frame_vis,roi.x,roi.y,roi.w,roi.h,(0,0,0))
                        ui_ref.root.after(0, lambda f=frame_vis: update_image(f))
                elif ev.type == "false_kill_while_low_health":
                    if checkDiff(prev_timestep , ev.timecode):
                        prev_timestep = ev.timecode
                        timestamps.append(ev.timecode)
                        print(f"[{ev.timecode}] --- LOW HEALTH KILLS EVENT ---")
                        prev_timestep = ev.timecode
                        for roi in rois:
                            if roi.name == "health":
                                frame_vis = draw_bbox(frame_vis,roi.x,roi.y,roi.w,roi.h,(255,0,0))
                            elif roi.name == "kills":
                                frame_vis = draw_bbox(frame_vis,roi.x,roi.y,roi.w,roi.h,(0,0,0))
                        ui_ref.root.after(0, lambda f=frame_vis: update_image(f))
                elif ev.type == "award_word_detected":
                    if checkDiff(prev_timestep , ev.timecode):
                        prev_timestep = ev.timecode
                        timestamps.append(ev.timecode)
                        print(f"[{ev.timecode}] --- EVENT AWARDS ---")
                        prev_timestep = ev.timecode
                        for roi in rois:
                            if roi.name == "health":
                                frame_vis = draw_bbox(frame_vis,roi.x,roi.y,roi.w,roi.h,(255,0,0))
                            elif roi.name == "kills":
                                frame_vis = draw_bbox(frame_vis,roi.x,roi.y,roi.w,roi.h,(0,0,0))
                            elif roi.name == "awards":
                                frame_vis = draw_bbox(frame_vis,roi.x,roi.y,roi.w,roi.h,(0,0,255))
                        ui_ref.root.after(0, lambda f=frame_vis: update_image(f))


    log("✔ Generating Commentary")
    ## Next 
    messagebox.showinfo("Automation",
        "ChatGPT automation will run.\nDo not touch the keyboard."
    )

    # uploader = ChatGPTImageUploader(EVENT_IMG_DIR)
    # uploader.run()

    ui_ref._continue_flag.set(False)
    ui_ref.root.after(0, ui_ref.wait_for_user_to_continue)
    ui_ref.root.wait_variable(ui_ref._continue_flag)

    log("✔ Generating Audio")
    tts = TTSGenerator(COMM_PATH)
    tts.generate()

    set_progress(90)

    log("✔ Merging Video")

    vid_creator = HighlightVideoCreator(
        video_path=video_path,
        timestamps=timestamps,
        audio_dir=AUDIO_DIR,
        buffer_l_seconds=BUFFER_L_SECONDS,
        buffer_r_seconds=BUFFER_R_SECONDS,
        output_path=OUT_PATH,
    )

    vid_creator.create()
    
    set_progress(100)
    log("✔ DONE!")

    # -----------------------------------------------------------
    return OUT_PATH
