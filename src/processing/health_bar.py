# src/processing/detectors/health_bar_color_alert.py
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional
import cv2
import numpy as np

# ----- Your crop type (for reference) -----
# @dataclass(frozen=True)
# class MultiCropSample:
#     roi_name: str
#     frame_idx: int
#     t_sec: float
#     timecode: str
#     crop: np.ndarray  # BGR

State = Literal["normal_gray", "low_health"]
Tone  = Literal["light_red", "medium_red", "dark_red", "unknown"]

@dataclass(frozen=True)
class HealthColorDetection:
    roi_name: str
    frame_idx: int
    t_sec: float
    timecode: str
    state: State                # normal_gray or low_health
    severity: float             # 0..100 (only meaningful for low_health)
    tone: Tone                  # light_red / medium_red / dark_red / unknown
    red_ratio: float            # fraction of red-ish pixels (0..1)

class HealthBarColorAlertDetector:
    """
    Detects 'low health' when the bar turns pink/red (default is gray).
    Grades severity from color intensity: deeper + darker red => higher severity.

    How it works (summary):
      1) Convert to HSV and focus on the central band to avoid borders.
      2) Identify 'red-ish' pixels (dual red hue ranges).
      3) Compute a redness severity S = mean( sat_norm * (1 - val_norm) ) over red pixels.
      4) If red fraction < τ_red_frac, it's considered normal_gray; else low_health.
      5) Map S to tone buckets (light/medium/dark red) and 0..100 severity.

    Tunables below should cover your examples (gray default, pink variant, dark red).
    """

    def __init__(
        self,
        central_band_frac: float = 0.6,   # analyze middle 60% height to avoid borders
        min_red_fraction: float = 0.06,   # >= this fraction of red-ish pixels => low health
        hsv_red1=(np.array([0, 40, 60]),  np.array([10, 255, 255])),   # allow light-pink (sat>=40)
        hsv_red2=(np.array([170, 40, 60]), np.array([180, 255, 255])),
        # tone cutoffs on S (0..1): light < t1 <= medium < t2 <= dark
        tone_t1: float = 0.28,
        tone_t2: float = 0.55,
        make_debug_vis: bool = False
    ):
        self.central_band_frac = np.clip(central_band_frac, 0.2, 1.0)
        self.min_red_fraction = min_red_fraction
        self.hsv_red1 = hsv_red1
        self.hsv_red2 = hsv_red2
        self.tone_t1 = tone_t1
        self.tone_t2 = tone_t2
        self.make_debug_vis = make_debug_vis

    @staticmethod
    def _central_band(img: np.ndarray, frac: float) -> np.ndarray:
        h = img.shape[0]
        band_h = int(max(1, round(h * frac)))
        y0 = (h - band_h) // 2
        return img[y0:y0+band_h, :]

    def _analyze(self, crop_bgr: np.ndarray):
        if crop_bgr.size == 0:
            return 0.0, "normal_gray", "unknown", 0.0, None

        # Light smoothing for thin bars
        blur = cv2.GaussianBlur(crop_bgr, (3,3), 0)
        hsv_full = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        hsv = self._central_band(hsv_full, self.central_band_frac)

        # Red-ish mask including pink (dual range)
        red_mask = cv2.inRange(hsv, *self.hsv_red1) | cv2.inRange(hsv, *self.hsv_red2)
        red_mask_bool = red_mask.astype(bool)

        total_px = hsv.shape[0] * hsv.shape[1]
        red_px = int(np.count_nonzero(red_mask_bool))
        red_ratio = red_px / max(1, total_px)

        if red_ratio < self.min_red_fraction:
            # Mostly gray (default)
            severity = 0.0
            tone: Tone = "unknown"
            state: State = "normal_gray"
        else:
            # Compute redness severity using saturation and value
            S = hsv[...,1].astype(np.float32) / 255.0
            V = hsv[...,2].astype(np.float32) / 255.0
            S_red = S[red_mask_bool]
            V_red = V[red_mask_bool]
            if S_red.size == 0:
                severity = 0.0
                tone = "unknown"
                state = "normal_gray"
            else:
                # Deeper red => higher S, lower V
                score = (S_red) * (1.0 - V_red)
                S_mean = float(score.mean())
                severity = float(np.clip(100.0 * S_mean, 0.0, 100.0))
                # tone buckets
                if S_mean < self.tone_t1:
                    tone = "light_red"
                elif S_mean < self.tone_t2:
                    tone = "medium_red"
                else:
                    tone = "dark_red"
                state = "low_health"

        debug_vis = None
        if self.make_debug_vis:
            vis = crop_bgr.copy()
            band = self._central_band(vis, self.central_band_frac)
            overlay = band.copy()
            # visualize red mask translucently
            overlay[red_mask_bool] = (0, 0, 255)  # BGR: mark red area
            cv2.addWeighted(overlay, 0.35, band, 0.65, 0, band)
            debug_vis = vis

        return severity, state, tone, red_ratio, debug_vis

    # ---- public API ----
    def detect_one(self, sample) -> HealthColorDetection:
        sev, state, tone, red_ratio, _ = self._analyze(sample.crop)
        return HealthColorDetection(
            roi_name=sample.roi_name,
            frame_idx=sample.frame_idx,
            t_sec=sample.t_sec,
            timecode=sample.timecode,
            state=state,
            severity=sev,
            tone=tone,
            red_ratio=red_ratio
        )

    def detect_batch(self, samples: Iterable) -> List[HealthColorDetection]:
        return [self.detect_one(s) for s in samples]
