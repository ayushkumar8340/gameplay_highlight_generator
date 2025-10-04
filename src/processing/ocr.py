# src/processing/ocr.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import pytesseract


# ---------- Project data models ----------

@dataclass(frozen=True)
class MultiCropSample:
    """One cropped sample for a specific ROI/class."""
    roi_name: str
    frame_idx: int
    t_sec: float
    timecode: str
    crop: np.ndarray  # BGR


@dataclass(frozen=True)
class OCRSpan:
    """One recognized span (word) with bbox and confidence."""
    text: str
    conf: float                          # 0–100
    bbox: Tuple[int, int, int, int]      # x, y, w, h
    level: int = 5
    line_num: int = 0
    block_num: int = 0
    par_num: int = 0


@dataclass(frozen=True)
class OCRResult:
    """OCR result aligned with your pipeline."""
    roi_name: str
    frame_idx: int
    t_sec: float
    timecode: str
    all_text: str
    spans: List[OCRSpan]
    meta: Dict[str, Any]


# ---------- Config for the red-HUD Tesseract pipeline ----------

@dataclass
class DeepOCRConfig:
    # HSV thresholds for red/orange text (two ranges because red wraps around hue=0)
    hsv_low1: Tuple[int, int, int] = (0, 80, 70)
    hsv_high1: Tuple[int, int, int] = (10, 255, 255)
    hsv_low2: Tuple[int, int, int] = (170, 80, 70)
    hsv_high2: Tuple[int, int, int] = (180, 255, 255)

    # Morphology to thicken very thin strokes after inversion
    dilate_ksize: Tuple[int, int] = (2, 2)
    dilate_iters: int = 1

    # Upscale factor before OCR (thickens strokes; use 2–3 for tiny HUD text)
    upscale: float = 3.0

    # Tesseract options
    tesseract_oem: int = 3           # LSTM default
    tesseract_psm: int = 7           # single line (good for HUD ribbons)
    whitelist: Optional[str] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
    blacklist: Optional[str] = None
    min_conf_keep: float = 40.0      # keep words with conf >= this (0..100)

    # If you need a custom tesseract binary path:
    tesseract_cmd: Optional[str] = None


class DeepTextDetector:
    """
    Red/Orange HUD OCR using:
      1) HSV dual-range red mask
      2) Invert to black-on-white
      3) Dilate (thicken)
      4) Upscale (CUBIC)
      5) Tesseract OCR (psm=7)

    Exposes the same API you've used elsewhere: detect(), detect_batch(), draw_annotations().
    """
    def __init__(self, cfg: DeepOCRConfig = DeepOCRConfig()):
        self.cfg = cfg
        if cfg.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

    # ---------- Public API ----------

    def detect(self, sample: MultiCropSample) -> OCRResult:
        bgr = sample.crop
        proc, meta = self._preprocess(bgr)

        # Build Tesseract config
        cfg_parts = [f"--oem {self.cfg.tesseract_oem}", f"--psm {self.cfg.tesseract_psm}"]
        if self.cfg.whitelist:
            cfg_parts.append(f'-c tessedit_char_whitelist="{self.cfg.whitelist}"')
        if self.cfg.blacklist:
            cfg_parts.append(f'-c tessedit_char_blacklist="{self.cfg.blacklist}"')
        cfg_str = " ".join(cfg_parts)

        # 1) All-text (quick)
        all_text = pytesseract.image_to_string(proc, config=cfg_str).strip()

        # 2) Word-level boxes
        data = pytesseract.image_to_data(proc, config=cfg_str, output_type=pytesseract.Output.DICT)

        spans: List[OCRSpan] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw)
            except Exception:
                conf = -1.0
            if not text or conf < self.cfg.min_conf_keep:
                continue

            x = int(data["left"][i]); y = int(data["top"][i])
            w = int(data["width"][i]); h = int(data["height"][i])
            level = int(data.get("level", [5]*n)[i])
            line_num = int(data.get("line_num", [0]*n)[i]) if "line_num" in data else 0
            block_num = int(data.get("block_num", [0]*n)[i]) if "block_num" in data else 0
            par_num = int(data.get("par_num", [0]*n)[i]) if "par_num" in data else 0

            spans.append(OCRSpan(
                text=text,
                conf=conf,
                bbox=(x, y, w, h),
                level=level,
                line_num=line_num,
                block_num=block_num,
                par_num=par_num,
            ))

        meta.update({
            "backend": "Tesseract(HSV-red)",
            "psm": self.cfg.tesseract_psm,
            "oem": self.cfg.tesseract_oem,
            "upscale": self.cfg.upscale,
            "dilate_iters": self.cfg.dilate_iters,
        })

        return OCRResult(
            roi_name=sample.roi_name,
            frame_idx=sample.frame_idx,
            t_sec=sample.t_sec,
            timecode=sample.timecode,
            all_text=all_text,
            spans=spans,
            meta=meta,
        )

    def detect_batch(self, samples: List[MultiCropSample]) -> List[OCRResult]:
        return [self.detect(s) for s in samples]

    def draw_annotations(
        self,
        sample: MultiCropSample,
        result: OCRResult,
        font_scale: float = 0.6,
        thickness: int = 1,
    ) -> np.ndarray:
        vis = sample.crop.copy()
        for span in result.spans:
            x, y, w, h = span.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
            label = f"{span.text} ({int(span.conf)})"
            ytxt = max(0, y - 3)
            cv2.putText(vis, label, (x, ytxt), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(vis, label, (x, ytxt), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

        # small footer with meta
        tag = result.meta.get("backend", "tess")
        cv2.putText(vis, f"{tag}", (8, vis.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{tag}", (8, vis.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        return vis

    # ---------- Internals (your method, generalized) ----------

    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Produce a single-channel, OCR-ready image:
          - Mask red/orange in HSV (two ranges)
          - Invert (black text on white)
          - Dilate to thicken strokes
          - Upscale (cubic)
        """
        meta: Dict[str, Any] = {}

        # 1) HSV mask for red/orange overlays
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low1 = np.array(self.cfg.hsv_low1, dtype=np.uint8)
        high1 = np.array(self.cfg.hsv_high1, dtype=np.uint8)
        low2 = np.array(self.cfg.hsv_low2, dtype=np.uint8)
        high2 = np.array(self.cfg.hsv_high2, dtype=np.uint8)

        m1 = cv2.inRange(hsv, low1, high1)
        m2 = cv2.inRange(hsv, low2, high2)
        mask = cv2.bitwise_or(m1, m2)

        # Minor cleanup on the mask
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        # 2) Invert so letters are black on white (Tesseract prefers this)
        inv = 255 - mask

        # 3) Dilate to thicken thin strokes
        if self.cfg.dilate_iters > 0:
            kd = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg.dilate_ksize)
            inv = cv2.dilate(inv, kd, iterations=self.cfg.dilate_iters)

        # 4) Upscale
        if self.cfg.upscale and self.cfg.upscale != 1.0:
            h, w = inv.shape[:2]
            inv = cv2.resize(inv, (int(w * self.cfg.upscale), int(h * self.cfg.upscale)),
                             interpolation=cv2.INTER_CUBIC)

        meta["shape"] = inv.shape
        meta["variant"] = "hsv_red_mask_invert_dilate_upscale"
        return inv, meta
