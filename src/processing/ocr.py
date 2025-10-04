# src/processing/ocr.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import re

# ---- Pretrained Deep OCR (EasyOCR) ----
# pip install easyocr torch torchvision
try:
    import easyocr
except Exception:
    easyocr = None  # clear error is raised if detect() is called


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


# ---------- Config for pretrained DeepOCR (digits-only) ----------

@dataclass
class DeepOCRConfig:
    # Broader orange/red coverage (HUD text is often orange-ish)
    hsv_low1: Tuple[int, int, int] = (0, 60, 60)
    hsv_high1: Tuple[int, int, int] = (25, 255, 255)     # up to orange
    hsv_low2: Tuple[int, int, int] = (160, 60, 60)       # deeper reds
    hsv_high2: Tuple[int, int, int] = (180, 255, 255)

    # Mask cleanup
    close_ksize: Tuple[int, int] = (2, 2)
    close_iters: int = 1
    dilate_ksize: Tuple[int, int] = (2, 2)
    dilate_iters: int = 1

    # Optional upscale (helps tiny HUD text)
    upscale: float = 2.0

    # EasyOCR settings
    languages: Tuple[str, ...] = ("en",)
    gpu: bool = False                     # set True if you have CUDA
    allowlist: str = "0123456789"         # digits only
    text_threshold: float = 0.4
    low_text: float = 0.3
    link_threshold: float = 0.4
    paragraph: bool = False
    rotation_info: Optional[List[int]] = None  # e.g. [0, 90, 180, 270]

    # Post-filtering
    min_conf_keep: float = 0.35           # a bit lower helps tiny HUD
    sort_left_to_right: bool = True


class DeepTextDetector:
    """
    Pretrained DeepOCR (EasyOCR) red-digits OCR:
      1) HSV dual-range red/orange mask -> keep only red/orange regions
      2) CLAHE contrast boost + optional upscale
      3) EasyOCR.readtext with digits allowlist
      4) Strictly keep digits; drop letters (e.g., 'eliminations')
      5) Gray fallback path if color mask underperforms
    """
    def __init__(self, cfg: DeepOCRConfig = DeepOCRConfig()):
        self.cfg = cfg
        self._reader: Optional["easyocr.Reader"] = None  # lazy init

    # ---------- Public API ----------

    def detect(self, sample: MultiCropSample) -> OCRResult:
        if easyocr is None:
            raise RuntimeError(
                "EasyOCR is not installed. Run `pip install easyocr torch torchvision`."
            )

        # Lazy-load the pretrained reader
        if self._reader is None:
            self._reader = easyocr.Reader(
                list(self.cfg.languages),
                gpu=self.cfg.gpu,
                verbose=False
            )

        bgr = sample.crop

        proc_color, meta_color = self._preprocess_color_mask(bgr)
        digits_color, spans_color = self._easyocr_digits(proc_color)

        proc_gray, meta_gray = self._preprocess_gray_boost(bgr)
        digits_gray, spans_gray = self._easyocr_digits(proc_gray, force_rgb=False)

        def score(d: str, spans: List[OCRSpan]):
            if not d:
                return (-1, 0.0)
            mean_conf = float(np.mean([s.conf for s in spans])) if spans else 0.0
            return (len(d), mean_conf)

        sc_color = score(digits_color, spans_color)
        sc_gray  = score(digits_gray, spans_gray)

        if sc_gray > sc_color:
            all_text = digits_gray
            spans = spans_gray
            meta = meta_gray
            meta["path"] = "gray_fallback"
        else:
            all_text = digits_color
            spans = spans_color
            meta = meta_color
            meta["path"] = "color_mask"

        meta.update({
            "backend": "EasyOCR(pretrained, digits-only)",
            "num_spans": len(spans),
            "allowlist": self.cfg.allowlist,
            "upscale": self.cfg.upscale,
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

        tag = result.meta.get("backend", "DeepOCR")
        cv2.putText(vis, f"{tag}: {result.all_text}", (8, vis.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{tag}: {result.all_text}", (8, vis.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        return vis

    # ---------- Internals ----------

    def _easyocr_digits(self, bgr_or_gray: np.ndarray, force_rgb: bool = True):
        """Run EasyOCR and return (concatenated_digits, spans)."""
        if force_rgb:
            rgb = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2RGB)
        else:
            if len(bgr_or_gray.shape) == 2:  # grayscale
                rgb = bgr_or_gray
            else:
                rgb = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2RGB)

        results = self._reader.readtext(
            rgb,
            detail=1,
            paragraph=self.cfg.paragraph,
            rotation_info=self.cfg.rotation_info,
            allowlist=self.cfg.allowlist,
            text_threshold=self.cfg.text_threshold,
            low_text=self.cfg.low_text,
            link_threshold=self.cfg.link_threshold,
            mag_ratio=1.0,
            slope_ths=0.1,
            ycenter_ths=0.7,
            height_ths=0.6,
        )

        spans: List[OCRSpan] = []
        for det in results:
            # det = [box(points), text, conf]
            box_pts, text, conf = det
            text = (text or "").strip()
            # STRICT numeric cleanup
            text_digits = "".join(ch for ch in text if ch.isdigit())
            if not text_digits:
                continue
            if conf < self.cfg.min_conf_keep:
                continue

            xs = [int(p[0]) for p in box_pts]
            ys = [int(p[1]) for p in box_pts]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y

            spans.append(OCRSpan(
                text=text_digits,
                conf=float(conf * 100.0),   # 0–100
                bbox=(x, y, w, h),
                level=5
            ))

        if self.cfg.sort_left_to_right:
            spans.sort(key=lambda s: s.bbox[0])

        # Concatenate only digits from all spans, then final guard via regex
        all_digits = "".join(s.text for s in spans)
        m = re.findall(r"\d+", all_digits)
        all_digits = "".join(m) if m else ""

        return all_digits, spans

    def _preprocess_color_mask(self, bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Keep only red/orange regions for OCR; mild denoise; optional upscale; CLAHE.
        Returns a BGR image (masked) suitable for EasyOCR.
        """
        meta: Dict[str, Any] = {}
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low1 = np.array(self.cfg.hsv_low1, dtype=np.uint8)
        high1 = np.array(self.cfg.hsv_high1, dtype=np.uint8)
        low2 = np.array(self.cfg.hsv_low2, dtype=np.uint8)
        high2 = np.array(self.cfg.hsv_high2, dtype=np.uint8)

        m1 = cv2.inRange(hsv, low1, high1)
        m2 = cv2.inRange(hsv, low2, high2)
        mask = cv2.bitwise_or(m1, m2)

        if self.cfg.close_iters > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg.close_ksize)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=self.cfg.close_iters)
        if self.cfg.dilate_iters > 0:
            kd = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg.dilate_ksize)
            mask = cv2.dilate(mask, kd, iterations=self.cfg.dilate_iters)

        # Keep only red/orange pixels
        masked = cv2.bitwise_and(bgr, bgr, mask=mask)

        # Optional upscale for tiny HUD text
        if self.cfg.upscale and self.cfg.upscale != 1.0:
            h, w = masked.shape[:2]
            masked = cv2.resize(
                masked, (int(w * self.cfg.upscale), int(h * self.cfg.upscale)),
                interpolation=cv2.INTER_CUBIC
            )

        # Mild contrast lift (CLAHE on L channel)
        lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        masked = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        meta["shape"] = masked.shape
        meta["variant"] = "color_mask"
        return masked, meta

    def _preprocess_gray_boost(self, bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Gray fallback with gamma + CLAHE. Returns single-channel image.
        """
        meta: Dict[str, Any] = {}
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # gamma (0.7) brightens midtones
        gray = np.clip((gray / 255.0) ** 0.7 * 255.0, 0, 255).astype(np.uint8)
        # local contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if self.cfg.upscale and self.cfg.upscale != 1.0:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (int(w * self.cfg.upscale), int(h * self.cfg.upscale)),
                              interpolation=cv2.INTER_CUBIC)

        meta["shape"] = gray.shape
        meta["variant"] = "gray_boost"
        return gray, meta
