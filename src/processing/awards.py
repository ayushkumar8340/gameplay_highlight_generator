# src/processing/ocr_white_text.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import re

# Reuse your data models
from .ocr import MultiCropSample, OCRSpan, OCRResult

# ---- Pretrained Deep OCR (EasyOCR) ----
# pip install easyocr torch torchvision
try:
    import easyocr
except Exception:
    easyocr = None  # clear error is raised if detect() is called


# ---------- Config for WHITE HUD text (letters/digits) ----------

@dataclass
class WhiteTextOCRConfig:
    """
    Config for white-text HUD OCR.
    The text is white; background can vary. We therefore emphasize luminance,
    do CLAHE, adaptive thresholds (both normal and inverted), and run multi-scale OCR.
    """
    # EasyOCR
    languages: Tuple[str, ...] = ("en",)
    gpu: bool = False
    # Letters by default; you can append digits if your HUD mixes both
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    text_threshold: float = 0.5
    low_text: float = 0.25
    link_threshold: float = 0.5
    paragraph: bool = False
    rotation_info: Optional[List[int]] = None

    # Keep spans with at least this confidence (0–1)
    min_conf_keep: float = 0.40
    sort_left_to_right: bool = True

    # Multi-scale OCR helps tiny fonts
    scales: Tuple[float, ...] = (1.0, 1.4, 1.8, 2.2)

    # Adaptive threshold params for binary candidates
    adaptive_block: int = 31   # must be odd
    adaptive_C: int = -10

    # Morphology to clean speckles
    open_ksize: Tuple[int, int] = (2, 2)
    open_iters: int = 1

    # Optional upscaling before building candidates (helps very small ROIs)
    pre_upscale: float = 1.0

    # Final text post-filter (keeps A–Z, a–z, 0–9 and spaces/hyphen by default)
    keep_pattern: str = r"[A-Za-z0-9 -]+"


class WhiteHudTextDetector:
    """
    Pretrained DeepOCR (EasyOCR) for WHITE HUD text:
      1) Luminance boost (CLAHE) + optional pre-upscale
      2) Build multiple grayscale candidates:
         - raw gray
         - adaptive binary (white glyphs on dark)
         - inverted binary (white glyphs on bright)
      3) Run EasyOCR across a small scale pyramid
      4) Keep spans that match allowlist and min_conf; merge to `all_text`
    """
    def __init__(self, cfg: WhiteTextOCRConfig = WhiteTextOCRConfig()):
        self.cfg = cfg
        self._reader: Optional["easyocr.Reader"] = None  # lazy init
        self._keep_re = re.compile(self.cfg.keep_pattern)

    # ---------- Public API ----------

    def detect(self, sample: MultiCropSample) -> OCRResult:
        if easyocr is None:
            raise RuntimeError(
                "EasyOCR is not installed. Run `pip install easyocr torch torchvision`."
            )

        # Lazy-load reader
        if self._reader is None:
            self._reader = easyocr.Reader(
                list(self.cfg.languages),
                gpu=self.cfg.gpu,
                verbose=False
            )

        bgr = sample.crop
        candidates, cand_meta = self._build_candidates(bgr)

        best_spans: List[OCRSpan] = []
        best_text: str = ""
        best_score: Tuple[int, float] = (-1, 0.0)  # (len, mean_conf)
        best_meta: Dict[str, Any] = {}

        for idx, cand in enumerate(candidates):
            for s in self.cfg.scales:
                im = self._rescale(cand, s)
                spans = self._easyocr_text(im)
                all_text = self._merge_text(spans)

                if all_text:
                    mean_conf = float(np.mean([sp.conf for sp in spans])) if spans else 0.0
                    score = (len(all_text), mean_conf)
                    if score > best_score:
                        # Reproject boxes back to *candidate* (no need to undo scale as we drew on ROI space)
                        if abs(s - 1.0) > 1e-3:
                            inv = 1.0 / s
                            spans = [self._scale_span(sp, inv) for sp in spans]
                        best_spans = spans
                        best_text = all_text
                        best_score = score
                        best_meta = {"candidate": cand_meta[idx], "scale": s}

        meta = {
            "backend": "EasyOCR(pretrained, white-text)",
            "num_spans": len(best_spans),
            "allowlist": self.cfg.allowlist,
            **best_meta
        }

        return OCRResult(
            roi_name=sample.roi_name,
            frame_idx=sample.frame_idx,
            t_sec=sample.t_sec,
            timecode=sample.timecode,
            all_text=best_text,
            spans=best_spans,
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
                        font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        tag = result.meta.get("backend", "DeepOCR")
        cv2.putText(vis, f"{tag}: {result.all_text}", (8, vis.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{tag}: {result.all_text}", (8, vis.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
        return vis

    # ---------- Internals ----------

    def _merge_text(self, spans: List[OCRSpan]) -> str:
        if self.cfg.sort_left_to_right:
            spans = sorted(spans, key=lambda s: s.bbox[0])
        raw = " ".join(s.text for s in spans).strip()
        m = self._keep_re.findall(raw)
        return "".join(m).strip()

    def _scale_span(self, sp: OCRSpan, k: float) -> OCRSpan:
        x, y, w, h = sp.bbox
        x = int(round(x * k)); y = int(round(y * k))
        w = int(round(w * k)); h = int(round(h * k))
        return OCRSpan(text=sp.text, conf=sp.conf, bbox=(x, y, w, h),
                       level=sp.level, line_num=sp.line_num, block_num=sp.block_num, par_num=sp.par_num)

    def _rescale(self, im: np.ndarray, s: float) -> np.ndarray:
        if abs(s - 1.0) < 1e-3:
            return im
        h, w = im.shape[:2]
        return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)

    def _build_candidates(self, bgr: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        # Optional pre-upscale
        if self.cfg.pre_upscale and self.cfg.pre_upscale != 1.0:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (int(w * self.cfg.pre_upscale), int(h * self.cfg.pre_upscale)),
                             interpolation=cv2.INTER_CUBIC)

        # Luminance / local contrast
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
        lab = cv2.merge([L, A, B])
        enh = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

        gray = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # Adaptive thresholds (emphasize white glyphs)
        block = max(3, self.cfg.adaptive_block | 1)  # odd
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block, self.cfg.adaptive_C
        )
        th_inv = cv2.bitwise_not(th)

        # Clean speckles
        if self.cfg.open_iters > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg.open_ksize)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=self.cfg.open_iters)
            th_inv = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, k, iterations=self.cfg.open_iters)

        # Return as 3-channel for EasyOCR convenience
        to_bgr = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        candidates = [enh, to_bgr(gray), to_bgr(th), to_bgr(th_inv)]
        meta = ["enh", "gray", "th_bin", "th_inv"]
        return candidates, meta

    def _easyocr_text(self, bgr_or_gray3: np.ndarray) -> List[OCRSpan]:
        # EasyOCR expects RGB or gray
        if len(bgr_or_gray3.shape) == 3 and bgr_or_gray3.shape[2] == 3:
            inp = cv2.cvtColor(bgr_or_gray3, cv2.COLOR_BGR2RGB)
        else:
            inp = bgr_or_gray3

        results = self._reader.readtext(
            inp,
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
            box_pts, text, conf = det
            text = (text or "").strip()
            if not text or conf < self.cfg.min_conf_keep:
                continue

            # Keep only allowed pattern to reduce HUD noise
            kept = "".join(self._keep_re.findall(text))
            kept = kept.strip()
            if not kept:
                continue

            xs = [int(p[0]) for p in box_pts]
            ys = [int(p[1]) for p in box_pts]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y

            spans.append(OCRSpan(
                text=kept,
                conf=float(conf * 100.0),   # align with your 0–100 scale
                bbox=(x, y, w, h),
                level=5
            ))

        if self.cfg.sort_left_to_right:
            spans.sort(key=lambda s: s.bbox[0])
        return spans
