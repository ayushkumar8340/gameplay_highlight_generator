from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, List, Dict, Any
import cv2
import yaml
import numpy as np
import csv
from dataclasses import dataclass
from typing import Union



# ---------- Data models ----------

@dataclass(frozen=True)
class ROI:
    name: str
    x: int
    y: int
    w: int
    h: int

@dataclass(frozen=True)
class CropSample:
    frame_idx: int
    t_sec: float
    crop: np.ndarray

@dataclass(frozen=True)
class MultiCropSample:
    """One cropped sample for a specific ROI/class."""
    roi_name: str
    frame_idx: int
    t_sec: float
    timecode: str
    crop: np.ndarray  # BGR


@dataclass(frozen=True)
class FullFrameSample:
    """The full original frame at a specific time."""
    frame_idx: int
    t_sec: float
    timecode: str
    frame: np.ndarray  # BGR


# ---------- ROICropper ----------

class ROICropper:
    """
    Supports YAML with:
      version, created_utc, base_width/base_height, base_resolution: [W,H], fps_analysis, areas: [{name,x,y,w,h}, ...]
    Backward compatible with legacy: base_resolution:{width,height}, roi:{...}
    """

    def __init__(self, video_path: str, yaml_path: str, roi_name: Optional[str] = None):
        self.video_path = video_path
        self.yaml_path = yaml_path
        self.roi_name = roi_name

        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not os.path.isfile(self.yaml_path):
            raise FileNotFoundError(f"YAML not found: {self.yaml_path}")

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Raw frame geometry
        self.frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # YAML metadata and ROIs
        meta = self._load_yaml(self.yaml_path)
        self.base_w, self.base_h = meta["base_w"], meta["base_h"]
        self.fps_yaml = meta.get("fps_analysis")
        self.rois_all: List[ROI] = meta["rois"]

        # Select single ROI for legacy APIs
        self.roi = None
        if self.roi_name:
            self.roi = next((r for r in self.rois_all if r.name == self.roi_name), None)
            if self.roi is None:
                raise ValueError(f"ROI named '{self.roi_name}' not found.")
        else:
            self.roi = self.rois_all[0]

        # Pre-compute scaled ROIs in the raw frame space (no rotation logic)
        self.scaled_rois_all: List[ROI] = self._scale_rois(
            self.rois_all,
            (self.base_w, self.base_h),
            (self.frame_w, self.frame_h),
        )
        self.scaled_roi: ROI = next(r for r in self.scaled_rois_all if r.name == self.roi.name)

    def __del__(self):
        try:
            if hasattr(self, "_cap") and self._cap is not None:
                self._cap.release()
        except Exception:
            pass

    # ---------- YAML ----------
    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Base resolution
        base_w = data.get("base_width")
        base_h = data.get("base_height")

        if (base_w is None or base_h is None) and "base_resolution" in data:
            br = data["base_resolution"]
            if isinstance(br, (list, tuple)) and len(br) >= 2:
                base_w, base_h = int(br[0]), int(br[1])
            elif isinstance(br, dict):
                base_w, base_h = int(br["width"]), int(br["height"])

        if base_w is None or base_h is None:
            raise ValueError("YAML must include base_width/base_height or base_resolution.")

        base_w, base_h = int(base_w), int(base_h)
        if base_w <= 0 or base_h <= 0:
            raise ValueError("Base resolution must be positive.")

        # ROIs
        rois: List[ROI] = []
        if "areas" in data and isinstance(data["areas"], list) and data["areas"]:
            for a in data["areas"]:
                rois.append(ROI(
                    name=str(a.get("name", "roi")),
                    x=int(a["x"]), y=int(a["y"]), w=int(a["w"]), h=int(a["h"])
                ))
        elif "roi" in data:
            r = data["roi"]
            rois.append(ROI(
                name=str(r.get("name", "roi")),
                x=int(r["x"]), y=int(r["y"]), w=int(r["w"]), h=int(r["h"])
            ))
        else:
            raise ValueError("No ROIs found: expected 'areas: [...]' or 'roi: {...}'")

        out = {"base_w": base_w, "base_h": base_h, "rois": rois}
        if "fps_analysis" in data:
            out["fps_analysis"] = data["fps_analysis"]
        return out

    # ---------- Scaling ----------
    @staticmethod
    def _scale_rois(rois: List[ROI], base_size: Tuple[int, int], frame_size: Tuple[int, int]) -> List[ROI]:
        base_w, base_h = base_size
        frame_w, frame_h = frame_size
        sx = frame_w / float(base_w)
        sy = frame_h / float(base_h)

        out: List[ROI] = []
        for r in rois:
            x = int(round(r.x * sx))
            y = int(round(r.y * sy))
            w = int(round(r.w * sx))
            h = int(round(r.h * sy))
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            out.append(ROI(name=r.name, x=x, y=y, w=w, h=h))
        return out

    # ---------- Time helpers ----------
    def frame_to_seconds(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    @staticmethod
    def seconds_to_timecode(t: float) -> str:
        ms = int(round((t - int(t)) * 1000))
        s = int(t) % 60
        m = (int(t) // 60) % 60
        h = int(t) // 3600
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    # ---------- Single-ROI iteration (legacy API) ----------
    def iter_crops(
        self,
        every_n: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        with_meta: bool = False,
    ) -> Generator[np.ndarray | CropSample, None, None]:
        if every_n <= 0:
            raise ValueError("every_n must be >= 1")
        total = self.frame_count
        end = min(end_frame, total) if end_frame is not None else total
        start = max(0, start_frame)
        if start >= end:
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        x, y, w, h = self.scaled_roi.x, self.scaled_roi.y, self.scaled_roi.w, self.scaled_roi.h

        idx = start
        while idx < end:
            ret, frame = self._cap.read()
            if not ret:
                break

            if (idx - start) % every_n == 0:
                crop = frame[y:y+h, x:x+w].copy()
                if with_meta:
                    yield CropSample(frame_idx=idx, t_sec=self.frame_to_seconds(idx), crop=crop)
                else:
                    yield crop
            idx += 1

    # ---------- All-ROIs in one pass ----------
    def iter_all_crops(
        self,
        every_n: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Generator[List[MultiCropSample], None, None]:
        """
        Yields a list of MultiCropSample (one per ROI) for each sampled frame.
        Each element has: roi_name, frame_idx, t_sec, timecode, crop (BGR).
        """
        if every_n <= 0:
            raise ValueError("every_n must be >= 1")
        total = self.frame_count
        end = min(end_frame, total) if end_frame is not None else total
        start = max(0, start_frame)
        if start >= end:
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        scaled = self.scaled_rois_all

        idx = start
        while idx < end:
            ret, frame = self._cap.read()
            if not ret:
                break

            if (idx - start) % every_n == 0:
                t = self.frame_to_seconds(idx)
                tc = self.seconds_to_timecode(t)
                row: List[MultiCropSample] = []
                for r in scaled:
                    crop = frame[r.y:r.y+r.h, r.x:r.x+r.w].copy()
                    row.append(MultiCropSample(roi_name=r.name, frame_idx=idx, t_sec=t, timecode=tc, crop=crop))
                yield row
            idx += 1

    def collect_all_crops(
        self,
        every_n: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[MultiCropSample]:
        """
        Convenience: return a flat list of MultiCropSample for all ROIs.
        """
        out: List[MultiCropSample] = []
        for row in self.iter_all_crops(every_n=every_n, start_frame=start_frame, end_frame=end_frame):
            out.extend(row)
            if limit is not None and len(out) >= limit:
                return out[:limit]
        return out

    def save_all_crops(
        self,
        output_root: str,
        pattern: str = "{name}_f{frame_idx:06d}.png",
        every_n: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        class_subdirs: bool = True,
        write_csv: bool = True,
        csv_name: str = "all_crops_manifest.csv",
    ) -> int:
        """
        Save every ROI's crop for every sampled frame.
        Layout:
          output_root/<class>/<name>_f000123.png  (if class_subdirs=True)
        Returns total images written and writes a manifest CSV.
        """
        total_written = 0
        rows = []
        if class_subdirs:
            for r in self.scaled_rois_all:
                os.makedirs(os.path.join(output_root, r.name), exist_ok=True)
        else:
            os.makedirs(output_root, exist_ok=True)

        i_counters: Dict[str, int] = {r.name: 0 for r in self.scaled_rois_all}

        for batch in self.iter_all_crops(every_n=every_n, start_frame=start_frame, end_frame=end_frame):
            for sample in batch:
                i = i_counters[sample.roi_name]
                filename = pattern.format(name=sample.roi_name, frame_idx=sample.frame_idx, i=i)
                out_dir = os.path.join(output_root, sample.roi_name) if class_subdirs else output_root
                path = os.path.join(out_dir, filename)
                cv2.imwrite(path, sample.crop)
                h, w = sample.crop.shape[:2]
                rows.append({
                    "filename": os.path.relpath(path, output_root),
                    "class": sample.roi_name,
                    "frame_idx": sample.frame_idx,
                    "t_sec": f"{sample.t_sec:.6f}",
                    "timecode": sample.timecode,
                    "width": w, "height": h
                })
                total_written += 1
                i_counters[sample.roi_name] = i + 1

        if write_csv and total_written:
            os.makedirs(output_root, exist_ok=True)
            with open(os.path.join(output_root, csv_name), "w", newline="") as f:
                wcsv = csv.DictWriter(f, fieldnames=["filename","class","frame_idx","t_sec","timecode","width","height"])
                wcsv.writeheader()
                wcsv.writerows(rows)

        return total_written

        # ---------- Time parsing ----------
    @staticmethod
    def timecode_to_seconds(tc: str) -> float:
        """
        Parse 'HH:MM:SS.mmm' into seconds (float).
        Accepts 'MM:SS.mmm' or 'SS.mmm' as well.
        """
        tc = tc.strip()
        if not tc:
            raise ValueError("Empty timecode.")
        parts = tc.split(":")
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = "0", parts[0], parts[1]
        elif len(parts) == 1:
            h, m, s = "0", "0", parts[0]
        else:
            raise ValueError(f"Invalid timecode format: {tc}")

        # seconds may contain milliseconds 'SS.mmm'
        if "." in s:
            sec_str, ms_str = s.split(".", 1)
            sec = int(sec_str)
            ms = int((ms_str + "000")[:3])  # pad/truncate to 3 digits
        else:
            sec = int(s)
            ms = 0

        hours = int(h)
        mins = int(m)
        total = hours * 3600 + mins * 60 + sec + ms / 1000.0
        return float(total)

    # ---------- Full-frame getters ----------
    def get_original_frame(self, time: Union[str, float], nearest: bool = True) -> FullFrameSample:
        """
        Return the original full frame at a given time.

        Args:
            time: Either a timecode string like 'HH:MM:SS.mmm' (same format as ROI outputs)
                  or a float seconds value.
            nearest: If True, rounds to nearest frame. If False, floors to earlier frame.

        Returns:
            FullFrameSample with frame (BGR), frame_idx, t_sec, and timecode.
        """
        # Convert input to seconds
        if isinstance(time, str):
            t_sec = self.timecode_to_seconds(time)
        elif isinstance(time, (int, float)):
            t_sec = float(time)
        else:
            raise TypeError("time must be a timecode string or a float (seconds).")

        if t_sec < 0:
            t_sec = 0.0

        # Map to frame index
        raw_idx = t_sec * self.fps
        frame_idx = int(round(raw_idx)) if nearest else int(np.floor(raw_idx))
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))

        # Seek and read
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame at index {frame_idx}")

        # Compute precise timestamp for the frame actually read
        t_exact = self.frame_to_seconds(frame_idx)
        tc_exact = self.seconds_to_timecode(t_exact)

        return FullFrameSample(
            frame_idx=frame_idx,
            t_sec=t_exact,
            timecode=tc_exact,
            frame=frame.copy(),
        )

    def get_original_frame_by_index(self, frame_idx: int) -> FullFrameSample:
        """
        Convenience: fetch by absolute frame index.
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"frame_idx out of range [0, {self.frame_count-1}]")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame at index {frame_idx}")
        t_exact = self.frame_to_seconds(frame_idx)
        tc_exact = self.seconds_to_timecode(t_exact)
        return FullFrameSample(
            frame_idx=frame_idx,
            t_sec=t_exact,
            timecode=tc_exact,
            frame=frame.copy(),
        )

