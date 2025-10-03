# helpers/roi_helper.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set

import cv2
import yaml
import numpy as np


@dataclass
class _ROIBox:
    name: str
    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.update({k: int(d[k]) for k in ("x", "y", "w", "h")})
        return d


class ROIAnnotator:
    """
    Lightweight helper to annotate & name Regions of Interest (ROIs) on the FIRST frame of a video.

    Controls:
      n : new ROI (drag to select)
      z : undo last
      s : save and exit
      q : quit without saving
      h : print help
    """

    def __init__(self, video_path: str | Path, version: str = "1.0.0", fps_analysis: int = 15):
        self.video_path = str(video_path)
        self.version = str(version)
        self.fps_analysis = int(fps_analysis)

        self._frame = self._read_first_frame(self.video_path)
        self._H, self._W = self._frame.shape[:2]
        self._areas: List[_ROIBox] = []
        self._window = "ROI Annotator"

    # -------------------- Public API --------------------

    @property
    def base_resolution(self) -> Tuple[int, int]:
        """(width, height) at which ROIs are defined."""
        return (self._W, self._H)

    def run(self) -> bool:
        """Open the interactive window. Returns True if user chose to save; False if quit."""
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        print(self._help_console())
        while True:
            vis = self._render()
            cv2.imshow(self._window, vis)
            key = cv2.waitKey(50) & 0xFF

            if key == ord('n'):
                self._add_roi_interactive()

            elif key == ord('z'):
                if self._areas:
                    removed = self._areas.pop()
                    print(f"[info] removed ROI '{removed.name}'.")
                else:
                    print("[info] no ROIs to remove.")

            elif key == ord('s'):
                if not self._areas:
                    print("[warn] no ROIs defined; nothing to save.")
                    continue
                cv2.destroyWindow(self._window)
                return True

            elif key == ord('h'):
                print(self._help_console())

            elif key == ord('q') or key == 27:
                print("[info] quit without saving.")
                cv2.destroyWindow(self._window)
                return False

    def save_yaml(self, out_path: str | Path) -> Dict:
        """
        Save the annotated ROIs to YAML.

        Includes explicit base_width and base_height so downstream code
        can handle resolution scaling elsewhere.
        """
        if not self._areas:
            raise RuntimeError("No ROIs to save. Run .run() and add at least one ROI.")

        payload = {
            "version": self.version,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            # Explicit fields for downstream scaling logic:
            "base_width": int(self._W),
            "base_height": int(self._H),
            # Kept for convenience/back-compat:
            "base_resolution": [int(self._W), int(self._H)],  # [width, height]
            "fps_analysis": self.fps_analysis,
            "areas": [a.to_dict() for a in self._areas],
        }

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

        return payload

    # -------------------- Internals --------------------

    @staticmethod
    def _read_first_frame(video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Could not read the first frame from the video.")
        return frame

    def _render(self) -> np.ndarray:
        vis = self._frame.copy()
        for r in self._areas:
            self._draw_box(vis, r)
        footer = "[n] new  [z] undo  [s] save  [q] quit  [h] help"
        cv2.putText(vis, f"{footer}    ROIs: {len(self._areas)}",
                    (10, self._H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    @staticmethod
    def _draw_box(img: np.ndarray, r: _ROIBox, color=(0, 255, 0)) -> None:
        x, y, w, h = r.x, r.y, r.w, r.h
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{r.name} ({w}x{h})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_top = max(0, y - th - 8)
        cv2.rectangle(img, (x, y_top), (x + tw + 6, y), (0, 0, 0), -1)
        cv2.putText(img, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def _add_roi_interactive(self) -> None:
        title = "Draw ROI (ENTER/SPACE=confirm, c=cancel)"
        r = cv2.selectROI(title, self._frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        x, y, w, h = map(int, r)
        if w <= 0 or h <= 0:
            print("[info] ROI selection canceled.")
            return
        try:
            name = input("Name this ROI (e.g., health_bar, kill_feed, weapon_icon): ").strip()
        except EOFError:
            name = ""
        if not name:
            name = "roi"
        name = self._unique_name(name, {a.name for a in self._areas})
        self._areas.append(_ROIBox(name=name, x=x, y=y, w=w, h=h))
        print(f"[info] added ROI '{name}': x={x} y={y} w={w} h={h}")

    @staticmethod
    def _unique_name(desired: str, existing: Set[str]) -> str:
        if desired not in existing:
            return desired
        i = 2
        while f"{desired}_{i}" in existing:
            i += 1
        new_name = f"{desired}_{i}"
        print(f"[info] name '{desired}' exists. Using '{new_name}'.")
        return new_name

    @staticmethod
    def _help_console() -> str:
        return (
            "\nControls:\n"
            "  n : create a new ROI (drag to select)\n"
            "  z : undo last ROI\n"
            "  s : save to YAML and exit\n"
            "  q : quit without saving\n"
            "  h : print this help\n"
        )
