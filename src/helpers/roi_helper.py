# pick_frame_and_annotate.py
from __future__ import annotations
import cv2
import yaml
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass(frozen=True)
class PickedFrame:
    frame_idx: int
    t_sec: float
    timecode: str
    frame_bgr: 'cv2.Mat'

def seconds_to_timecode(t: float) -> str:
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

class VideoFramePicker:
    """
    Play a video, pause anywhere, and press 's' to pick the current frame.
    Controls:
      Space = pause/resume
      S = select current frame
      A/D = step -1/+1 frame (paused)
      J/L = step -10/+10 frames (paused)
      ,/. = step -1s/+1s (paused)
      Q or ESC = quit
    """
    def __init__(self, video_path: str, window_name: str = "Frame Picker"):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(video_path)
        self.video_path = video_path
        self.window_name = window_name
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open: {video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.paused = False

    def _goto(self, idx: int) -> Tuple[Optional['cv2.Mat'], int]:
        idx = max(0, min(idx, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None, idx
        return frame, idx

    def _overlay_hud(self, frame, idx: int):
        t = idx / self.fps
        tc = seconds_to_timecode(t)
        hud = f"{idx+1}/{self.total} | {t:0.3f}s | {tc} | {'PAUSED' if self.paused else 'PLAY'}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(frame, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Space: pause/resume | S: select | A/D: -/+1f | J/L: -/+10f | ,/. : -/+1s | Q/ESC: quit",
                    (10, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    def pick(self) -> Optional[PickedFrame]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        idx = 0
        frame, idx = self._goto(idx)
        if frame is None:
            return None

        while True:
            vis = frame.copy()
            self._overlay_hud(vis, idx)
            cv2.imshow(self.window_name, vis)

            # playback delay: ~1/fps when playing, short when paused
            delay = 1 if not self.paused else 50
            key = cv2.waitKey(delay) & 0xFF

            if key == 255:  # no key
                if not self.paused:
                    # advance by one frame while playing
                    nxt = idx + 1
                    nf, idx = self._goto(nxt)
                    if nf is None:
                        break
                    frame = nf
                continue

            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == 32:  # space
                self.paused = not self.paused
            elif key in (ord('s'), ord('S')):
                t = idx / self.fps
                tc = seconds_to_timecode(t)
                picked = PickedFrame(frame_idx=idx, t_sec=t, timecode=tc, frame_bgr=frame.copy())
                cv2.destroyWindow(self.window_name)
                return picked
            # stepping only when paused (more predictable)
            elif self.paused and key in (ord('a'), ord('A')):
                nf, idx = self._goto(idx - 1)
                if nf is not None: frame = nf
            elif self.paused and key in (ord('d'), ord('D')):
                nf, idx = self._goto(idx + 1)
                if nf is not None: frame = nf
            elif self.paused and key in (ord('j'), ord('J')):
                nf, idx = self._goto(idx - 10)
                if nf is not None: frame = nf
            elif self.paused and key in (ord('l'), ord('L')):
                nf, idx = self._goto(idx + 10)
                if nf is not None: frame = nf
            elif self.paused and key == ord(','):
                step = int(round(self.fps))
                nf, idx = self._goto(idx - step)
                if nf is not None: frame = nf
            elif self.paused and key == ord('.'):
                step = int(round(self.fps))
                nf, idx = self._goto(idx + step)
                if nf is not None: frame = nf

        cv2.destroyWindow(self.window_name)
        return None

class ROIAnnotator:
    """
    After you pick a frame, call annotate(frame) to draw one or more ROIs.
    Uses cv2.selectROIs for quick multi-rectangle selection, then asks for names.
    """
    def __init__(self, window_name: str = "Annotate ROIs"):
        self.window_name = window_name

    def annotate(self, frame_bgr) -> List[Tuple[str, int, int, int, int]]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, frame_bgr)
        cv2.waitKey(1)

        # OpenCV returns Nx4 array of [x, y, w, h]
        rects = cv2.selectROIs(self.window_name, frame_bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(self.window_name)

        rects = rects if rects is not None else []
        named: List[Tuple[str, int, int, int, int]] = []
        if len(rects) == 0:
            print("No ROIs drawn.")
            return named

        print("\nEnter names for each ROI (press Enter to accept the suggested default):")
        for i, (x, y, w, h) in enumerate(rects):
            default_name = f"roi_{i+1}"
            try:
                name = input(f"Name for ROI #{i+1} [default: {default_name}]: ").strip()
            except EOFError:
                name = default_name
            if not name:
                name = default_name
            named.append((name, int(x), int(y), int(w), int(h)))
        return named

def save_yaml(yaml_path: str, frame_shape: Tuple[int, int, int], areas: List[Tuple[str, int, int, int, int]]):
    h, w = frame_shape[:2]
    data = {
        "version": "1.0.0",
        "base_width": int(w),
        "base_height": int(h),
        "areas": [{"name": n, "x": x, "y": y, "w": w0, "h": h0} for (n, x, y, w0, h0) in areas],
    }
    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Wrote YAML: {yaml_path}")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Pick a frame from video and annotate ROIs.")
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--out-yaml", default="rois_from_picked_frame.yaml", help="Where to save the ROI YAML")
    ap.add_argument("--save-picked", default=None, help="Optional path to save the picked frame (PNG/JPG)")
    args = ap.parse_args()

    picker = VideoFramePicker(args.video)
    picked = picker.pick()
    if picked is None:
        print("No frame selected. Exiting.")
        return

    print(f"Picked frame {picked.frame_idx} at {picked.t_sec:.3f}s ({picked.timecode}).")
    if args.save_picked:
        cv2.imwrite(args.save_picked, picked.frame_bgr)
        print(f"Saved picked frame: {args.save_picked}")

    annot = ROIAnnotator()
    areas = annot.annotate(picked.frame_bgr)

    if areas:
        save_yaml(args.out_yaml, picked.frame_bgr.shape, areas)
    else:
        print("No ROIs to save. Exiting.")

if __name__ == "__main__":
    main()
