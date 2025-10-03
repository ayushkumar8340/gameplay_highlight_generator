#!/usr/bin/env python3
import argparse
from pathlib import Path

from roi_helper import ROIAnnotator


def main():
    parser = argparse.ArgumentParser(
        description="Helper: annotate named ROIs on the first frame of a video and save to YAML."
    )
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--out",   required=True, help="Path to output YAML (e.g., rois/rois.yaml).")
    parser.add_argument("--version", default="1.0.0", help="ROI spec version to embed (default: 1.0.0).")
    parser.add_argument("--fps-analysis", type=int, default=15, help="Analysis FPS to embed (default: 15).")
    args = parser.parse_args()

    annot = ROIAnnotator(
        video_path=args.video,
        version=args.version,
        fps_analysis=args.fps_analysis
    )

    print(f"[info] base resolution: {annot.base_resolution[0]}x{annot.base_resolution[1]}")
    saved = annot.run()
    if saved:
        payload = annot.save_yaml(Path(args.out))
        print(f"[ok] saved {len(payload['areas'])} ROI(s) → {args.out}")
        print("---")
        print(f"version: {payload['version']}")
        print(f"base_resolution: {payload['base_resolution'][0]}x{payload['base_resolution'][1]}")
        print(f"fps_analysis: {payload['fps_analysis']}")
        for a in payload["areas"]:
            print(f" - {a['name']}: x={a['x']} y={a['y']} w={a['w']} h={a['h']}")
    else:
        print("[info] nothing saved.")

if __name__ == "__main__":
    main()
