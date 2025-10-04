# roi_main.py
"""
Main script to pick a frame from a video and annotate ROIs.
Uses helper classes from helper/pick_frame_and_annotate.py
"""

import argparse
from roi_helper import VideoFramePicker, ROIAnnotator, save_yaml
import cv2

def main():
    parser = argparse.ArgumentParser(description="Pick a frame from video and annotate ROIs.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--out-yaml", default="rois.yaml", help="Output YAML file for ROIs")
    parser.add_argument("--save-frame", default=None, help="Optional path to save picked frame (PNG/JPG)")
    args = parser.parse_args()

    # Step 1: pick a frame interactively
    picker = VideoFramePicker(args.video)
    picked = picker.pick()
    if picked is None:
        print("❌ No frame selected. Exiting.")
        return

    print(f"✅ Picked frame {picked.frame_idx} at {picked.t_sec:.3f}s ({picked.timecode}).")

    # Save picked frame if requested
    if args.save_frame:
        cv2.imwrite(args.save_frame, picked.frame_bgr)
        print(f"💾 Saved picked frame to: {args.save_frame}")

    # Step 2: Annotate ROIs
    annotator = ROIAnnotator()
    areas = annotator.annotate(picked.frame_bgr)
    if not areas:
        print("❌ No ROIs annotated. Exiting.")
        return

    # Step 3: Save YAML
    save_yaml(args.out_yaml, picked.frame_bgr.shape, areas)
    print(f"🎉 Finished! ROIs saved to {args.out_yaml}")

if __name__ == "__main__":
    main()
