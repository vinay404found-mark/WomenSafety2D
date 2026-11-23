"""
eval.py

Runs every clip in dataset/violent/ and dataset/non-violent/ through the
EXACT SAME ThreatPipeline used by main.py (no display, no real SMS -- always
dry-run here), and reports:
  - which behaviors fired on each clip, and at what timestamp
  - for non-violent clips: any trigger at all = FALSE POSITIVE
  - for violent clips: no trigger = FALSE NEGATIVE (missed detection)
  - an aggregate precision/recall-style summary at the end

Usage:
    python3 eval.py
Expects:
    dataset/violent/*.mp4
    dataset/non-violent/*.mp4
(matching the folder structure from your VS Code project)
"""

import os
import glob
import cv2
from ultralytics import YOLO
from pipeline_core import ThreatPipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
VIOLENT_DIR = os.path.join(DATASET_DIR, "violent")
NON_VIOLENT_DIR = os.path.join(DATASET_DIR, "non-violent")

# Skip frames for speed during bulk evaluation (1 = every frame, 2 = every other, etc.)
FRAME_STRIDE = 1


def run_clip(yolo_model, video_path):
    pipeline = ThreatPipeline(dry_run_sms=True, enable_gender=True, verbose=False)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"could not open {video_path}"

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_STRIDE == 0:
            pipeline.process_frame(frame, yolo_model, frame_index=frame_idx)
        frame_idx += 1
    cap.release()

    return pipeline.event_log, None


def format_events(event_log, clip_start_time=None):
    lines = []
    for e in event_log:
        lines.append(f"    - {e['behavior']}: {e['message']}")
    return lines


def main():
    print("🚀 Loading YOLO Engine for evaluation run...")
    yolo_model = YOLO("yolo11s-pose.pt" if os.path.exists("yolo11s-pose.pt") else "yolo11n-pose.pt")

    violent_clips = sorted(glob.glob(os.path.join(VIOLENT_DIR, "*.mp4")))
    non_violent_clips = sorted(glob.glob(os.path.join(NON_VIOLENT_DIR, "*.mp4")))

    if not violent_clips and not non_violent_clips:
        print(f"⚠️  No clips found under {DATASET_DIR}. Expected dataset/violent/*.mp4 "
              f"and dataset/non-violent/*.mp4")
        return

    results = {"violent": [], "non_violent": []}

    print(f"\n{'='*70}\nEvaluating {len(non_violent_clips)} NON-VIOLENT clips (any trigger = false positive)\n{'='*70}")
    for clip_path in non_violent_clips:
        name = os.path.basename(clip_path)
        event_log, err = run_clip(yolo_model, clip_path)
        if err:
            print(f"  {name}: ERROR -- {err}")
            continue
        triggered = len(event_log) > 0
        results["non_violent"].append({"name": name, "triggered": triggered, "events": event_log})
        status = "❌ FALSE POSITIVE" if triggered else "✅ correctly quiet"
        print(f"  {name}: {status} ({len(event_log)} event(s))")
        for line in format_events(event_log):
            print(line)

    print(f"\n{'='*70}\nEvaluating {len(violent_clips)} VIOLENT clips (no trigger = missed detection)\n{'='*70}")
    for clip_path in violent_clips:
        name = os.path.basename(clip_path)
        event_log, err = run_clip(yolo_model, clip_path)
        if err:
            print(f"  {name}: ERROR -- {err}")
            continue
        triggered = len(event_log) > 0
        results["violent"].append({"name": name, "triggered": triggered, "events": event_log})
        status = "✅ detected" if triggered else "❌ MISSED"
        print(f"  {name}: {status} ({len(event_log)} event(s))")
        for line in format_events(event_log):
            print(line)

    # --- summary ---
    nv_total = len(results["non_violent"])
    nv_false_positives = sum(1 for r in results["non_violent"] if r["triggered"])
    v_total = len(results["violent"])
    v_detected = sum(1 for r in results["violent"] if r["triggered"])

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    if nv_total:
        print(f"Non-violent clips: {nv_total} total, {nv_false_positives} false positive(s) "
              f"({100*nv_false_positives/nv_total:.0f}% false-positive rate)")
    if v_total:
        print(f"Violent clips:     {v_total} total, {v_detected} detected "
              f"({100*v_detected/v_total:.0f}% recall)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
