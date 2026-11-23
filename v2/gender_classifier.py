"""
gender_classifier.py

Wraps the HF pedestrian-gender pipeline with:
  - Per-track-ID caching (classify once every N frames per person, not every frame)
  - A rolling majority vote so one bad crop/frame doesn't flip the label
  - Confidence-based fallback to "unknown" so low-confidence guesses don't
    poison your alert messages

This keeps the video loop fast -- the classifier is one of the more expensive
parts of the pipeline, so we do NOT want to call it on every person, every frame.
"""

import time
from collections import deque, Counter
from PIL import Image
import cv2

from transformers import pipeline as hf_pipeline

RECLASSIFY_EVERY_N_FRAMES = 10   # only re-run the model this often per person
VOTE_WINDOW = 7                  # how many past predictions to majority-vote over
MIN_CONFIDENCE = 0.55            # below this, treat as "unknown" rather than guessing
MIN_CROP_W, MIN_CROP_H = 50, 80  # ignore detections too small to classify reliably


class GenderClassifier:
    def __init__(self, model_name="NTQAI/pedestrian_gender_recognition"):
        print("🚻 Loading gender classification model...")
        self.model = hf_pipeline(task="image-classification", model=model_name)

        # id -> {"frame_count": int, "history": deque[label], "label": str, "conf": float}
        self.state = {}

    def _classify_crop(self, crop_bgr):
        if crop_bgr.size == 0:
            return None, 0.0
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        try:
            predictions = self.model(pil_image)
            if not predictions:
                return None, 0.0
            best = predictions[0]
            return best["label"].lower(), float(best["score"])
        except Exception as e:
            print(f"⚠️ Gender classification error: {e}")
            return None, 0.0

    def get_gender(self, person_id, crop_bgr):
        """
        Call this once per tracked person per frame. Internally throttles the
        actual model call and returns a smoothed (label, confidence) pair.
        Returns ("unknown", 0.0) if not enough info yet or confidence too low.
        """
        if person_id not in self.state:
            self.state[person_id] = {
                "frame_count": 0,
                "history": deque(maxlen=VOTE_WINDOW),
                "label": "unknown",
                "conf": 0.0,
            }

        st = self.state[person_id]
        st["frame_count"] += 1

        h, w = crop_bgr.shape[:2]
        should_classify = (
            (w >= MIN_CROP_W and h >= MIN_CROP_H)
            and (st["frame_count"] % RECLASSIFY_EVERY_N_FRAMES == 1)
        )

        if should_classify:
            label, conf = self._classify_crop(crop_bgr)
            if label is not None and conf >= MIN_CONFIDENCE:
                st["history"].append(label)

        if st["history"]:
            vote_counts = Counter(st["history"])
            majority_label, count = vote_counts.most_common(1)[0]
            st["label"] = majority_label
            st["conf"] = count / len(st["history"])  # agreement ratio as displayed "confidence"
        else:
            st["label"] = "unknown"
            st["conf"] = 0.0

        return st["label"], st["conf"]

    def cleanup(self, active_ids):
        stale = [pid for pid in self.state if pid not in active_ids]
        for pid in stale:
            del self.state[pid]
