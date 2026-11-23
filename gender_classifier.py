
import time
from collections import deque, Counter
from PIL import Image
import cv2

from transformers import pipeline as hf_pipeline

# --- Reclassification schedule ---
FAST_RECLASSIFY_EVERY_N_FRAMES = 5    # while not yet confirmed -- converge to an answer quickly
SLOW_RECLASSIFY_EVERY_N_FRAMES = 15   # once provisionally confirmed but not yet locked

# --- Vote / confidence gating ---
VOTE_WINDOW = 9                 # how many past predictions to majority-vote over
MIN_MODEL_CONFIDENCE = 0.65     # per-frame model confidence floor -- below this, discard the vote
MIN_VOTES_BEFORE_COMMIT = 4     # need at least this many accepted votes before returning male/female at all
MIN_AGREEMENT_RATIO = 0.70      # majority must hold at least this fraction of votes, else -> unknown

# --- Hard lock ---
LOCK_MIN_VOTES = 5              # once this many accepted votes exist...
LOCK_MIN_AGREEMENT = 0.80       # ...and at least this fraction agree, LOCK the label permanently

# --- Crop quality gating ---
MIN_CROP_W, MIN_CROP_H = 50, 80
MAX_ASPECT_RATIO = 3.2          # reject absurdly thin/wide crops (tracking glitches, partial bodies)

# --- Label normalization ---
# Maps whatever the underlying model calls things to a canonical label.
# Anything not in here is treated as UNRECOGNIZED (discarded), never guessed.
LABEL_MAP = {
    "male": "male", "man": "male", "men": "male", "m": "male", "boy": "male",
    "female": "female", "woman": "female", "women": "female", "f": "female", "girl": "female",
}


def _normalize_label(raw_label):
    if raw_label is None:
        return None
    key = str(raw_label).strip().lower()
    return LABEL_MAP.get(key, None)  # None if unrecognized -- never silently guess


class GenderClassifier:
    def __init__(self, model_name="NTQAI/pedestrian_gender_recognition"):
        print("🚻 Loading gender classification model...")
        self.model = hf_pipeline(task="image-classification", model=model_name)

        # id -> {"frame_count": int, "history": deque[label], "label": str, "conf": float, "confirmed": bool}
        self.state = {}

    def _crop_is_usable(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        h, w = crop_bgr.shape[:2]
        if w < MIN_CROP_W or h < MIN_CROP_H:
            return False
        aspect = h / max(w, 1)
        if aspect > MAX_ASPECT_RATIO or aspect < (1.0 / MAX_ASPECT_RATIO):
            return False  # malformed crop -- not a plausible standing/sitting person silhouette
        return True

    def _classify_crop(self, crop_bgr):
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
        try:
            predictions = self.model(pil_image)
            if not predictions:
                return None, 0.0
            best = predictions[0]
            label = _normalize_label(best.get("label"))
            conf = float(best.get("score", 0.0))
            return label, conf
        except Exception as e:
            print(f"⚠️ Gender classification error: {e}")
            return None, 0.0

    def get_gender(self, person_id, crop_bgr):
        """
        Call this once per tracked person per frame. Internally throttles the
        actual model call and returns a smoothed (label, confidence) pair.
        Returns ("unknown", 0.0) whenever the evidence doesn't clearly support
        a label yet. Once LOCKED, always returns the same frozen label/1.0
        with no further model calls -- see module docstring.
        """
        if person_id not in self.state:
            self.state[person_id] = {
                "frame_count": 0,
                "history": deque(maxlen=VOTE_WINDOW),
                "label": "unknown",
                "conf": 0.0,
                "confirmed": False,
                "locked": False,
            }

        st = self.state[person_id]

        # --- Already locked: skip everything, return the frozen answer ---
        if st["locked"]:
            return st["label"], 1.0

        st["frame_count"] += 1

        reclassify_interval = SLOW_RECLASSIFY_EVERY_N_FRAMES if st["confirmed"] else FAST_RECLASSIFY_EVERY_N_FRAMES
        should_classify = (
            self._crop_is_usable(crop_bgr)
            and (st["frame_count"] % reclassify_interval == 1)
        )

        if should_classify:
            label, conf = self._classify_crop(crop_bgr)
            if label is not None and conf >= MIN_MODEL_CONFIDENCE:
                st["history"].append(label)
            # if label is None (unrecognized) or conf too low, the vote is simply
            # discarded -- it does NOT get counted as evidence for either label

        if len(st["history"]) >= MIN_VOTES_BEFORE_COMMIT:
            vote_counts = Counter(st["history"])
            majority_label, count = vote_counts.most_common(1)[0]
            agreement = count / len(st["history"])

            # --- Check for LOCK first: enough votes, strong enough agreement -> freeze forever ---
            if len(st["history"]) >= LOCK_MIN_VOTES and agreement >= LOCK_MIN_AGREEMENT:
                st["label"] = majority_label
                st["conf"] = 1.0
                st["confirmed"] = True
                st["locked"] = True
                return st["label"], st["conf"]

            if agreement >= MIN_AGREEMENT_RATIO:
                st["label"] = majority_label
                st["conf"] = agreement
                st["confirmed"] = True
            else:
                # votes are too split to be confident -- genuinely ambiguous, keep collecting
                st["label"] = "unknown"
                st["conf"] = agreement
                st["confirmed"] = False
        else:
            st["label"] = "unknown"
            st["conf"] = 0.0
            st["confirmed"] = False

        return st["label"], st["conf"]

    def cleanup(self, active_ids):
        stale = [pid for pid in self.state if pid not in active_ids]
        for pid in stale:
            del self.state[pid]