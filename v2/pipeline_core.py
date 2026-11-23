"""
pipeline_core.py

The actual detection pipeline, with no cv2.imshow / no SMS side effects baked
in directly -- both main.py (live, with display + real alerts) and eval.py
(headless, for testing against your labeled dataset) call the SAME
ThreatPipeline.process_frame() so we're never debugging one thing while
running another.
"""

import time
import numpy as np

from keypoint_utils import (
    PersonHistory, get_head_center, get_torso_center, any_wrist_near_body,
    get_pair_scale, keypoint_quality,
)
from duration_tracker import DurationTracker
from alert_manager import AlertManager
from gender_classifier import GenderClassifier
import risk_engine

# --- Sustained-duration requirements (seconds) ---
HARASSMENT_PATTERN_DURATION = 3.5   # fused multi-signal condition must hold this long
VIOLENT_ACTION_DURATION = 0.4       # punch/kick -- short, just enough to reject 1-frame noise
FALL_DURATION = 0.3
MALE_HAND_ON_FEMALE_DURATION = 3.0
MALE_HAND_ON_FEMALE_CONTACT_SCORE = 0.5  # contact_intensity threshold (0..1) for this rule

# --- Crowd/occlusion reliability gate ---
# In dense crowds, YOLO can only see fragments of partially-occluded people,
# and the tracker (ByteTrack) can briefly swap IDs between two different real
# people as they cross paths -- both produce noise that looks like violent
# motion or contact. Refuse to evaluate anyone the system can't actually see
# reliably, rather than trying to threshold-tune around the noise.
MIN_KEYPOINT_QUALITY = 0.55      # fraction of 17 keypoints that must be confidently detected
MIN_TRACK_AGE_SECONDS = 0.4      # ID must have been stably tracked at least this long
MIN_BBOX_AREA_FRACTION = 0.015   # bbox must cover at least this fraction of the frame area
                                  # (filters out tiny, far-away crowd figures where body-scale
                                  # normalization becomes unreliable)

# --- Alert/cooldown tuning ---
PAIR_BEHAVIOR_COOLDOWN_SECONDS = 45.0
GLOBAL_SMS_COOLDOWN_SECONDS = 30.0

RISK_FLOOR = 0.10
RISK_DECAY_RATE = 0.94
RISK_NORMAL_CUTOFF = 0.40


class ThreatPipeline:
    def __init__(self, dry_run_sms=True, enable_gender=True, verbose=True):
        self.history = PersonHistory(maxlen_seconds=3.0)
        self.tracker = DurationTracker(leak_rate_per_sec=0.6, cooldown_seconds=PAIR_BEHAVIOR_COOLDOWN_SECONDS)
        self.alerts = AlertManager(global_cooldown_seconds=GLOBAL_SMS_COOLDOWN_SECONDS, dry_run=dry_run_sms)
        self.enable_gender = enable_gender
        self.gender_clf = GenderClassifier() if enable_gender else None
        self.verbose = verbose

        self.smoothed_risk = RISK_FLOOR
        self.last_active_reason = "Normal Environment"

        # collected trigger log -- useful for both live HUD and eval.py reporting
        self.event_log = []  # list of dicts: {time, behavior, pair, message}

    def process_frame(self, frame, yolo_model, frame_index=None, frame_timestamp=None):
        """
        Runs one frame through YOLO pose + tracking, evaluates risk_engine per
        pair, updates duration tracker, fires alerts, updates the continuous
        risk meter. Returns a dict with everything a caller (live display or
        eval harness) needs -- it does NOT draw anything or touch cv2 display.
        """
        frame_h, frame_w = frame.shape[:2]
        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        current_ids = []
        people = []  # (id, kp, head_xy, gender_label, gender_conf, bbox)
        current_frame_risk = RISK_FLOOR
        current_frame_reason = "Normal Environment"
        frame_events = []

        if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
            box_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints_list = results[0].keypoints.data.cpu().numpy()
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

            for idx, pid in enumerate(box_ids):
                if idx >= len(keypoints_list):
                    continue
                kp = keypoints_list[idx]
                head_xy = get_head_center(kp)
                torso_xy = get_torso_center(kp)
                self.history.update(pid, head_xy, torso_xy)

                x1, y1, x2, y2 = map(int, boxes_xyxy[idx])
                gender_label, gender_conf = "unknown", 0.0
                if self.enable_gender:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    gender_label, gender_conf = self.gender_clf.get_gender(pid, crop)

                people.append((pid, kp, head_xy, gender_label, gender_conf, (x1, y1, x2, y2)))
                current_ids.append(pid)

            # --- Reliability gate: only evaluate people we can actually see well ---
            # (still keep the full `people` list for gender-label display -- this
            # filtering only affects which people are considered for RISK checks)
            frame_area = float(frame_w * frame_h)
            eligible_people = []
            for (pid, kp, head_xy, gender_label, gender_conf, (x1, y1, x2, y2)) in people:
                quality = keypoint_quality(kp)
                age = self.history.track_age(pid)
                bbox_area_frac = ((x2 - x1) * (y2 - y1)) / frame_area if frame_area > 0 else 0.0
                if quality >= MIN_KEYPOINT_QUALITY and age >= MIN_TRACK_AGE_SECONDS \
                   and bbox_area_frac >= MIN_BBOX_AREA_FRACTION and head_xy is not None:
                    eligible_people.append((pid, kp, head_xy, gender_label, gender_conf, (x1, y1, x2, y2)))

            # --- standalone per-person: fall detection (reliability-gated) ---
            for (pid, kp, head_xy, gender_label, gender_conf, bbox) in eligible_people:
                fall = risk_engine.ground_fall_detected(kp)
                status = self.tracker.update(pid, pid, "fall_detected", fall, FALL_DURATION)
                if fall:
                    current_frame_risk = max(current_frame_risk, 0.9 * status["progress"] + 0.05)
                    current_frame_reason = "Possible Fall / Pushed Down"
                if status["trigger"]:
                    msg = f"Person appears to have fallen/been pushed down (ID {int(pid)})"
                    self._fire_alert("Fall/Ground", "Fall Detected", msg, frame_events)

            # --- pairwise: fused harassment pattern + violent action + gender rule (reliability-gated) ---
            for i in range(len(eligible_people)):
                for j in range(i + 1, len(eligible_people)):
                    id1, p1, head1, gender1, gconf1, _ = eligible_people[i]
                    id2, p2, head2, gender2, gconf2, _ = eligible_people[j]
                    if head1 is None or head2 is None:
                        continue

                    result = risk_engine.evaluate_pair(p1, p2, id1, id2, self.history)

                    gender_tag = f"[{gender1} #{int(id1)} & {gender2} #{int(id2)}]"

                    # continuous risk meter feed -- fused_score always contributes
                    if result["fused_score"] > current_frame_risk:
                        current_frame_risk = result["fused_score"]
                        current_frame_reason = f"Elevated interaction risk {gender_tag}"

                    # --- fused harassment pattern (multi-signal, sustained) ---
                    hp_status = self.tracker.update(id1, id2, "harassment_pattern",
                                                     result["harassment_pattern"], HARASSMENT_PATTERN_DURATION)
                    if hp_status["trigger"]:
                        msg = (f"Sustained forced-interaction pattern {gender_tag} "
                               f"(contact={result['signals']['contact']:.2f}, "
                               f"motion={result['signals']['motion']:.2f}, "
                               f"escape={result['signals']['escape']:.2f})")
                        self._fire_alert("Harassment Pattern", "Harassment Pattern", msg, frame_events)

                    # --- standalone violent action (punch/kick) ---
                    va_status = self.tracker.update(id1, id2, "violent_action",
                                                     result["violent_action"], VIOLENT_ACTION_DURATION)
                    if result["violent_action"]:
                        current_frame_risk = max(current_frame_risk, 0.85)
                        current_frame_reason = f"{result['violent_label'].title()} detected {gender_tag}"
                    if va_status["trigger"]:
                        msg = f"{result['violent_label'].title()} detected, sustained {gender_tag}"
                        self._fire_alert("Violent Action", result["violent_label"].title(), msg, frame_events)

                    # --- gender-aware rule: male hand on female, sustained ---
                    if self.enable_gender and gender1 == "male" and gender2 == "female":
                        mhf_condition = result["signals"]["contact"] >= MALE_HAND_ON_FEMALE_CONTACT_SCORE
                    elif self.enable_gender and gender2 == "male" and gender1 == "female":
                        mhf_condition = result["signals"]["contact"] >= MALE_HAND_ON_FEMALE_CONTACT_SCORE
                    else:
                        mhf_condition = False
                    mhf_status = self.tracker.update(id1, id2, "male_hand_on_female",
                                                      mhf_condition, MALE_HAND_ON_FEMALE_DURATION)
                    if mhf_status["trigger"]:
                        msg = f"Sustained male-hand-on-female contact {gender_tag}"
                        self._fire_alert("Gender-Aware Contact", "Male Hand On Female", msg, frame_events)

        self.history.cleanup(current_ids)
        if self.enable_gender:
            self.gender_clf.cleanup(current_ids)

        # --- continuous risk meter update ---
        if current_frame_risk >= self.smoothed_risk:
            self.smoothed_risk = current_frame_risk
            self.last_active_reason = current_frame_reason
        else:
            self.smoothed_risk = max(RISK_FLOOR, self.smoothed_risk * RISK_DECAY_RATE)
            if self.smoothed_risk < RISK_NORMAL_CUTOFF:
                self.last_active_reason = "Normal Environment"

        self.event_log.extend(frame_events)

        return {
            "people": people,
            "smoothed_risk": self.smoothed_risk,
            "reason": self.last_active_reason,
            "events_this_frame": frame_events,
            "yolo_plotted_frame": results[0].plot() if people else frame,
        }

    def _fire_alert(self, class_name, behavior_title, msg, frame_events):
        self.alerts.send(class_name, behavior_title, msg)
        entry = {"time": time.time(), "behavior": behavior_title, "message": msg}
        frame_events.append(entry)
        if self.verbose:
            print(f"🚨 [{class_name}] {msg}")