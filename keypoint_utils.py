
import numpy as np
import time
from collections import deque

# COCO-17 keypoint indices (ultralytics yolo pose models)
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

MIN_KEYPOINT_CONFIDENCE = 0.35
DEFAULT_BODY_SCALE = 150.0


def get_distance(pt1, pt2):
    return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))


def kp_valid(person_kp, idx):
    x, y = person_kp[idx][0], person_kp[idx][1]
    conf = person_kp[idx][2] if person_kp.shape[-1] > 2 else 1.0
    return x > 0 and y > 0 and conf >= MIN_KEYPOINT_CONFIDENCE


def safe_pt(person_kp, idx):
    return person_kp[idx][:2] if kp_valid(person_kp, idx) else None


def get_head_center(person_kp):
    idxs = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
    pts = [person_kp[i][:2] for i in idxs if kp_valid(person_kp, i)]
    return np.mean(pts, axis=0) if pts else None


def get_torso_center(person_kp):
    idxs = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    pts = [person_kp[i][:2] for i in idxs if kp_valid(person_kp, i)]
    return np.mean(pts, axis=0) if pts else None


def get_body_scale(person_kp):
    """Per-person ruler: average shoulder-to-hip distance in pixels."""
    pairs = [(L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP)]
    measurements = []
    for sh_idx, hip_idx in pairs:
        sh, hip = safe_pt(person_kp, sh_idx), safe_pt(person_kp, hip_idx)
        if sh is not None and hip is not None:
            measurements.append(get_distance(sh, hip))
    return float(np.mean(measurements)) if measurements else DEFAULT_BODY_SCALE


def keypoint_quality(person_kp):
    """Fraction of all 17 keypoints confidently detected -- used to reject
    heavily-occluded people before they can produce false detections."""
    return sum(1 for i in range(17) if kp_valid(person_kp, i)) / 17.0


class PersonHistory:
    """Tracks recent head position per person ID, for speed/stillness checks."""
    MIN_VELOCITY_WINDOW = 0.25  # seconds

    def __init__(self, maxlen_seconds=3.0, max_samples=90):
        self.maxlen_seconds = maxlen_seconds
        self.buffers = {}
        self.max_samples = max_samples

    def update(self, person_id, head_xy):
        t = time.time()
        if person_id not in self.buffers:
            self.buffers[person_id] = deque(maxlen=self.max_samples)
        self.buffers[person_id].append((t, head_xy))
        self._trim(person_id)

    def _trim(self, person_id):
        buf = self.buffers[person_id]
        if not buf:
            return
        latest = buf[-1][0]
        while buf and (latest - buf[0][0]) > self.maxlen_seconds:
            buf.popleft()

    def cleanup(self, active_ids):
        for pid in [p for p in self.buffers if p not in active_ids]:
            del self.buffers[pid]

    def speed(self, person_id):
        """Median px/sec speed over the buffered window. Robust to single-frame jitter."""
        buf = self.buffers.get(person_id)
        if not buf or len(buf) < 3:
            return 0.0
        pts = [(t, h) for (t, h) in buf if h is not None]
        if len(pts) < 3:
            return 0.0
        total_dt = pts[-1][0] - pts[0][0]
        if total_dt < self.MIN_VELOCITY_WINDOW:
            return 0.0
        step_speeds = []
        for i in range(1, len(pts)):
            dt = pts[i][0] - pts[i - 1][0]
            if dt < 0.01:
                continue
            disp = get_distance(pts[i][1], pts[i - 1][1])
            step_speeds.append(disp / dt)
        return float(np.median(step_speeds)) if step_speeds else 0.0

    def track_age(self, person_id):
        buf = self.buffers.get(person_id)
        if not buf or len(buf) < 2:
            return 0.0
        return buf[-1][0] - buf[0][0]
