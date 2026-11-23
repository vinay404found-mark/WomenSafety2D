"""
keypoint_utils.py (v2)
Shared geometry helpers + keypoint index constants + per-person history.

Two root-cause fixes vs v1:
  1. CONFIDENCE GATING: YOLO pose gives [x, y, confidence] per keypoint. v1 only
     checked x>0,y>0 -- it never looked at confidence, so low-confidence "guessed"
     keypoint positions (jitter) were treated as ground truth. Now kp_valid()
     requires confidence >= MIN_KEYPOINT_CONFIDENCE.
  2. BODY-SCALE NORMALIZATION: v1 normalized pixel thresholds by frame width.
     That's wrong when two people are at different distances from the camera
     within the SAME frame. Now every threshold is normalized by each person's
     own body scale (shoulder-to-hip distance) -- a natural per-person ruler
     that adapts to camera distance/zoom automatically.
"""

import numpy as np
import time
from collections import deque

# ---------------------------------------------------------------------------
# COCO-17 keypoint indices (ultralytics yolo11n-pose.pt)
# ---------------------------------------------------------------------------
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

HEAD_IDXS = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]

MIN_KEYPOINT_CONFIDENCE = 0.35   # below this, treat the keypoint as "not detected"
DEFAULT_BODY_SCALE = 150.0       # fallback px value when body scale can't be computed


# ---------------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------------
def get_distance(pt1, pt2):
    return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))


def kp_valid(person_kp, idx):
    """A keypoint is valid only if it has a real position AND the model is
    actually confident about it -- fixes the jitter-from-low-confidence-guess bug."""
    x, y = person_kp[idx][0], person_kp[idx][1]
    conf = person_kp[idx][2] if person_kp.shape[-1] > 2 else 1.0
    return x > 0 and y > 0 and conf >= MIN_KEYPOINT_CONFIDENCE


def safe_pt(person_kp, idx):
    return person_kp[idx][:2] if kp_valid(person_kp, idx) else None


def first_valid_pt(person_kp, *idxs):
    """Returns the first valid keypoint among the given indices, else None.
    Use this instead of `safe_pt(...) or safe_pt(...)` -- numpy arrays can't
    be used with Python's `or` (raises ValueError: ambiguous truth value)."""
    for idx in idxs:
        pt = safe_pt(person_kp, idx)
        if pt is not None:
            return pt
    return None


def safe_distance(person_kp1, idx1, person_kp2, idx2, fallback=9999.0):
    p1 = safe_pt(person_kp1, idx1)
    p2 = safe_pt(person_kp2, idx2)
    if p1 is None or p2 is None:
        return fallback
    return get_distance(p1, p2)


def get_head_center(person_kp):
    valid_pts = [person_kp[idx][:2] for idx in HEAD_IDXS if kp_valid(person_kp, idx)]
    if len(valid_pts) > 0:
        return np.mean(valid_pts, axis=0)
    return None


def get_torso_center(person_kp):
    idxs = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    valid_pts = [person_kp[i][:2] for i in idxs if kp_valid(person_kp, i)]
    if len(valid_pts) > 0:
        return np.mean(valid_pts, axis=0)
    return None


def get_body_scale(person_kp):
    """
    Per-person normalization unit: average shoulder-to-hip distance (torso
    height). This is the natural "ruler" for this specific person at their
    specific distance from the camera -- much more robust than frame-width
    scaling when people are at different depths in the same shot.
    Falls back to DEFAULT_BODY_SCALE if not enough keypoints are confidently
    visible (e.g. person is mostly occluded).
    """
    pairs = [(L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP)]
    measurements = []
    for sh_idx, hip_idx in pairs:
        sh = safe_pt(person_kp, sh_idx)
        hip = safe_pt(person_kp, hip_idx)
        if sh is not None and hip is not None:
            measurements.append(get_distance(sh, hip))
    if measurements:
        return float(np.mean(measurements))
    return DEFAULT_BODY_SCALE


def get_pair_scale(kp1, kp2):
    """Average body scale across both people in a pair -- used to normalize
    inter-person distances (which depend on both people's depth/size)."""
    return (get_body_scale(kp1) + get_body_scale(kp2)) / 2.0


def get_shoulder_angle(person_kp):
    ls, rs = safe_pt(person_kp, L_SHOULDER), safe_pt(person_kp, R_SHOULDER)
    if ls is None or rs is None:
        return None
    dx, dy = rs[0] - ls[0], rs[1] - ls[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def unit_vector(vec):
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return np.array([0.0, 0.0])
    return vec / n


def keypoint_quality(person_kp):
    """
    Fraction of all 17 keypoints that are confidently detected (0..1).
    A person mostly hidden behind others in a crowd will have most keypoints
    at low confidence/guessed positions -- this catches that BEFORE it can
    poison any distance/velocity calculation.
    """
    valid_count = sum(1 for idx in range(17) if kp_valid(person_kp, idx))
    return valid_count / 17.0


def any_wrist_near_body(owner_kp, target_kp, threshold):
    """True if either wrist of owner_kp is within `threshold` px of any of
    target_kp's torso/limb keypoints (shoulders, elbows, hips)."""
    wrist_idxs = [L_WRIST, R_WRIST]
    target_idxs = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_HIP, R_HIP]
    for w_idx in wrist_idxs:
        wrist_pt = safe_pt(owner_kp, w_idx)
        if wrist_pt is None:
            continue
        for t_idx in target_idxs:
            target_pt = safe_pt(target_kp, t_idx)
            if target_pt is None:
                continue
            if get_distance(wrist_pt, target_pt) < threshold:
                return True
    return False


def min_wrist_to_body_distance(owner_kp, target_kp):
    """Minimum distance from either wrist of owner_kp to any torso/limb
    keypoint of target_kp. Returns None if no valid pair exists."""
    wrist_idxs = [L_WRIST, R_WRIST]
    target_idxs = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_HIP, R_HIP]
    best = None
    for w_idx in wrist_idxs:
        wrist_pt = safe_pt(owner_kp, w_idx)
        if wrist_pt is None:
            continue
        for t_idx in target_idxs:
            target_pt = safe_pt(target_kp, t_idx)
            if target_pt is None:
                continue
            d = get_distance(wrist_pt, target_pt)
            if best is None or d < best:
                best = d
    return best


# ---------------------------------------------------------------------------
# Per-person history tracker
# ---------------------------------------------------------------------------
class PersonHistory:
    MIN_VELOCITY_WINDOW = 0.25  # seconds -- need at least this much real time baseline

    def __init__(self, maxlen_seconds=3.0, max_samples=90):
        self.maxlen_seconds = maxlen_seconds
        self.buffers = {}  # id -> deque[(t, head_xy, torso_xy)]
        self.max_samples = max_samples

    def update(self, person_id, head_xy, torso_xy):
        t = time.time()
        if person_id not in self.buffers:
            self.buffers[person_id] = deque(maxlen=self.max_samples)
        self.buffers[person_id].append((t, head_xy, torso_xy))
        self._trim(person_id)

    def _trim(self, person_id):
        buf = self.buffers[person_id]
        now = time.time()
        while buf and (now - buf[0][0]) > self.maxlen_seconds:
            buf.popleft()

    def cleanup(self, active_ids):
        stale = [pid for pid in self.buffers if pid not in active_ids]
        for pid in stale:
            del self.buffers[pid]

    def velocity(self, person_id):
        """
        Returns (speed_px_per_sec, direction_unit_vector).
        Uses MEDIAN of per-step speeds (robust to a single jittery frame)
        combined with the NET direction over the whole window (smoother than
        frame-to-frame direction). Requires a minimum real-time window before
        trusting the estimate at all -- prevents a tiny dt + a few px of
        jitter from producing a huge fake speed.
        """
        buf = self.buffers.get(person_id)
        if not buf or len(buf) < 3:
            return 0.0, np.array([0.0, 0.0])

        pts = [(t, h) for (t, h, _) in buf if h is not None]
        if len(pts) < 3:
            return 0.0, np.array([0.0, 0.0])

        total_dt = pts[-1][0] - pts[0][0]
        if total_dt < self.MIN_VELOCITY_WINDOW:
            return 0.0, np.array([0.0, 0.0])

        step_speeds = []
        for i in range(1, len(pts)):
            t0, h0 = pts[i - 1]
            t1, h1 = pts[i]
            dt = t1 - t0
            if dt < 0.01:
                continue
            disp = np.array(h1) - np.array(h0)
            step_speeds.append(np.linalg.norm(disp) / dt)

        if not step_speeds:
            return 0.0, np.array([0.0, 0.0])

        median_speed = float(np.median(step_speeds))
        net_disp = np.array(pts[-1][1]) - np.array(pts[0][1])
        direction = unit_vector(net_disp)
        return median_speed, direction

    def distance_history(self, person_id, other_id):
        b1 = self.buffers.get(person_id)
        b2 = self.buffers.get(other_id)
        if not b1 or not b2:
            return []
        dists = []
        for (t, h, _) in b1:
            closest = min(b2, key=lambda s: abs(s[0] - t))
            if h is not None and closest[1] is not None:
                dists.append(get_distance(h, closest[1]))
        return dists

    def is_still(self, person_id, speed_threshold=1.5):
        speed, _ = self.velocity(person_id)
        return speed < speed_threshold

    def track_age(self, person_id):
        """
        How long (seconds) this ID has been continuously tracked, based on
        the history buffer span. A brand-new or just-reassigned ID (common
        when ByteTrack swaps identities during crowd occlusion) will have a
        near-zero track age -- useful to require a minimum age before
        trusting velocity/behavior computed from that ID's history.
        """
        buf = self.buffers.get(person_id)
        if not buf or len(buf) < 2:
            return 0.0
        return buf[-1][0] - buf[0][0]
