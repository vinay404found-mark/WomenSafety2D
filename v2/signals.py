"""
signals.py

Computes CONTINUOUS evidence scores (0.0-1.0) for a tracked pair, per frame.
These are deliberately NOT booleans -- the risk_engine fuses several of these
together and requires multiple to agree before anything counts as "true" for
the duration tracker. This directly implements the anti-false-positive spec:
proximity alone (or any single signal alone) must never be enough.

All distance thresholds are normalized by get_pair_scale() (per-person body
size in pixels), not frame resolution -- so behavior is consistent whether
someone is close to or far from the camera.
"""

import numpy as np
from keypoint_utils import (
    get_distance, safe_pt, safe_distance, get_pair_scale, min_wrist_to_body_distance,
    L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER,
)


def _clamp01(x):
    return max(0.0, min(1.0, x))


def contact_intensity(p1, p2, scale):
    """
    0..1 score for how much either person's hand is on the other's body.
    0 = no contact (wrist further than 1.6x body-scale from any body point)
    1 = deep/firm contact (wrist within 0.35x body-scale)
    """
    d1 = min_wrist_to_body_distance(p1, p2)
    d2 = min_wrist_to_body_distance(p2, p1)
    candidates = [d for d in [d1, d2] if d is not None]
    if not candidates:
        return 0.0
    min_dist = min(candidates)

    near = 0.35 * scale
    far = 1.6 * scale
    if min_dist >= far:
        return 0.0
    if min_dist <= near:
        return 1.0
    return _clamp01((far - min_dist) / (far - near))


def aggressive_motion_score(p1, p2, id1, id2, history, scale):
    """
    0..1 score for a sharp velocity spike by either person directed toward
    the other -- the geometric signature of a lunge/grab/shove/pull motion,
    as opposed to calm approach or incidental passing.
    """
    speed1, dir1 = history.velocity(id1)
    speed2, dir2 = history.velocity(id2)
    head1_buf = history.buffers.get(id1)
    head2_buf = history.buffers.get(id2)
    if not head1_buf or not head2_buf:
        return 0.0
    head1 = head1_buf[-1][1]
    head2 = head2_buf[-1][1]
    if head1 is None or head2 is None:
        return 0.0

    def toward_score(speed, direction, my_head, other_head):
        # normalize speed against body scale: "body-heights per second" instead of raw px/sec
        normalized_speed = speed / scale
        if normalized_speed < 0.3:
            return 0.0
        to_other = np.array(other_head) - np.array(my_head)
        norm = np.linalg.norm(to_other)
        if norm < 1e-6:
            return 0.0
        alignment = np.dot(direction, to_other / norm)
        if alignment < 0.5:
            return 0.0
        # 0 body-heights/sec -> 0 score, 3+ body-heights/sec -> full score
        return _clamp01(normalized_speed / 3.0) * alignment

    return max(toward_score(speed1, dir1, head1, head2), toward_score(speed2, dir2, head2, head1))


def escape_attempt_score(p1, p2, id1, id2, history, scale):
    """
    0..1 score for the specific "victim trying to create distance while the
    other person keeps closing it" pattern -- distinguishes an actual chase
    from two people simply walking in the same direction together.
    """
    dist_hist = history.distance_history(id1, id2)
    if len(dist_hist) < 5:
        return 0.0

    net_change = dist_hist[-1] - dist_hist[0]
    if net_change >= 0:
        return 0.0  # distance isn't closing at all -- no chase signature

    speed1, dir1 = history.velocity(id1)
    speed2, dir2 = history.velocity(id2)
    head1_buf = history.buffers.get(id1)
    head2_buf = history.buffers.get(id2)
    if not head1_buf or not head2_buf:
        return 0.0
    head1 = head1_buf[-1][1]
    head2 = head2_buf[-1][1]
    if head1 is None or head2 is None:
        return 0.0

    def fleeing_while_caught(speed_a, dir_a, head_a, head_b, speed_b):
        away = np.array(head_a) - np.array(head_b)
        norm = np.linalg.norm(away)
        if norm < 1e-6 or (speed_a / scale) < 0.25:
            return 0.0
        alignment_away = np.dot(dir_a, away / norm)
        if alignment_away < 0.4:
            return 0.0  # not actually moving away from the other
        catch_up_ratio = _clamp01(speed_b / max(speed_a, 1e-6))
        closing_magnitude = _clamp01(abs(net_change) / scale)
        return alignment_away * catch_up_ratio * closing_magnitude

    return max(
        fleeing_while_caught(speed1, dir1, head1, head2, speed2),
        fleeing_while_caught(speed2, dir2, head2, head1, speed1),
    )


def violent_action_detected(p1, p2, scale):
    """
    Standalone (non-fused) detection of unambiguous violent actions: punches,
    kicks. These don't need multi-signal fusion because the action itself is
    already high-confidence evidence -- but they DO need real contact,
    normalized by body scale, not a loose fixed radius.
    Returns (detected: bool, label: str or None)
    """
    punch_dist = min(
        safe_distance(p1, L_WRIST, p2, L_SHOULDER), safe_distance(p1, R_WRIST, p2, R_SHOULDER),
        safe_distance(p2, L_WRIST, p1, L_SHOULDER), safe_distance(p2, R_WRIST, p1, R_SHOULDER),
    )
    if punch_dist < 0.45 * scale:
        return True, "punch"

    kick_dist = min(
        safe_distance(p1, L_ANKLE, p2, L_HIP), safe_distance(p1, R_ANKLE, p2, R_HIP),
        safe_distance(p2, L_ANKLE, p1, L_HIP), safe_distance(p2, R_ANKLE, p1, R_HIP),
    )
    if kick_dist < 0.5 * scale:
        return True, "kick"

    return False, None


def ground_fall_detected(person_kp):
    """True if hip and shoulder are at nearly the same height (lying down),
    threshold scaled by this person's own body size."""
    from keypoint_utils import get_body_scale
    try:
        hip_y = person_kp[L_HIP][1]
        shoulder_y = person_kp[L_SHOULDER][1]
        scale = get_body_scale(person_kp)
        if hip_y > 0 and shoulder_y > 0 and abs(hip_y - shoulder_y) < 0.25 * scale:
            return True
    except (IndexError, TypeError):
        pass
    return False
