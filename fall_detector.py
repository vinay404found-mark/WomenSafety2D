"""
fall_detector.py

Detects a person lying on the ground: hip and shoulder keypoints at nearly
the same vertical height, which happens when standing/sitting anatomy gets
"flattened" horizontally. Threshold is scaled by the person's own body size,
not a fixed pixel number.
"""

from keypoint_utils import safe_pt, get_body_scale, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP

# If |hip_y - shoulder_y| is less than this fraction of the person's own
# body scale, they're considered horizontal (fallen), not standing/sitting.
FALL_HEIGHT_RATIO = 0.25


def fall_detected(person_kp):
    shoulder = safe_pt(person_kp, L_SHOULDER)
    if shoulder is None:
        shoulder = safe_pt(person_kp, R_SHOULDER)
    hip = safe_pt(person_kp, L_HIP)
    if hip is None:
        hip = safe_pt(person_kp, R_HIP)

    if shoulder is None or hip is None:
        return False

    scale = get_body_scale(person_kp)
    threshold = FALL_HEIGHT_RATIO * scale

    return abs(hip[1] - shoulder[1]) < threshold
