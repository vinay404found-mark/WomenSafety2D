"""
sos_detector.py

Detects the SOS gesture: both wrists raised clearly above the head.
This is a deliberate, unambiguous, sustained pose -- almost nothing in normal
daily behavior accidentally holds both arms up for several continuous seconds,
which is exactly why it's one of the most reliable signals available from 2D pose.
"""

from keypoint_utils import safe_pt, get_body_scale, NOSE, L_WRIST, R_WRIST

# How far above the nose (in body-scale units) a wrist must be to count as
# "clearly raised" -- not just at head height, genuinely above it.
RAISED_MARGIN_RATIO = 0.15


def sos_gesture_detected(person_kp):
    """
    Returns True if BOTH wrists are confidently detected and clearly above
    the head level for this person, this frame.
    """
    nose = safe_pt(person_kp, NOSE)
    left_wrist = safe_pt(person_kp, L_WRIST)
    right_wrist = safe_pt(person_kp, R_WRIST)

    if nose is None or left_wrist is None or right_wrist is None:
        return False

    scale = get_body_scale(person_kp)
    margin = RAISED_MARGIN_RATIO * scale

    # Image y-coordinates increase downward -- "above" means a SMALLER y value.
    left_raised = left_wrist[1] < (nose[1] - margin)
    right_raised = right_wrist[1] < (nose[1] - margin)

    return left_raised and right_raised
