"""
contact_detector.py

Detects "Prolonged Physical Contact": a male's wrist touching a female's
shoulder or torso, while BOTH people are mostly stationary.

KEY FIX: the old version only checked "is a wrist within X distance of the
other person's torso/shoulder" -- but torso-center is a big, forgiving
target. Two people just standing near each other in a crowd, arms resting
normally at their sides, can easily have a wrist fall within that radius
from proximity ALONE, with zero actual touching. That was the source of
"false alerts just from standing near each other."

The real signal that distinguishes "arm resting at your side, incidentally
near someone" from "arm actively placed on someone" is REACH: a resting arm
stays close to your OWN torso. A hand placed on someone else's body requires
your arm to be extended AWAY from your own torso. This function now requires
BOTH conditions:
  1. wrist is close to the other person's shoulder/torso (proximity)
  2. that same wrist is far from the OWNER's own torso (reach/extension)
Only satisfying both means a real touch, not incidental closeness.
"""

from keypoint_utils import safe_pt, get_distance, get_body_scale, get_torso_center, L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER

# Contact distance threshold, in units of body scale (avg of both people).
CONTACT_DISTANCE_RATIO = 0.30

# How far the touching wrist must be from the OWNER's own torso, in units of
# the owner's own body scale, to count as "reaching out" rather than resting.
MIN_REACH_RATIO = 0.55

# Stillness threshold in px/sec, normalized by body scale (roughly "how many
# body-heights per second" -- below this counts as "mostly stationary").
STILL_SPEED_RATIO = 0.5


def _best_contact_wrist(owner_kp, target_kp):
    """
    Finds the owner's wrist that is BOTH:
      - close to the target's shoulder/torso, AND
      - extended away from the owner's own torso (not just resting at their side)
    Returns the contact distance if such a wrist exists, else None.
    """
    target_points = []
    ls, rs = safe_pt(target_kp, L_SHOULDER), safe_pt(target_kp, R_SHOULDER)
    torso = get_torso_center(target_kp)
    if ls is not None:
        target_points.append(ls)
    if rs is not None:
        target_points.append(rs)
    if torso is not None:
        target_points.append(torso)
    if not target_points:
        return None

    owner_torso = get_torso_center(owner_kp)
    owner_scale = get_body_scale(owner_kp)
    if owner_torso is None:
        return None

    best = None
    for w_idx in [L_WRIST, R_WRIST]:
        wrist = safe_pt(owner_kp, w_idx)
        if wrist is None:
            continue

        # Reach check: is this wrist actually extended away from the owner's
        # own body, or just resting near it? A resting hand fails this check
        # regardless of how close it happens to be to someone else.
        reach = get_distance(wrist, owner_torso)
        if reach < MIN_REACH_RATIO * owner_scale:
            continue

        for tp in target_points:
            d = get_distance(wrist, tp)
            if best is None or d < best:
                best = d
    return best


def prolonged_contact_condition(male_kp, female_kp, male_id, female_id, history):
    """
    Returns True only if, on THIS frame:
      - male's wrist is EXTENDED (reaching, not resting) AND within contact
        distance of female's shoulder/torso
      - BOTH people are moving slower than the stillness threshold

    Caller is responsible for sustaining this over the required duration
    (via duration_tracker) and for confirming the gender pairing beforehand.
    """
    scale = (get_body_scale(male_kp) + get_body_scale(female_kp)) / 2.0

    contact_dist = _best_contact_wrist(male_kp, female_kp)
    if contact_dist is None or contact_dist >= CONTACT_DISTANCE_RATIO * scale:
        return False

    male_speed = history.speed(male_id)
    female_speed = history.speed(female_id)
    still_threshold = STILL_SPEED_RATIO * scale

    return male_speed < still_threshold and female_speed < still_threshold