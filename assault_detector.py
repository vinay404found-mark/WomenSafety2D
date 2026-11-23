"""
assault_detector.py

Detects direct physical strikes -- kicks and punches -- between two tracked
people. Distances are normalized by body scale (average of both people's
shoulder-to-hip size), not fixed pixels, so it behaves consistently
regardless of camera distance.

Deliberately a SHORT sustained-duration behavior (see ASSAULT_HOLD_SECONDS in
main.py) -- strikes are fast, but a single frame of proximity is still noise;
requiring a brief hold filters that out without requiring a real strike to be
held anywhere near as long as the SOS/Fall/Contact behaviors.
"""

from keypoint_utils import safe_pt, get_distance, get_body_scale, L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP

PUNCH_DISTANCE_RATIO = 0.45   # wrist-to-head/shoulder, in units of body scale
KICK_DISTANCE_RATIO = 0.50    # ankle-to-hip/torso, in units of body scale

# Very low absolute floor -- just rejects near-unusable/noisy tiny detections.
# Deliberately NOT the main discriminator, since real deployments may have
# cameras mounted far from the action (whole scene appears small) -- an
# absolute pixel cutoff would reject genuine footage in that case.
MIN_BODY_SCALE_FOR_ASSAULT = 25.0

# THE MAIN DISCRIMINATOR: how similar the two people's apparent body sizes
# must be to each other. Two people actually near each other in the real
# world are at roughly the SAME distance from the camera, so they appear
# roughly the SAME size on screen -- regardless of whether the camera itself
# is close or far away. Two unrelated people who just happen to overlap in
# the 2D image (but are at different real-world depths) will have
# noticeably different apparent sizes. Measured from real footage: a
# genuine fighting pair had a scale ratio of 1.00-1.01; a false-positive
# pair from unrelated people at different depths measured 1.57. This ratio
# check works the same whether the camera is 2 meters or 20 meters away.
MAX_SCALE_RATIO_FOR_ASSAULT = 1.20


def _pair_scale(kp1, kp2):
    return (get_body_scale(kp1) + get_body_scale(kp2)) / 2.0


def assault_detected(p1, p2):
    """
    Returns (detected: bool, label: str or None) -- "punch" or "kick".
    Checks both directions (p1 striking p2, and p2 striking p1).
    Returns (False, None) if either person is too small to evaluate at all,
    or if the two people's apparent sizes differ too much (implying they're
    at different real-world depths and any proximity is likely a 2D
    coincidence, not real contact) -- see constants above.
    """
    from keypoint_utils import get_body_scale

    scale1, scale2 = get_body_scale(p1), get_body_scale(p2)
    scale = (scale1 + scale2) / 2.0

    if scale < MIN_BODY_SCALE_FOR_ASSAULT:
        return False, None

    scale_ratio = max(scale1, scale2) / max(min(scale1, scale2), 1e-6)
    if scale_ratio > MAX_SCALE_RATIO_FOR_ASSAULT:
        return False, None

    # --- Punch: either person's wrist near the other's shoulder/head area ---
    punch_candidates = []
    for owner, target in [(p1, p2), (p2, p1)]:
        for w_idx in [L_WRIST, R_WRIST]:
            wrist = safe_pt(owner, w_idx)
            if wrist is None:
                continue
            for t_idx in [L_SHOULDER, R_SHOULDER]:
                target_pt = safe_pt(target, t_idx)
                if target_pt is None:
                    continue
                punch_candidates.append(get_distance(wrist, target_pt))

    if punch_candidates and min(punch_candidates) < PUNCH_DISTANCE_RATIO * scale:
        return True, "punch"

    # --- Kick: either person's ankle near the other's hip/torso area ---
    kick_candidates = []
    for owner, target in [(p1, p2), (p2, p1)]:
        for a_idx in [L_ANKLE, R_ANKLE]:
            ankle = safe_pt(owner, a_idx)
            if ankle is None:
                continue
            for t_idx in [L_HIP, R_HIP]:
                target_pt = safe_pt(target, t_idx)
                if target_pt is None:
                    continue
                kick_candidates.append(get_distance(ankle, target_pt))

    if kick_candidates and min(kick_candidates) < KICK_DISTANCE_RATIO * scale:
        return True, "kick"

    return False, None