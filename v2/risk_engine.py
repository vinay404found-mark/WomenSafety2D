"""
risk_engine.py

Fuses the continuous signals from signals.py into a single decision per pair,
per frame. This is where the anti-false-positive spec gets enforced in code:

  - Proximity alone is NEVER a signal here at all (it doesn't even appear).
  - No single signal, however high, is sufficient on its own for the fused
    "harassment_pattern" condition -- at least 2 of {contact, aggressive
    motion, escape attempt} must each individually clear a sub-threshold in
    the SAME frame before the fused condition can even be True.
  - The fused condition still has to be sustained for several seconds via
    duration_tracker.py before an alert fires -- this module only decides
    what counts as "true" on a given frame, not when to alert.

Violent actions (punch/kick/fall) bypass fusion entirely -- they're treated
as standalone high-confidence evidence per the spec ("hitting, punching,
kicking" are listed as trigger-worthy on their own), but still have to be
sustained for a short duration (not just 1 frame) before alerting.
"""

from signals import (
    contact_intensity, aggressive_motion_score, escape_attempt_score,
    violent_action_detected, ground_fall_detected,
)
from keypoint_utils import get_pair_scale

# Sub-thresholds: how strong a signal must be, on its own, to "count" toward
# the multi-signal requirement.
CONTACT_SUBTHRESHOLD = 0.4
MOTION_SUBTHRESHOLD = 0.4
ESCAPE_SUBTHRESHOLD = 0.35

# Combined fused score needed (weighted sum) on top of the 2-signal requirement.
FUSED_SCORE_THRESHOLD = 0.5
MIN_SIGNALS_REQUIRED = 2

WEIGHTS = {
    "contact": 0.35,
    "motion": 0.30,
    "escape": 0.35,
}


def evaluate_pair(p1, p2, id1, id2, history):
    """
    Returns a dict describing this pair's risk state for the current frame:
      {
        "harassment_pattern": bool,   # fused, multi-signal, feed into duration_tracker
        "violent_action": bool,       # standalone punch/kick, feed into duration_tracker separately
        "violent_label": str or None,
        "fall_detected_p1": bool,
        "fall_detected_p2": bool,
        "fused_score": float,         # 0..1, useful for the continuous risk meter/HUD
        "signals": {"contact":.., "motion":.., "escape":..},
      }
    """
    scale = get_pair_scale(p1, p2)

    contact = contact_intensity(p1, p2, scale)
    motion = aggressive_motion_score(p1, p2, id1, id2, history, scale)
    escape = escape_attempt_score(p1, p2, id1, id2, history, scale)

    signals_above_subthreshold = sum([
        contact >= CONTACT_SUBTHRESHOLD,
        motion >= MOTION_SUBTHRESHOLD,
        escape >= ESCAPE_SUBTHRESHOLD,
    ])

    fused_score = (
        WEIGHTS["contact"] * contact +
        WEIGHTS["motion"] * motion +
        WEIGHTS["escape"] * escape
    )

    harassment_pattern = (
        signals_above_subthreshold >= MIN_SIGNALS_REQUIRED
        and fused_score >= FUSED_SCORE_THRESHOLD
    )

    violent_action, violent_label = violent_action_detected(p1, p2, scale)

    return {
        "harassment_pattern": harassment_pattern,
        "violent_action": violent_action,
        "violent_label": violent_label,
        "fall_detected_p1": ground_fall_detected(p1),
        "fall_detected_p2": ground_fall_detected(p2),
        "fused_score": fused_score,
        "signals": {"contact": contact, "motion": motion, "escape": escape},
    }
