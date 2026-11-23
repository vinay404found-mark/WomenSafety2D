"""
duration_tracker.py

A behavior must be sustained for its configured duration before it fires --
not just true on one frame. Uses a leaky bucket: accumulated "credit" drains
away when the condition isn't true, so brief flicker/jitter can't slowly
accumulate into a false alert over a long span. Includes a per-key cooldown
so a single ongoing incident doesn't spam repeat alerts.
"""

import time


class DurationTracker:
    def __init__(self, leak_rate_per_sec=1.3, cooldown_seconds=30.0):
        self.leak_rate = leak_rate_per_sec
        self.cooldown_seconds = cooldown_seconds
        self.state = {}  # key -> {"accum": float, "last_update": t, "last_fired": t or None}

    @staticmethod
    def _key(id1, id2, behavior):
        a, b = sorted([int(id1), int(id2)])
        return (a, b, behavior)

    def update(self, id1, id2, behavior, condition_true, threshold_seconds):
        key = self._key(id1, id2, behavior)
        now = time.time()

        if key not in self.state:
            self.state[key] = {"accum": 0.0, "last_update": now, "last_fired": None}

        st = self.state[key]
        dt = max(0.0, now - st["last_update"])
        st["last_update"] = now

        if condition_true:
            st["accum"] += dt
        else:
            st["accum"] = max(0.0, st["accum"] - dt * self.leak_rate)

        st["accum"] = min(st["accum"], threshold_seconds * 1.5)

        in_cooldown = (
            st["last_fired"] is not None
            and (now - st["last_fired"]) < self.cooldown_seconds
        )

        trigger = False
        if st["accum"] >= threshold_seconds and not in_cooldown:
            trigger = True
            st["last_fired"] = now
            st["accum"] = 0.0

        progress = min(1.0, st["accum"] / threshold_seconds) if threshold_seconds > 0 else 0.0

        return {"accum": st["accum"], "trigger": trigger, "progress": progress, "in_cooldown": in_cooldown}

    def cleanup(self, active_pair_keys):
        stale = [k for k in self.state if (k[0], k[1]) not in active_pair_keys]
        for k in stale:
            del self.state[k]
