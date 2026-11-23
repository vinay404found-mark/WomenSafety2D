"""
duration_tracker.py

Implements the "how long before we trust this" logic we discussed:
- Each (pair_id, behavior_name) gets its own leaky-bucket accumulator.
- Accumulator rises while the behavior condition is true, decays (doesn't
  hard-reset) when it briefly drops out (handles pose-jitter/flicker).
- Once accumulator crosses the behavior's configured threshold (seconds),
  it fires ONCE, then enters a cooldown window before it can fire again
  for that same pair+behavior.
- Also exposes a "watch" level (~50% of threshold) so main.py can show a
  building-up indicator on the HUD before the real alert fires.
"""

import time


class DurationTracker:
    def __init__(self, leak_rate_per_sec=0.6, cooldown_seconds=8.0, max_dt_per_update=0.35):
        """
        leak_rate_per_sec: how fast the accumulator drains (in seconds of
            'credit') per real second when the condition is false.
            Lower = more forgiving of brief flicker/noise.
        cooldown_seconds: after firing, how long to wait before this exact
            pair+behavior can fire again.
        max_dt_per_update: hard cap (seconds) on how much time a single
            update() call can add to the accumulator. Without this, a slow
            frame (heavy model inference, low FPS) could inject several
            seconds of "sustained" credit from a single True reading --
            exactly the bug that causes near-instant false triggers on
            short clips or laggy hardware.
        """
        self.leak_rate = leak_rate_per_sec
        self.cooldown_seconds = cooldown_seconds
        self.max_dt_per_update = max_dt_per_update

        # key -> {"accum": float_seconds, "last_update": t, "last_fired": t or None}
        self.state = {}

    @staticmethod
    def _key(id1, id2, behavior):
        a, b = sorted([int(id1), int(id2)])
        return (a, b, behavior)

    def update(self, id1, id2, behavior, condition_true, threshold_seconds):
        """
        Call this every frame for every (pair, behavior) you're evaluating.
        Returns a dict:
            {
              "accum": current accumulated seconds,
              "watch": True if >= 50% of threshold,
              "trigger": True exactly once when threshold crossed & cooldown clear,
              "progress": 0..1 fraction toward threshold
            }
        """
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

        # cap accumulation so it doesn't grow unbounded past threshold
        st["accum"] = min(st["accum"], threshold_seconds * 1.5)

        in_cooldown = (
            st["last_fired"] is not None
            and (now - st["last_fired"]) < self.cooldown_seconds
        )

        trigger = False
        if st["accum"] >= threshold_seconds and not in_cooldown:
            trigger = True
            st["last_fired"] = now
            st["accum"] = 0.0  # reset after firing so it must build up again

        progress = min(1.0, st["accum"] / threshold_seconds) if threshold_seconds > 0 else 0.0

        return {
            "accum": st["accum"],
            "watch": progress >= 0.5,
            "trigger": trigger,
            "progress": progress,
            "in_cooldown": in_cooldown,
        }

    def cleanup(self, active_pair_keys):
        """Optional: drop state for pairs that no longer co-exist in frame."""
        stale = [k for k in self.state if (k[0], k[1]) not in active_pair_keys]
        for k in stale:
            del self.state[k]
