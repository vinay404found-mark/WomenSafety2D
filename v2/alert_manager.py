"""
alert_manager.py

Sends an SMS via Twilio when a behavior crosses its sustained-duration
threshold. Reads credentials from environment variables so you never hardcode
secrets into the script:

    export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    export TWILIO_AUTH_TOKEN="your_auth_token"
    export TWILIO_FROM_NUMBER="+1xxxxxxxxxx"      # Twilio number you bought
    export ALERT_TO_NUMBER="+91xxxxxxxxxx"        # your phone

Install once:
    pip install twilio --break-system-packages
"""

import os
import time
import threading

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


class AlertManager:
    def __init__(self, global_cooldown_seconds=15.0, dry_run=False):
        """
        global_cooldown_seconds: minimum gap between ANY two SMS sends,
            regardless of behavior type, so a chaotic scene can't fire off
            20 texts in 10 seconds.
        dry_run: if True (or Twilio creds/lib missing), just prints instead
            of sending -- lets you test the pipeline without burning SMS credits.
        """
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = os.environ.get("TWILIO_FROM_NUMBER")
        self.to_number = os.environ.get("ALERT_TO_NUMBER")

        self.global_cooldown = global_cooldown_seconds
        self.last_sent = 0.0
        self.lock = threading.Lock()

        creds_ok = all([self.account_sid, self.auth_token, self.from_number, self.to_number])
        self.dry_run = dry_run or (not TWILIO_AVAILABLE) or (not creds_ok)

        if self.dry_run:
            reason = []
            if not TWILIO_AVAILABLE:
                reason.append("twilio package not installed")
            if not creds_ok:
                reason.append("missing one or more TWILIO_*/ALERT_TO_NUMBER env vars")
            print(f"⚠️  AlertManager running in DRY-RUN mode ({', '.join(reason) or 'forced'}). "
                  f"SMS will be printed, not sent.")
        else:
            self.client = Client(self.account_sid, self.auth_token)
            print("✅ AlertManager: Twilio client ready.")

    def send(self, class_name, behavior_name, extra_info=""):
        """
        Fire-and-forget SMS send, respecting the global cooldown.
        Runs the actual network call in a background thread so it never
        blocks/stutters the video loop.
        """
        now = time.time()
        with self.lock:
            if (now - self.last_sent) < self.global_cooldown:
                return False  # still in global cooldown, skip silently
            self.last_sent = now

        message_body = (
            f"🚨 THREAT ALERT [{class_name}]\n"
            f"Behavior: {behavior_name}\n"
            f"{extra_info}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        threading.Thread(target=self._dispatch, args=(message_body,), daemon=True).start()
        return True

    def _dispatch(self, body):
        if self.dry_run:
            print(f"--- [DRY-RUN SMS] ---\n{body}\n----------------------")
            return
        try:
            msg = self.client.messages.create(
                body=body,
                from_=self.from_number,
                to=self.to_number,
            )
            print(f"📤 SMS sent (sid={msg.sid})")
        except Exception as e:
            print(f"❌ Twilio send failed: {e}")
