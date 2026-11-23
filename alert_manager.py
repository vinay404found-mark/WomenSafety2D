"""
alert_manager.py

Sends SMS via Twilio. For the "Prolonged Physical Contact" alert specifically,
also saves a snapshot frame to disk (per your demo flow: contact -> Snapshot ->
SMS -> Please Verify) and references the saved file in the message.

Note on MMS (actual image IN the text message): Twilio requires the image be
hosted at a public HTTPS URL it can fetch (mediaUrl) -- a raw local file path
won't work over MMS. This script saves the snapshot locally and includes the
filename/path in the SMS body so whoever receives it can go pull the image
from the machine/server. If you want the actual photo embedded in the text
itself, you'd need to host the snapshots folder somewhere public (e.g. a
small file server, S3 bucket, etc.) and pass that URL to Twilio's media_url
parameter -- ask me if you want that wired in.

Read credentials from environment variables:
    export TWILIO_ACCOUNT_SID=...
    export TWILIO_AUTH_TOKEN=...
    export TWILIO_FROM_NUMBER=+1...
    export ALERT_TO_NUMBER=+91...
"""

import os
import time
import threading
import cv2

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


class AlertManager:
    def __init__(self, global_cooldown_seconds=20.0, dry_run=False, camera_label="Camera 1"):
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = os.environ.get("TWILIO_FROM_NUMBER")
        self.to_number = os.environ.get("ALERT_TO_NUMBER")
        self.camera_label = camera_label

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
                reason.append("missing TWILIO_*/ALERT_TO_NUMBER env vars")
            print(f"⚠️  AlertManager in DRY-RUN mode ({', '.join(reason) or 'forced'}). SMS printed, not sent.")
        else:
            self.client = Client(self.account_sid, self.auth_token)
            print("✅ AlertManager: Twilio client ready.")

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    def save_snapshot(self, frame, tag):
        """Saves a snapshot frame to disk, returns the saved path."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}_{timestamp}.jpg"
        path = os.path.join(SNAPSHOT_DIR, filename)
        try:
            cv2.imwrite(path, frame)
            return path
        except Exception as e:
            print(f"⚠️ Failed to save snapshot: {e}")
            return None

    def send(self, title, body_lines):
        """
        title: short header, e.g. "SOS Gesture Detected"
        body_lines: list of strings, each becomes its own line in the SMS
        """
        now = time.time()
        with self.lock:
            if (now - self.last_sent) < self.global_cooldown:
                return False
            self.last_sent = now

        full_body = f"🚨 {title}\n\n" + "\n".join(body_lines) + f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        threading.Thread(target=self._dispatch, args=(full_body,), daemon=True).start()
        return True

    def _dispatch(self, body):
        if self.dry_run:
            print(f"--- [DRY-RUN SMS] ---\n{body}\n----------------------")
            return
        try:
            msg = self.client.messages.create(body=body, from_=self.from_number, to=self.to_number)
            print(f"📤 SMS sent (sid={msg.sid})")
        except Exception as e:
            print(f"❌ Twilio send failed: {e}")
