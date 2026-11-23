"""
main.py (v2)
Live webcam/video runner. All detection logic lives in pipeline_core.py --
this file only handles capture, display, and drawing the HUD.

Run:
    pip install ultralytics twilio opencv-python transformers torch --break-system-packages
    export TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=... TWILIO_FROM_NUMBER=+1... ALERT_TO_NUMBER=+91...
    python3 main.py
"""

import os
import cv2
from ultralytics import YOLO
from pipeline_core import ThreatPipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(SCRIPT_DIR, "dataset", "non-violent", "try.mp4")
# For webcam: VIDEO_SOURCE = 0

DRY_RUN_SMS = True  # set False once you've tested and are ready for real texts


def main():
    print("🚀 Loading YOLO Engine (v2 -- fused risk engine)...")
    yolo_model = YOLO("yolo11s-pose.pt" if os.path.exists("yolo11s-pose.pt") else "yolo11n-pose.pt")

    pipeline = ThreatPipeline(dry_run_sms=DRY_RUN_SMS, enable_gender=True, verbose=True)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video source: '{VIDEO_SOURCE}'")
        return

    active_alerts_hud = []

    print("⚡ Analyzing frames... Press 'q' to terminate.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame, yolo_model)

        for event in result["events_this_frame"]:
            active_alerts_hud.append(event["message"])
        active_alerts_hud[:] = active_alerts_hud[-4:]

        frame_out = result["yolo_plotted_frame"]
        frame_h, frame_w = frame.shape[:2]
        scale_x, scale_y = 1280 / frame_w, 720 / frame_h
        frame_out = cv2.resize(frame_out, (1280, 720))

        risk = result["smoothed_risk"]
        if risk > 0.80:
            risk_color = (0, 0, 255)
        elif risk > 0.45:
            risk_color = (0, 165, 255)
        else:
            risk_color = (0, 255, 0)

        cv2.rectangle(frame_out, (15, 15), (560, 95), (20, 20, 20), -1)
        cv2.rectangle(frame_out, (15, 15), (560, 95), risk_color, 2)
        cv2.putText(frame_out, f"RISK: {risk*100:.1f}%", (28, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
        cv2.putText(frame_out, result["reason"][:52], (28, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if active_alerts_hud:
            log_h = 24 * len(active_alerts_hud) + 20
            cv2.rectangle(frame_out, (15, 720 - log_h - 15), (1000, 705), (20, 20, 20), -1)
            for k, msg in enumerate(active_alerts_hud):
                cv2.putText(frame_out, f"ALERT: {msg[:90]}", (25, 720 - log_h + 10 + 24 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        for (pid, kp, head_xy, gender_label, gender_conf, (x1, y1, x2, y2)) in result["people"]:
            if gender_label == "male":
                g_color = (255, 0, 0)
            elif gender_label == "female":
                g_color = (0, 255, 255)
            else:
                g_color = (0, 255, 0)
            sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
            label_text = f"{gender_label} {gender_conf:.0%}"
            text_y = max(20, sy1 - 10)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_out, (sx1, text_y - th - 6), (sx1 + tw + 6, text_y + 4), (0, 0, 0), -1)
            cv2.putText(frame_out, label_text, (sx1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, g_color, 2)

        cv2.imshow("Threat Detection v2 (Fused Risk Engine)", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📊 Process execution terminated cleanly.")


if __name__ == "__main__":
    main()
