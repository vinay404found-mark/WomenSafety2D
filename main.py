import os
import cv2
from ultralytics import YOLO
from keypoint_utils import PersonHistory, get_head_center, keypoint_quality
from duration_tracker import DurationTracker
from alert_manager import AlertManager
from gender_classifier import GenderClassifier
from sos_detector import sos_gesture_detected
from fall_detector import fall_detected
from contact_detector import prolonged_contact_condition
from assault_detector import assault_detected

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SOURCE = os.path.join(SCRIPT_DIR, "dataset","non-violent","try.mp4")
# For webcam: VIDEO_SOURCE = 0

DRY_RUN_SMS = True  
CAMERA_LABEL = "Floor 2"
SOS_HOLD_SECONDS = 4.0
FALL_HOLD_SECONDS = 4.0
ASSAULT_HOLD_SECONDS = 0.2      
CONTACT_HOLD_SECONDS = 5.0


MIN_QUALITY_FOR_FALL = 0.20     # fall_detected only needs one shoulder+hip pair -- kept
MIN_QUALITY_FOR_SOS = 0.45      # needs nose + both wrists
MIN_QUALITY_FOR_PAIRWISE = 0.55 # assault/contact need wrists + shoulders/hips reliably

ALERT_COOLDOWN_SECONDS = 30.0


def main():
    print("🚀 Loading YOLO pose model...")
    yolo_model = YOLO("yolo11n-pose.pt")

    history = PersonHistory(maxlen_seconds=3.0)
    tracker = DurationTracker(leak_rate_per_sec=1.3, cooldown_seconds=ALERT_COOLDOWN_SECONDS)
    alerts = AlertManager(global_cooldown_seconds=20.0, dry_run=DRY_RUN_SMS, camera_label=CAMERA_LABEL)
    gender_clf = GenderClassifier()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f" ERROR: Could not open video source: '{VIDEO_SOURCE}'")
        return

    active_alerts_hud = []

    print(" Analyzing frames... Press 'q' to terminate")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        current_ids = []
        people = []  # (id, kp, head_xy, gender, gender_conf, bbox, quality)

        if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
            box_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints_list = results[0].keypoints.data.cpu().numpy()
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

            for idx, pid in enumerate(box_ids):
                if idx >= len(keypoints_list):
                    continue
                kp = keypoints_list[idx]
                head_xy = get_head_center(kp)
                if head_xy is None:
                    continue  # can't even track this person's position at all -- skip entirely

                quality = keypoint_quality(kp)
                history.update(pid, head_xy)

                x1, y1, x2, y2 = map(int, boxes_xyxy[idx])
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                gender_label, gender_conf = gender_clf.get_gender(pid, crop)

                people.append((pid, kp, head_xy, gender_label, gender_conf, (x1, y1, x2, y2), quality))
                current_ids.append(pid)

            # ---------------- 1. SOS Gesture (per person) ----------------
            for (pid, kp, head_xy, gender_label, gender_conf, bbox, quality) in people:
                if quality < MIN_QUALITY_FOR_SOS:
                    continue
                sos_now = sos_gesture_detected(kp)
                status = tracker.update(pid, pid, "sos_gesture", sos_now, SOS_HOLD_SECONDS)
                if status["trigger"]:
                    msg = f"🚨 SOS Gesture Detected -- Camera: {CAMERA_LABEL}"
                    alerts.send("SOS Gesture Detected", [f"Camera: {CAMERA_LABEL}", "Emergency Alert Sent"])
                    active_alerts_hud.append(msg)

            # ---------------- 2. Fall Detection (per person) ----------------
            for (pid, kp, head_xy, gender_label, gender_conf, bbox, quality) in people:
                if quality < MIN_QUALITY_FOR_FALL:
                    continue
                fall_now = fall_detected(kp)
                status = tracker.update(pid, pid, "fall", fall_now, FALL_HOLD_SECONDS)
                if status["trigger"]:
                    msg = f"🚨 Fall Detected -- Possible Medical Emergency"
                    alerts.send("Fall Detected", ["Possible Medical Emergency", "Alert Sent"])
                    active_alerts_hud.append(msg)

            # ---------------- 3 & 4. Pairwise: Assault + Prolonged Contact ----------------
            for i in range(len(people)):
                for j in range(i + 1, len(people)):
                    id1, p1, head1, gender1, gconf1, _, q1 = people[i]
                    id2, p2, head2, gender2, gconf2, _, q2 = people[j]

                    if min(q1, q2) < MIN_QUALITY_FOR_PAIRWISE:
                        continue

                    # --- 3. Physical Assault (kick/punch) -- fires regardless of gender ---
                    is_assault, assault_label = assault_detected(p1, p2)
                    status_a = tracker.update(id1, id2, "assault", is_assault, ASSAULT_HOLD_SECONDS)
                    if status_a["trigger"]:
                        gender_note = f"({gender1} & {gender2})" if "unknown" not in (gender1, gender2) else ""
                        msg = f"🚨 Physical Assault Detected ({assault_label}) {gender_note}"
                        alerts.send("Physical Assault Detected", [
                            f"Type: {assault_label.title()}",
                            f"Camera: {CAMERA_LABEL}",
                            "Alert Sent",
                        ])
                        active_alerts_hud.append(msg)

                    # --- 4. Prolonged Physical Contact -- requires confirmed male+female ---
                    if gender1 == "male" and gender2 == "female":
                        male_kp, female_kp, male_id, female_id = p1, p2, id1, id2
                    elif gender2 == "male" and gender1 == "female":
                        male_kp, female_kp, male_id, female_id = p2, p1, id2, id1
                    else:
                        continue  # gender not confidently both determined -- skip, don't guess

                    condition = prolonged_contact_condition(male_kp, female_kp, male_id, female_id, history)
                    status_c = tracker.update(male_id, female_id, "prolonged_contact", condition, CONTACT_HOLD_SECONDS)

                    if status_c["trigger"]:
                        snapshot_path = alerts.save_snapshot(frame, "prolonged_contact")
                        msg = " Prolonged Physical Contact Detected"
                        alerts.send("Prolonged Physical Contact Detected", [
                            f"Camera: {CAMERA_LABEL}",
                            f"Snapshot: {os.path.basename(snapshot_path) if snapshot_path else 'N/A'}",
                            "Please Verify",
                        ])
                        active_alerts_hud.append(msg)
        history.cleanup(current_ids)
        gender_clf.cleanup(current_ids)
        active_alerts_hud[:] = active_alerts_hud[-4:]
        frame_h, frame_w = frame.shape[:2]
        scale_x, scale_y = 1280 / frame_w, 720 / frame_h
        frame_out = results[0].plot() if people else frame
        frame_out = cv2.resize(frame_out, (1280, 720))

        if active_alerts_hud:
            log_h = 24 * len(active_alerts_hud) + 20
            cv2.rectangle(frame_out, (15, 15), (900, 15 + log_h), (20, 20, 20), -1)
            for k, msg in enumerate(active_alerts_hud):
                cv2.putText(frame_out, msg, (25, 40 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        else:
            cv2.rectangle(frame_out, (15, 15), (300, 55), (20, 20, 20), -1)
            cv2.putText(frame_out, "No Alert", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (pid, kp, head_xy, gender_label, gender_conf, (x1, y1, x2, y2), quality) in people:
            g_color = (255, 0, 0) if gender_label == "male" else (0, 255, 255) if gender_label == "female" else (0, 255, 0)
            sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
            label_text = f"{gender_label} {gender_conf:.0%}"
            text_y = max(20, sy1 - 10)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_out, (sx1, text_y - th - 6), (sx1 + tw + 6, text_y + 4), (0, 0, 0), -1)
            cv2.putText(frame_out, label_text, (sx1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, g_color, 2)
        cv2.imshow("Safety Monitor (SOS / Fall / Assault / Prolonged Contact)", frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(" Process execution terminated cleanly.")

if __name__ == "__main__":
    main()