import cv2
import numpy as np
import threading
import time
import csv
from collections import deque
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't access!")
    exit()


# ============================================================
# YOLO
# ============================================================

model = YOLO("best.pt")

latest_frame = None
latest_detection = None
frame_lock = threading.Lock()
running = True


def detection_worker():
    global latest_frame, latest_detection
    while running:
        with frame_lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None
        if frame_copy is None:
            time.sleep(0.001)
            continue

        results = model(frame_copy, imgsz=320, conf=0.5, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            best_idx = int(boxes.conf.argmax())
            box = boxes[best_idx]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            with frame_lock:
                latest_detection = (cx, cy, x1, y1, x2, y2, conf)
        else:
            with frame_lock:
                latest_detection = None


detector_thread = threading.Thread(target=detection_worker, daemon=True)
detector_thread.start()


# ============================================================
# KALMAN FILTER
# ============================================================

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)
kalman_ready = False


# ============================================================
# PREDICTION SETTINGS
# ============================================================

LEAD_TIME = 0.15          # seconds into the future to predict -- FIX: time-based, not frame-count-based
HISTORY_SIZE = 15

position_history = deque(maxlen=HISTORY_SIZE)
timestamp_history = deque(maxlen=HISTORY_SIZE)   # FIX: real elapsed time, paired with each position

predicted_point = None    # single (px, py) prediction for LEAD_TIME ahead, used for aiming/servo
predicted_path = []       # short visual-only path, for on-screen drawing

# Logged for delayed, time-correct accuracy comparison (FIX for Bug 2)
session_timestamps = []
session_positions = []          # Kalman-smoothed (x, y) at each session_timestamps[i]
pending_predictions = []        # each: {"made_t": ..., "target_t": ..., "pred_x": ..., "pred_y": ...}

start_time = time.time()


# ============================================================
# VIRTUAL SERVO
# ============================================================

servoPos = [90, 90]


# ============================================================
# FPS
# ============================================================

prev_time = 0
fps_list = []


# ============================================================
# MAIN LOOP
# ============================================================

while True:
    success, img = cap.read()
    if not success:
        print("Camera frame couldn't be read!")
        break

    with frame_lock:
        latest_frame = img.copy()
        detection = latest_detection

    hs_actual, ws_actual = img.shape[:2]
    center_x = ws_actual // 2
    center_y = hs_actual // 2

    curr_time = time.time()
    if prev_time:
        fps = 1 / (curr_time - prev_time)
        fps_list.append(fps)
    prev_time = curr_time
    if len(fps_list) > 30:
        fps_list.pop(0)
    avg_fps = int(sum(fps_list) / len(fps_list)) if fps_list else 0

    now_rel = curr_time - start_time

    if detection:
        raw_fx, raw_fy, x1, y1, x2, y2, conf = detection
        measurement = np.array([[np.float32(raw_fx)], [np.float32(raw_fy)]])

        if not kalman_ready:
            kalman.statePre = np.array([[raw_fx], [raw_fy], [0], [0]], np.float32)
            kalman.statePost = np.array([[raw_fx], [raw_fy], [0], [0]], np.float32)
            kalman_ready = True

        kalman.correct(measurement)
        prediction = kalman.predict()
        kx, ky = float(prediction[0, 0]), float(prediction[1, 0])
        kvx, kvy = float(prediction[2, 0]), float(prediction[3, 0])

        # ---- Store for the fit AND for the session log, both timestamped ----
        position_history.append((kx, ky))
        timestamp_history.append(now_rel)

        session_timestamps.append(now_rel)
        session_positions.append((kx, ky))

        # ============================================================
        # TRAJECTORY PREDICTION -- FIXED: fit against real elapsed time
        # ============================================================

        predicted_point = None
        predicted_path = []

        if len(position_history) >= 5:
            points = np.array(position_history)
            times = np.array(timestamp_history)
            t_rel = times - times[0]   # FIX: seconds, not frame index

            try:
                x_coeff = np.polyfit(t_rel, points[:, 0], 2)
                y_coeff = np.polyfit(t_rel, points[:, 1], 2)

                # Single prediction point, LEAD_TIME seconds ahead -- this is
                # what actually gets used for aiming/servo and for validation.
                future_t = t_rel[-1] + LEAD_TIME
                pred_x = float(np.polyval(x_coeff, future_t))
                pred_y = float(np.polyval(y_coeff, future_t))
                pred_x = np.clip(pred_x, 0, ws_actual - 1)
                pred_y = np.clip(pred_y, 0, hs_actual - 1)
                predicted_point = (pred_x, pred_y)

                # Log this prediction with WHEN it was made and WHAT time it's
                # predicting for -- lets us fairly compare later (FIX for Bug 2).
                pending_predictions.append({
                    "made_t": now_rel,
                    "target_t": now_rel + LEAD_TIME,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                })

                # Short visual-only path (several points between now and LEAD_TIME)
                for frac in np.linspace(0.2, 1.0, 6):
                    vt = t_rel[-1] + LEAD_TIME * frac
                    vx = np.clip(np.polyval(x_coeff, vt), 0, ws_actual - 1)
                    vy = np.clip(np.polyval(y_coeff, vt), 0, hs_actual - 1)
                    predicted_path.append((int(vx), int(vy)))

            except np.linalg.LinAlgError:
                predicted_point = None
                predicted_path = []

        # ============================================================
        # VIRTUAL SERVO -- now driven by the PREDICTED point, not raw Kalman
        # ============================================================

        aim_x, aim_y = predicted_point if predicted_point else (kx, ky)

        servoX = np.clip(np.interp(aim_x, [0, ws_actual], [0, 180]), 0, 180)
        servoY = np.clip(np.interp(aim_y, [0, hs_actual], [0, 180]), 0, 180)
        servoPos[0] = servoX
        servoPos[1] = servoY

        # ---- Draw YOLO detection ----
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(img, (int(raw_fx), int(raw_fy)), 5, (0, 255, 255), -1)

        # ---- Draw Kalman ----
        cv2.circle(img, (int(kx), int(ky)), 12, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (int(kx), int(ky)), 60, (0, 0, 255), 2)

        # ---- Draw history ----
        if len(position_history) >= 2:
            pts = list(position_history)
            for i in range(len(pts) - 1):
                p1 = (int(pts[i][0]), int(pts[i][1]))
                p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                cv2.line(img, p1, p2, (255, 0, 0), 2)

        # ---- Draw predicted path ----
        if len(predicted_path) >= 2:
            for i in range(len(predicted_path) - 1):
                cv2.line(img, predicted_path[i], predicted_path[i + 1], (255, 0, 255), 3)
        if predicted_point:
            cv2.circle(img, (int(predicted_point[0]), int(predicted_point[1])), 8, (255, 0, 255), cv2.FILLED)

        # ---- Crosshair ----
        cv2.line(img, (center_x - 30, center_y), (center_x + 30, center_y), (255, 255, 255), 2)
        cv2.line(img, (center_x, center_y - 30), (center_x, center_y + 30), (255, 255, 255), 2)

        # ---- Aiming status ----
        distance = np.sqrt((aim_x - center_x) ** 2 + (aim_y - center_y) ** 2)
        status_text = "AIM LOCKED" if distance < 50 else "TRACKING"
        status_color = (0, 255, 0) if distance < 50 else (0, 255, 255)
        cv2.putText(img, status_text, (center_x - 100, center_y - 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, status_color, 3)

        # ---- Text overlay ----
        cv2.putText(img, f"Raw: ({int(raw_fx)}, {int(raw_fy)})", (30, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        cv2.putText(img, f"Kalman: ({int(kx)}, {int(ky)})", (30, 75),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        cv2.putText(img, f"Velocity: ({kvx:.1f}, {kvy:.1f})", (30, 110),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.putText(img, f"Lead time: {LEAD_TIME*1000:.0f} ms", (30, 145),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
        cv2.putText(img, f"Virtual Servo X: {int(servoPos[0])} deg", (30, 180),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.putText(img, f"Virtual Servo Y: {int(servoPos[1])} deg", (30, 215),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        cv2.putText(img, f"FPS: {avg_fps}", (30, 250),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
        cv2.putText(img, "PURPLE = PREDICTED PATH", (30, hs_actual - 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    else:
        cv2.putText(img, "NO TARGET", (center_x - 100, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        cv2.circle(img, (center_x, center_y), 60, (0, 0, 255), 2)

    cv2.imshow("Balloon Tracking + Prediction", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

running = False
cap.release()
cv2.destroyAllWindows()


# ============================================================
# DELAYED-COMPARISON ACCURACY ANALYSIS -- FIX for Bug 2
#
# For each prediction made at time `made_t` targeting `target_t`,
# find the ACTUAL position closest to `target_t` (not the position
# at the time the prediction was made) and compare against that.
# ============================================================

session_timestamps_arr = np.array(session_timestamps)
session_positions_arr = np.array(session_positions)

matched_actual = []
matched_predicted = []
MAX_MATCH_TOLERANCE = 0.08  # seconds -- only accept a match if an actual sample exists this close to target_t

for p in pending_predictions:
    target_t = p["target_t"]
    if len(session_timestamps_arr) == 0:
        continue
    idx = np.argmin(np.abs(session_timestamps_arr - target_t))
    if abs(session_timestamps_arr[idx] - target_t) <= MAX_MATCH_TOLERANCE:
        matched_actual.append(session_positions_arr[idx])
        matched_predicted.append((p["pred_x"], p["pred_y"]))

if len(matched_actual) > 10:
    actual = np.array(matched_actual)
    predicted = np.array(matched_predicted)

    errors = np.sqrt((actual[:, 0] - predicted[:, 0]) ** 2 + (actual[:, 1] - predicted[:, 1]) ** 2)

    print("\n===== PREDICTION ANALYSIS (time-correct, delayed comparison) =====")
    print(f"Matched prediction/actual pairs: {len(matched_actual)} "
          f"(out of {len(pending_predictions)} predictions made)")
    print(f"Lead time: {LEAD_TIME*1000:.0f} ms")
    print(f"Mean prediction error: {np.mean(errors):.2f} px")
    print(f"Median prediction error: {np.median(errors):.2f} px")
    print(f"Maximum prediction error: {np.max(errors):.2f} px")

    plt.figure(figsize=(12, 6))
    plt.plot(actual[:, 0], label="Actual X")
    plt.plot(predicted[:, 0], label="Predicted X", linestyle="--")
    plt.xlabel("Matched sample #")
    plt.ylabel("X position (pixels)")
    plt.title(f"Balloon X: Actual vs Predicted ({LEAD_TIME*1000:.0f}ms lead, time-correct)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_x_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(actual[:, 1], label="Actual Y")
    plt.plot(predicted[:, 1], label="Predicted Y", linestyle="--")
    plt.xlabel("Matched sample #")
    plt.ylabel("Y position (pixels)")
    plt.title(f"Balloon Y: Actual vs Predicted ({LEAD_TIME*1000:.0f}ms lead, time-correct)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_y_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(actual[:, 0], actual[:, 1], label="Actual trajectory")
    plt.plot(predicted[:, 0], predicted[:, 1], "--", label="Predicted trajectory")
    plt.scatter(actual[0, 0], actual[0, 1], label="Start")
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.gca().invert_yaxis()
    plt.title("Balloon Trajectory: Actual vs Predicted (time-correct)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trajectory_prediction_comparison.png", dpi=150)
    plt.close()

    print("\nSaved:\n  prediction_x_comparison.png\n  prediction_y_comparison.png"
          "\n  trajectory_prediction_comparison.png")

    with open("prediction_test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual_x", "actual_y", "predicted_x", "predicted_y", "error_px"])
        for a, p, e in zip(actual, predicted, errors):
            writer.writerow([a[0], a[1], p[0], p[1], e])
    print("Prediction data saved to prediction_test.csv")

else:
    print(f"\nNot enough matched pairs ({len(matched_actual)}) for prediction analysis. "
          f"Try a longer session or a larger MAX_MATCH_TOLERANCE.")
