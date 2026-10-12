import cv2
import numpy as np
import threading
import time
import csv
from ultralytics import YOLO


# =========================
# YOLO SETUP
# =========================

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_frame = None
latest_detection = None
frame_lock = threading.Lock()
running = True


# =========================
# YOLO DETECTION THREAD
# =========================

def detection_worker():
    global latest_frame, latest_detection

    while running:

        with frame_lock:
            frame_copy = (
                latest_frame.copy()
                if latest_frame is not None
                else None
            )

        if frame_copy is None:
            time.sleep(0.001)
            continue

        # Run YOLO with a stricter confidence threshold
        results = model(
            frame_copy,
            imgsz=640,
            conf=0.5,
            verbose=False
        )

        boxes = results[0].boxes

        if len(boxes) > 0:

            # Only accept the two balloon classes
            valid_indices = []

            for i, cls in enumerate(boxes.cls):
                class_id = int(cls.item())

                if class_id in [0, 1]:
                    valid_indices.append(i)

            if len(valid_indices) > 0:

                # Choose highest-confidence valid balloon
                best_idx = max(
                    valid_indices,
                    key=lambda i: boxes.conf[i].item()
                )

                box = boxes[best_idx]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detection = (
                    cx, cy,
                    x1, y1,
                    x2, y2,
                    conf
                )

                with frame_lock:
                    latest_detection = detection

            else:
                with frame_lock:
                    latest_detection = None

        else:
            with frame_lock:
                latest_detection = None


detector_thread = threading.Thread(
    target=detection_worker,
    daemon=True
)

detector_thread.start()


# =========================
# KALMAN FILTER
# =========================

kalman = cv2.KalmanFilter(4, 2)

# Measurement = [x, y]
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], np.float32)

# Approximate camera frame interval
dt = 1.0 / 30.0

# State = [x, y, vx, vy]
kalman.transitionMatrix = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)

# Process noise
kalman.processNoiseCov = (
    np.eye(4, dtype=np.float32) * 0.001
)

# Measurement noise
kalman.measurementNoiseCov = (
    np.eye(2, dtype=np.float32) * 10.0
)

kalman.errorCovPost = np.eye(
    4,
    dtype=np.float32
)

kalman_ready = False


# =========================
# LOGGING
# =========================

throw_active = False

log_filename = f"balloon_log_{int(time.time())}.csv"

csv_file = open(
    log_filename,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "timestamp",
    "raw_cx",
    "raw_cy",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "confidence",
    "kalman_x",
    "kalman_y",
    "kalman_vx",
    "kalman_vy",
    "throw_active"
])


# =========================
# MAIN LOOP
# =========================

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    # Give latest frame to YOLO thread
    with frame_lock:
        latest_frame = frame.copy()
        detection = latest_detection

    now = time.time() - start_time


    # =========================
    # BALLOON DETECTED
    # =========================

    if detection is not None:

        (
            cx, cy,
            x1, y1,
            x2, y2,
            conf
        ) = detection

        measurement = np.array([
            [np.float32(cx)],
            [np.float32(cy)]
        ])


        # Initialize Kalman on first detection
        if not kalman_ready:

            kalman.statePre = np.array([
                [cx],
                [cy],
                [0],
                [0]
            ], np.float32)

            kalman.statePost = np.array([
                [cx],
                [cy],
                [0],
                [0]
            ], np.float32)

            kalman_ready = True


        # Predict
        prediction = kalman.predict()

        # Correct using YOLO measurement
        corrected = kalman.correct(measurement)

        kx = corrected[0, 0]
        ky = corrected[1, 0]

        kvx = corrected[2, 0]
        kvy = corrected[3, 0]


        # =========================
        # LOG
        # =========================

        csv_writer.writerow([
            now,
            cx,
            cy,
            x1,
            y1,
            x2,
            y2,
            conf,
            kx,
            ky,
            kvx,
            kvy,
            throw_active
        ])


        # =========================
        # DRAW YOLO BOX
        # =========================

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Balloon {conf:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        # Raw YOLO center
        cv2.circle(
            frame,
            (int(cx), int(cy)),
            5,
            (0, 255, 255),
            -1
        )


        # Kalman center
        cv2.circle(
            frame,
            (int(kx), int(ky)),
            6,
            (0, 0, 255),
            -1
        )


        # Velocity
        cv2.putText(
            frame,
            f"V: ({kvx:.1f}, {kvy:.1f})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


    # =========================
    # NO BALLOON DETECTED
    # =========================

    else:

        # IMPORTANT:
        # Do NOT draw a Kalman prediction here.
        # This prevents the system from visually
        # appearing to detect a balloon when YOLO
        # actually sees nothing.

        csv_writer.writerow([
            now,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            throw_active
        ])


    # =========================
    # STATUS
    # =========================

    status = "THROW ACTIVE" if throw_active else "IDLE"

    cv2.putText(
        frame,
        status,
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "T = toggle throw | Q = quit",
        (10, 435),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    # =========================
    # DISPLAY
    # =========================

    cv2.imshow(
        "Balloon Tracker",
        frame
    )

    key = cv2.waitKey(1) & 0xFF


    # Toggle throw
    if key == ord("t"):

        throw_active = not throw_active

        # Start a new throw with fresh Kalman state
        if throw_active:
            kalman_ready = False


    # Quit
    elif key == ord("q"):

        break


# =========================
# CLEANUP
# =========================

running = False

detector_thread.join(timeout=1)

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"Log saved to: {log_filename}")