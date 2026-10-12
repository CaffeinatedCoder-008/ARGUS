import cv2
import numpy as np
import threading
import time
import csv
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "best.pt"

OUTPUT_CSV = "hand_tracking_data.csv"
OUTPUT_VIDEO = "hand_tracking_result.mp4"

CONF_THRESHOLD = 0.5

IMG_SIZE = 320

Q = 0.15
R = 1.0

MAX_JUMP_DISTANCE = 250
MAX_CONSECUTIVE_REJECTIONS = 5


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

model = YOLO(MODEL_PATH)

latest_frame = None
latest_detection = None

frame_lock = threading.Lock()

running = True


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


        results = model(
            frame_copy,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )


        boxes = results[0].boxes


        if len(boxes) > 0:

            best_idx = int(boxes.conf.argmax())

            box = boxes[best_idx]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            conf = box.conf[0].item()

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2


            with frame_lock:

                latest_detection = (
                    cx,
                    cy,
                    x1,
                    y1,
                    x2,
                    y2,
                    conf
                )

        else:

            with frame_lock:

                latest_detection = None


detector_thread = threading.Thread(
    target=detection_worker,
    daemon=True
)

detector_thread.start()


# ============================================================
# KALMAN FILTER
# ============================================================

kalman = cv2.KalmanFilter(4, 2)


# State:
#
# x
# y
# vx
# vy
decay = 0.75
kalman.transitionMatrix = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, decay, 0],
        [0, 0, 0, decay ]
    ],
    np.float32
)


# Measurement:
#
# x
# y

kalman.measurementMatrix = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ],
    np.float32
)


kalman.processNoiseCov = (
    np.eye(4, dtype=np.float32) * Q
)


kalman.measurementNoiseCov = (
    np.eye(2, dtype=np.float32) * R
)


kalman.errorCovPost = np.eye(
    4,
    dtype=np.float32
)


kalman_ready = False


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    30,
    (ws, hs)
)


# ============================================================
# CSV
# ============================================================

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline=""
)

writer = csv.writer(csv_file)


writer.writerow([
    "frame",
    "timestamp",

    "detected",

    "yolo_x",
    "yolo_y",
    "confidence",

    "kalman_x",
    "kalman_y",

    "kalman_vx",
    "kalman_vy",

    "yolo_step",
    "kalman_step",

    "yolo_kalman_distance",

    "x_separation",
    "y_separation"
])


# ============================================================
# VARIABLES
# ============================================================

frame_number = 0

start_time = time.time()

previous_yolo = None
previous_kalman = None

consecutive_rejections = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, img = cap.read()

    if not success:

        print("Camera frame couldn't be read!")

        break


    # --------------------------------------------------------
    # Send frame to YOLO
    # --------------------------------------------------------

    with frame_lock:

        latest_frame = img.copy()

        detection = latest_detection


    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    timestamp = time.time() - start_time


    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    hs_actual, ws_actual = img.shape[:2]

    center_x = ws_actual // 2
    center_y = hs_actual // 2


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    detected = False

    yolo_x = np.nan
    yolo_y = np.nan
    confidence = np.nan

    kalman_x = np.nan
    kalman_y = np.nan

    kalman_vx = np.nan
    kalman_vy = np.nan

    yolo_step = np.nan
    kalman_step = np.nan

    yolo_kalman_distance = np.nan

    x_separation = np.nan
    y_separation = np.nan


    # ========================================================
    # DETECTION
    # ========================================================

    if detection:

        raw_fx, raw_fy, x1, y1, x2, y2, conf = detection


        # ----------------------------------------------------
        # Jump gating
        # ----------------------------------------------------

        if kalman_ready:

            current_kx = float(
                kalman.statePost[0, 0]
            )

            current_ky = float(
                kalman.statePost[1, 0]
            )


            jump = np.sqrt(
                (raw_fx - current_kx) ** 2 +
                (raw_fy - current_ky) ** 2
            )


            if jump > MAX_JUMP_DISTANCE:

                consecutive_rejections += 1


                if consecutive_rejections < MAX_CONSECUTIVE_REJECTIONS:

                    detection = None

                else:

                    consecutive_rejections = 0

            else:

                consecutive_rejections = 0


    # ========================================================
    # VALID DETECTION
    # ========================================================

    if detection:

        raw_fx, raw_fy, x1, y1, x2, y2, conf = detection

        detected = True

        yolo_x = raw_fx
        yolo_y = raw_fy
        confidence = conf


        # ----------------------------------------------------
        # Measurement
        # ----------------------------------------------------

        measurement = np.array(
            [
                [np.float32(raw_fx)],
                [np.float32(raw_fy)]
            ]
        )


        # ----------------------------------------------------
        # Initialize Kalman
        # ----------------------------------------------------

        if not kalman_ready:

            kalman.statePre = np.array(
                [
                    [raw_fx],
                    [raw_fy],
                    [0],
                    [0]
                ],
                np.float32
            )


            kalman.statePost = np.array(
                [
                    [raw_fx],
                    [raw_fy],
                    [0],
                    [0]
                ],
                np.float32
            )


            kalman_ready = True


        # ----------------------------------------------------
        # Correct
        # ----------------------------------------------------

        kalman.correct(measurement)


        # ----------------------------------------------------
        # Predict next state
        # ----------------------------------------------------

        prediction = kalman.predict()


        kx = float(prediction[0, 0])
        ky = float(prediction[1, 0])

        kvx = float(prediction[2, 0])
        kvy = float(prediction[3, 0])


        kalman_x = kx
        kalman_y = ky

        kalman_vx = kvx
        kalman_vy = kvy


        # ====================================================
        # YOLO FRAME-TO-FRAME MOVEMENT
        # ====================================================

        if previous_yolo is not None:

            yolo_step = np.sqrt(
                (raw_fx - previous_yolo[0]) ** 2 +
                (raw_fy - previous_yolo[1]) ** 2
            )


        # ====================================================
        # KALMAN FRAME-TO-FRAME MOVEMENT
        # ====================================================

        if previous_kalman is not None:

            kalman_step = np.sqrt(
                (kx - previous_kalman[0]) ** 2 +
                (ky - previous_kalman[1]) ** 2
            )


        # ====================================================
        # YOLO ↔ KALMAN SEPARATION
        # ====================================================

        x_separation = abs(raw_fx - kx)

        y_separation = abs(raw_fy - ky)


        yolo_kalman_distance = np.sqrt(
            (raw_fx - kx) ** 2 +
            (raw_fy - ky) ** 2
        )


        previous_yolo = (
            raw_fx,
            raw_fy
        )


        previous_kalman = (
            kx,
            ky
        )


        # ====================================================
        # DRAW YOLO
        # ====================================================

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )


        cv2.circle(
            img,
            (int(raw_fx), int(raw_fy)),
            5,
            (0, 255, 255),
            -1
        )


        # ====================================================
        # DRAW KALMAN
        # ====================================================

        cv2.circle(
            img,
            (int(kx), int(ky)),
            12,
            (0, 0, 255),
            -1
        )


        cv2.circle(
            img,
            (int(kx), int(ky)),
            60,
            (0, 0, 255),
            2
        )


        # ====================================================
        # CROSSHAIR
        # ====================================================

        cv2.line(
            img,
            (center_x - 30, center_y),
            (center_x + 30, center_y),
            (255, 255, 255),
            2
        )

        cv2.line(
            img,
            (center_x, center_y - 30),
            (center_x, center_y + 30),
            (255, 255, 255),
            2
        )


        # ====================================================
        # TEXT
        # ====================================================

        cv2.putText(
            img,
            f"YOLO: ({int(raw_fx)}, {int(raw_fy)})",
            (30, 40),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Kalman: ({int(kx)}, {int(ky)})",
            (30, 75),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
            2
        )


        cv2.putText(
            img,
            f"Velocity: ({kvx:.1f}, {kvy:.1f}) px/frame",
            (30, 110),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 0, 0),
            2
        )


        cv2.putText(
            img,
            f"YOLO-Kalman: {yolo_kalman_distance:.1f}px",
            (30, 145),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Q={Q}  R={R}",
            (30, 180),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            img,
            "NO TARGET",
            (center_x - 100, 60),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            3
        )


    # ========================================================
    # CSV
    # ========================================================

    writer.writerow([
        frame_number,
        timestamp,

        detected,

        yolo_x,
        yolo_y,
        confidence,

        kalman_x,
        kalman_y,

        kalman_vx,
        kalman_vy,

        yolo_step,
        kalman_step,

        yolo_kalman_distance,

        x_separation,
        y_separation
    ])


    # ========================================================
    # SAVE VIDEO
    # ========================================================

    video_writer.write(img)


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Hand Tracking - YOLO + Kalman",
        img
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    frame_number += 1


# ============================================================
# CLEANUP
# ============================================================

running = False

cap.release()

video_writer.release()

csv_file.close()

cv2.destroyAllWindows()


print("\nRecording finished.")

print(f"CSV saved to: {OUTPUT_CSV}")

print(f"Video saved to: {OUTPUT_VIDEO}")