import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "hand_tracking_data.csv"

Q = 0.15
R = 1.0

DECAY_VALUES = [1.00, 0.85, 0.75, 0.65]

OUTPUT_DIR = "decay_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# KALMAN FILTER
# ============================================================

def run_kalman(df, decay):

    kalman = cv2.KalmanFilter(4, 2)

    # State:
    # [x, y, vx, vy]
    #
    # Position:
    # x(t+1) = x(t) + vx(t)
    # y(t+1) = y(t) + vy(t)
    #
    # Velocity:
    # vx(t+1) = decay * vx(t)
    # vy(t+1) = decay * vy(t)

    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, decay, 0],
        [0, 0, 0, decay]
    ], dtype=np.float32)

    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * Q
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * R
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    initialized = False

    results = []

    previous_x = None
    previous_y = None

    for _, row in df.iterrows():

        frame = int(row["frame"])

        detected = bool(row["detected"])

        # ----------------------------------------------------
        # No detection
        # ----------------------------------------------------

        if (
            not detected
            or pd.isna(row["yolo_x"])
            or pd.isna(row["yolo_y"])
        ):
            results.append({
                "frame": frame,
                "yolo_x": np.nan,
                "yolo_y": np.nan,
                "kalman_x": np.nan,
                "kalman_y": np.nan,
                "kalman_vx": np.nan,
                "kalman_vy": np.nan,
                "separation": np.nan,
                "kalman_step": np.nan
            })

            continue

        x = float(row["yolo_x"])
        y = float(row["yolo_y"])

        # ----------------------------------------------------
        # Initialize
        # ----------------------------------------------------

        if not initialized:

            kalman.statePost = np.array([
                [x],
                [y],
                [0],
                [0]
            ], dtype=np.float32)

            kx = x
            ky = y
            vx = 0.0
            vy = 0.0

            initialized = True

        else:

            # Predict
            kalman.predict()

            # Measurement
            measurement = np.array([
                [x],
                [y]
            ], dtype=np.float32)

            # Correct
            corrected = kalman.correct(measurement)

            kx = float(corrected[0, 0])
            ky = float(corrected[1, 0])
            vx = float(corrected[2, 0])
            vy = float(corrected[3, 0])

        # ----------------------------------------------------
        # Kalman movement
        # ----------------------------------------------------

        if previous_x is not None:

            step = np.sqrt(
                (kx - previous_x) ** 2 +
                (ky - previous_y) ** 2
            )

        else:
            step = np.nan

        previous_x = kx
        previous_y = ky

        # ----------------------------------------------------
        # YOLO ↔ Kalman separation
        # ----------------------------------------------------

        separation = np.sqrt(
            (x - kx) ** 2 +
            (y - ky) ** 2
        )

        results.append({
            "frame": frame,
            "yolo_x": x,
            "yolo_y": y,
            "kalman_x": kx,
            "kalman_y": ky,
            "kalman_vx": vx,
            "kalman_vy": vy,
            "separation": separation,
            "kalman_step": step
        })

    return pd.DataFrame(results)


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 65)
print("KALMAN DECAY VISUAL COMPARISON")
print("=" * 65)

df = pd.read_csv(CSV_PATH)

print(f"\nInput CSV       : {CSV_PATH}")
print(f"Total frames    : {len(df)}")

valid = (
    df["detected"].astype(bool)
    & df["yolo_x"].notna()
    & df["yolo_y"].notna()
)

print(f"YOLO detections : {valid.sum()}")


# ============================================================
# RUN ALL DECAY VALUES
# ============================================================

all_results = {}

for decay in DECAY_VALUES:

    print(f"\nProcessing DECAY = {decay}")

    result = run_kalman(df, decay)

    all_results[decay] = result

    # Save individual CSV
    filename = os.path.join(
        OUTPUT_DIR,
        f"kalman_decay_{decay:.2f}.csv"
    )

    result.to_csv(filename, index=False)

    print(f"Saved: {filename}")


# ============================================================
# PLOT 1
# YOLO VS KALMAN TRAJECTORIES
# ============================================================

plt.figure(figsize=(12, 8))

# YOLO trajectory
yolo_x = df.loc[valid, "yolo_x"]
yolo_y = df.loc[valid, "yolo_y"]

plt.plot(
    yolo_x,
    yolo_y,
    label="YOLO",
    linewidth=2,
    alpha=0.8
)

# Kalman trajectories
for decay in DECAY_VALUES:

    result = all_results[decay]

    plt.plot(
        result["kalman_x"],
        result["kalman_y"],
        label=f"Kalman decay={decay}",
        linewidth=1.8
    )

plt.xlabel("X position (pixels)")
plt.ylabel("Y position (pixels)")
plt.title("YOLO vs Kalman Trajectory — Different Velocity Decay")
plt.legend()
plt.grid(True, alpha=0.3)

# Camera coordinates usually have Y increasing downward.
plt.gca().invert_yaxis()

plt.tight_layout()

trajectory_path = os.path.join(
    OUTPUT_DIR,
    "decay_trajectory_comparison.png"
)

plt.savefig(trajectory_path, dpi=200)
plt.show()

print(f"\nSaved: {trajectory_path}")


# ============================================================
# PLOT 2
# YOLO-KALMAN SEPARATION
# ============================================================

plt.figure(figsize=(12, 6))

for decay in DECAY_VALUES:

    result = all_results[decay]

    plt.plot(
        result["frame"],
        result["separation"],
        label=f"Decay={decay}",
        linewidth=1.5
    )

plt.xlabel("Frame")
plt.ylabel("YOLO–Kalman separation (pixels)")
plt.title("YOLO–Kalman Separation vs Frame")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

separation_path = os.path.join(
    OUTPUT_DIR,
    "decay_separation_comparison.png"
)

plt.savefig(separation_path, dpi=200)
plt.show()

print(f"Saved: {separation_path}")


# ============================================================
# PLOT 3
# KALMAN VELOCITY
# ============================================================

plt.figure(figsize=(12, 6))

for decay in DECAY_VALUES:

    result = all_results[decay]

    speed = np.sqrt(
        result["kalman_vx"] ** 2 +
        result["kalman_vy"] ** 2
    )

    plt.plot(
        result["frame"],
        speed,
        label=f"Decay={decay}",
        linewidth=1.5
    )

plt.xlabel("Frame")
plt.ylabel("Kalman speed (pixels/frame)")
plt.title("Kalman Estimated Speed vs Frame")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

velocity_path = os.path.join(
    OUTPUT_DIR,
    "decay_velocity_comparison.png"
)

plt.savefig(velocity_path, dpi=200)
plt.show()

print(f"Saved: {velocity_path}")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 65)
print("DONE")
print("=" * 65)

print(f"""
Generated:

1. {trajectory_path}
2. {separation_path}
3. {velocity_path}

Individual Kalman CSVs are inside:

{OUTPUT_DIR}/

Decay values tested:
{DECAY_VALUES}

Q = {Q}
R = {R}
""")