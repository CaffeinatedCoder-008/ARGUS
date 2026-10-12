import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
# ============================================================

CSV_FILE = "hand_tracking_data.csv"

df = pd.read_csv(CSV_FILE)


# Only frames where YOLO + Kalman produced valid data

valid = df[
    df["detected"] &
    df["yolo_x"].notna() &
    df["kalman_x"].notna()
].copy()


print("\n")
print("=" * 55)
print("HAND-HELD BALLOON — YOLO + KALMAN ANALYSIS")
print("=" * 55)


# ============================================================
# DETECTION RATE
# ============================================================

total_frames = len(df)

detected_frames = df["detected"].sum()

detection_rate = (
    detected_frames / total_frames * 100
    if total_frames > 0
    else 0
)


print("\nDETECTION")
print("-" * 55)

print(f"Total frames       : {total_frames}")
print(f"Detected frames    : {detected_frames}")
print(f"Detection rate     : {detection_rate:.2f}%")


# ============================================================
# YOLO FRAME MOVEMENT
# ============================================================

yolo_steps = valid["yolo_step"].dropna()

print("\nYOLO MOVEMENT")
print("-" * 55)

print(
    f"Mean frame movement   : "
    f"{yolo_steps.mean():.2f} px"
)

print(
    f"Median frame movement : "
    f"{yolo_steps.median():.2f} px"
)

print(
    f"Maximum movement      : "
    f"{yolo_steps.max():.2f} px"
)


# ============================================================
# KALMAN FRAME MOVEMENT
# ============================================================

kalman_steps = valid["kalman_step"].dropna()

print("\nKALMAN MOVEMENT")
print("-" * 55)

print(
    f"Mean frame movement   : "
    f"{kalman_steps.mean():.2f} px"
)

print(
    f"Median frame movement : "
    f"{kalman_steps.median():.2f} px"
)

print(
    f"Maximum movement      : "
    f"{kalman_steps.max():.2f} px"
)


# ============================================================
# SMOOTHNESS IMPROVEMENT
# ============================================================

if yolo_steps.mean() > 0:

    smoothness_improvement = (
        1 -
        kalman_steps.mean() /
        yolo_steps.mean()
    ) * 100

else:

    smoothness_improvement = 0


print("\nSMOOTHNESS")
print("-" * 55)

print(
    f"Movement reduction : "
    f"{smoothness_improvement:.2f}%"
)


# ============================================================
# YOLO ↔ KALMAN SEPARATION
# ============================================================

separation = valid[
    "yolo_kalman_distance"
].dropna()


x_sep = valid[
    "x_separation"
].dropna()


y_sep = valid[
    "y_separation"
].dropna()


print("\nYOLO ↔ KALMAN SEPARATION")
print("-" * 55)

print(
    f"Mean distance   : "
    f"{separation.mean():.2f} px"
)

print(
    f"Median distance : "
    f"{separation.median():.2f} px"
)

print(
    f"Maximum distance: "
    f"{separation.max():.2f} px"
)

print(
    f"Mean X distance : "
    f"{x_sep.mean():.2f} px"
)

print(
    f"Mean Y distance : "
    f"{y_sep.mean():.2f} px"
)


# ============================================================
# VELOCITY
# ============================================================

vx = valid["kalman_vx"].dropna()

vy = valid["kalman_vy"].dropna()


speed = np.sqrt(
    vx ** 2 +
    vy ** 2
)


print("\nKALMAN VELOCITY")
print("-" * 55)

print(
    f"Average VX      : "
    f"{vx.mean():.2f} px/frame"
)

print(
    f"Average VY      : "
    f"{vy.mean():.2f} px/frame"
)

print(
    f"Average speed   : "
    f"{speed.mean():.2f} px/frame"
)

print(
    f"Maximum speed   : "
    f"{speed.max():.2f} px/frame"
)


# ============================================================
# HIGH-MOTION FRAMES
# ============================================================

if len(yolo_steps) > 0:

    threshold = yolo_steps.quantile(0.90)

    high_motion = valid[
        valid["yolo_step"] >= threshold
    ]


    print("\nHIGH-MOTION FRAMES")
    print("-" * 55)

    print(
        f"90th percentile YOLO movement : "
        f"{threshold:.2f} px/frame"
    )

    print(
        f"Frames in high-motion region  : "
        f"{len(high_motion)}"
    )

    if len(high_motion) > 0:

        print(
            f"Mean Kalman separation there : "
            f"{high_motion['yolo_kalman_distance'].mean():.2f} px"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 55)
print("SUMMARY")
print("=" * 55)

print(
    f"Detection rate       : {detection_rate:.2f}%"
)

print(
    f"YOLO movement        : {yolo_steps.mean():.2f} px/frame"
)

print(
    f"Kalman movement      : {kalman_steps.mean():.2f} px/frame"
)

print(
    f"Smoothness reduction : {smoothness_improvement:.2f}%"
)

print(
    f"YOLO-Kalman distance : {separation.mean():.2f} px"
)

print(
    f"Average speed        : {speed.mean():.2f} px/frame"
)

print("=" * 55)