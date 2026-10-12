import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "kalman_data.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

# Only use frames where both YOLO and Kalman exist
valid = df[
    df["yolo_detected"] &
    df["yolo_x"].notna() &
    df["yolo_y"].notna() &
    df["kalman_x"].notna() &
    df["kalman_y"].notna()
].copy()


if len(valid) == 0:
    print("No valid frames found.")
    exit()


# ============================================================
# YOLO ↔ KALMAN ERROR
# ============================================================

valid["error_x"] = (
    valid["yolo_x"] - valid["kalman_x"]
)

valid["error_y"] = (
    valid["yolo_y"] - valid["kalman_y"]
)

valid["distance"] = np.sqrt(
    valid["error_x"]**2 +
    valid["error_y"]**2
)


# ============================================================
# RESULTS
# ============================================================

print()
print("================================================")
print("KALMAN FILTER ANALYSIS")
print("================================================")

print()
print(f"Valid frames: {len(valid)}")

print()
print("YOLO ↔ KALMAN SEPARATION")
print("--------------------------------")

print(
    f"Mean distance   : "
    f"{valid['distance'].mean():.2f} px"
)

print(
    f"Median distance : "
    f"{valid['distance'].median():.2f} px"
)

print(
    f"Maximum distance: "
    f"{valid['distance'].max():.2f} px"
)

print(
    f"Mean X distance : "
    f"{valid['error_x'].abs().mean():.2f} px"
)

print(
    f"Mean Y distance : "
    f"{valid['error_y'].abs().mean():.2f} px"
)


# ============================================================
# JITTER
# ============================================================

# Frame-to-frame movement of YOLO

yolo_dx = valid["yolo_x"].diff()
yolo_dy = valid["yolo_y"].diff()

yolo_jitter = np.sqrt(
    yolo_dx**2 +
    yolo_dy**2
).dropna()


# Frame-to-frame movement of Kalman

kalman_dx = valid["kalman_x"].diff()
kalman_dy = valid["kalman_y"].diff()

kalman_jitter = np.sqrt(
    kalman_dx**2 +
    kalman_dy**2
).dropna()


print()
print("TRAJECTORY SMOOTHNESS")
print("--------------------------------")

print(
    f"YOLO mean frame movement    : "
    f"{yolo_jitter.mean():.2f} px"
)

print(
    f"Kalman mean frame movement  : "
    f"{kalman_jitter.mean():.2f} px"
)


# ============================================================
# VELOCITY
# ============================================================

print()
print("KALMAN VELOCITY")
print("--------------------------------")

print(
    f"Average VX : "
    f"{valid['vx'].mean():.2f} px/s"
)

print(
    f"Average VY : "
    f"{valid['vy'].mean():.2f} px/s"
)

print(
    f"Maximum speed : "
    f"{valid['speed'].max():.2f} px/s"
)


# ============================================================
# PLOT X POSITION
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    valid["frame"],
    valid["yolo_x"],
    label="YOLO X"
)

plt.plot(
    valid["frame"],
    valid["kalman_x"],
    label="Kalman X"
)

plt.xlabel("Frame")
plt.ylabel("X position (pixels)")
plt.title("YOLO vs Kalman — X Position")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# PLOT Y POSITION
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    valid["frame"],
    valid["yolo_y"],
    label="YOLO Y"
)

plt.plot(
    valid["frame"],
    valid["kalman_y"],
    label="Kalman Y"
)

plt.xlabel("Frame")
plt.ylabel("Y position (pixels)")
plt.title("YOLO vs Kalman — Y Position")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# PLOT TRAJECTORY
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    valid["yolo_x"],
    valid["yolo_y"],
    label="YOLO"
)

plt.plot(
    valid["kalman_x"],
    valid["kalman_y"],
    label="Kalman"
)

plt.xlabel("X position (pixels)")
plt.ylabel("Y position (pixels)")

plt.title("Balloon Trajectory")

plt.legend()
plt.grid(True)

# Camera coordinates: y increases downward
plt.gca().invert_yaxis()

plt.show()


print()
print("================================================")
print("ANALYSIS COMPLETE")
print("================================================")