import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
 
 
def load_log(filename):
    rows = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
 
 
def split_into_throws(rows):
    """Group rows into separate throws based on contiguous throw_active=True blocks."""
    throws = []
    current = []
    for row in rows:
        active = row["throw_active"] == "True"
        if active:
            current.append(row)
        else:
            if current:
                throws.append(current)
                current = []
    if current:
        throws.append(current)
    return throws
 
 
def extract_arrays(throw_rows):
    """Pull out numeric arrays from a throw's rows, keeping only frames with a real detection."""
    t, raw_x, raw_y, kx, ky, kvx, kvy = [], [], [], [], [], [], []
    n_missed = 0
    for row in throw_rows:
        if row["raw_cx"] == "":
            n_missed += 1
            continue
        t.append(float(row["timestamp"]))
        raw_x.append(float(row["raw_cx"]))
        raw_y.append(float(row["raw_cy"]))
        kx.append(float(row["kalman_x"]))
        ky.append(float(row["kalman_y"]))
        kvx.append(float(row["kalman_vx"]))
        kvy.append(float(row["kalman_vy"]))
    return (np.array(t), np.array(raw_x), np.array(raw_y),
            np.array(kx), np.array(ky), np.array(kvx), np.array(kvy),
            n_missed, len(throw_rows))
 
 
def analyze_throw(throw_idx, throw_rows):
    t, raw_x, raw_y, kx, ky, kvx, kvy, n_missed, n_total = extract_arrays(throw_rows)
 
    if len(t) < 5:
        print(f"Throw {throw_idx}: too few detected frames ({len(t)}) to analyze, skipping.")
        return None
 
    detection_rate = 100 * (n_total - n_missed) / n_total
 
    # ---- Fit a parabola to raw x,y vs time as our best physics-informed estimate
    #      of the "true" trajectory (since we have no external ground truth) ----
    t_rel = t - t[0]
    x_fit_coeffs = np.polyfit(t_rel, raw_x, 1)   # x: roughly linear
    y_fit_coeffs = np.polyfit(t_rel, raw_y, 2)   # y: parabola (gravity + drag approx.)
    x_fit = np.polyval(x_fit_coeffs, t_rel)
    y_fit = np.polyval(y_fit_coeffs, t_rel)
 
    # ---- Error of raw detections vs the fitted curve ----
    raw_error = np.sqrt((raw_x - x_fit) ** 2 + (raw_y - y_fit) ** 2)
    raw_rmse = np.sqrt(np.mean(raw_error ** 2))
 
    # ---- Error of Kalman-smoothed estimate vs the fitted curve ----
    kalman_error = np.sqrt((kx - x_fit) ** 2 + (ky - y_fit) ** 2)
    kalman_rmse = np.sqrt(np.mean(kalman_error ** 2))
 
    # ---- Frame-to-frame jitter: how much position jumps frame to frame
    #      (a smoother trajectory should have lower jitter) ----
    raw_jitter = np.mean(np.sqrt(np.diff(raw_x) ** 2 + np.diff(raw_y) ** 2))
    kalman_jitter = np.mean(np.sqrt(np.diff(kx) ** 2 + np.diff(ky) ** 2))
 
    print(f"\n--- Throw {throw_idx} ---")
    print(f"Frames: {n_total} total, {n_missed} missed detections "
          f"(detection rate: {detection_rate:.1f}%)")
    print(f"RMSE vs fitted trajectory   -> raw: {raw_rmse:.2f} px | kalman: {kalman_rmse:.2f} px")
    print(f"Avg frame-to-frame jitter   -> raw: {raw_jitter:.2f} px | kalman: {kalman_jitter:.2f} px")
    if kalman_rmse < raw_rmse:
        print("-> Kalman filter is tracking closer to the fitted trajectory than raw detections.")
    else:
        print("-> Kalman filter is NOT beating raw detections here -- worth re-tuning noise values.")
 
    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Throw {throw_idx} analysis")
 
    # Trajectory in image space (y inverted since image y grows downward)
    ax = axes[0, 0]
    ax.plot(raw_x, -raw_y, 'yo-', label='raw detection', markersize=4, alpha=0.6)
    ax.plot(kx, -ky, 'r.-', label='kalman estimate', markersize=4)
    ax.plot(x_fit, -y_fit, 'g--', label='fitted trajectory', alpha=0.7)
    ax.set_title("Trajectory (image space)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("-y (px, inverted so up is up)")
    ax.legend()
 
    # x vs time
    ax = axes[0, 1]
    ax.plot(t_rel, raw_x, 'yo-', label='raw x', markersize=4, alpha=0.6)
    ax.plot(t_rel, kx, 'r.-', label='kalman x', markersize=4)
    ax.set_title("Horizontal position over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x (px)")
    ax.legend()
 
    # y vs time
    ax = axes[1, 0]
    ax.plot(t_rel, raw_y, 'yo-', label='raw y', markersize=4, alpha=0.6)
    ax.plot(t_rel, ky, 'r.-', label='kalman y', markersize=4)
    ax.plot(t_rel, y_fit, 'g--', label='fitted parabola', alpha=0.7)
    ax.set_title("Vertical position over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y (px, image coords)")
    ax.legend()
 
    # velocity over time
    ax = axes[1, 1]
    ax.plot(t_rel, kvx, label='kalman vx')
    ax.plot(t_rel, kvy, label='kalman vy')
    ax.set_title("Kalman-estimated velocity over time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("velocity (px/frame)")
    ax.legend()
 
    plt.tight_layout()
    out_name = f"throw_{throw_idx}_analysis.png"
    plt.savefig(out_name, dpi=120)
    print(f"Saved plot: {out_name}")
    plt.close(fig)
 
    return {
        "throw_idx": throw_idx,
        "detection_rate": detection_rate,
        "raw_rmse": raw_rmse,
        "kalman_rmse": kalman_rmse,
        "raw_jitter": raw_jitter,
        "kalman_jitter": kalman_jitter,
    }
 
 
def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_balloon_log.py <log_file.csv>")
        sys.exit(1)
 
    filename = sys.argv[1]
    rows = load_log(filename)
    throws = split_into_throws(rows)
    print(f"Loaded {len(rows)} rows, found {len(throws)} throw(s) marked with 't'.")
 
    all_results = []
    for i, throw_rows in enumerate(throws, start=1):
        result = analyze_throw(i, throw_rows)
        if result:
            all_results.append(result)
 
    if len(all_results) > 1:
        print("\n=== Summary across all throws ===")
        avg_detection_rate = np.mean([r["detection_rate"] for r in all_results])
        avg_kalman_rmse = np.mean([r["kalman_rmse"] for r in all_results])
        avg_raw_rmse = np.mean([r["raw_rmse"] for r in all_results])
        print(f"Average detection rate: {avg_detection_rate:.1f}%")
        print(f"Average RMSE -> raw: {avg_raw_rmse:.2f} px | kalman: {avg_kalman_rmse:.2f} px")
 
 
if __name__ == "__main__":
    main()
    