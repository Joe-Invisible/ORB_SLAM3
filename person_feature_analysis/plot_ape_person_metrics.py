#!/usr/bin/env python3
"""
Plot ORB-SLAM3 APE together with person-feature metrics on a common time axis.

Requires:
    pip install evo pandas matplotlib numpy

Example:
    python plot_ape_person_metrics.py \
        --gt "$TUMRGBD/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt" \
        --traj "$RESULT/CameraTrajectory.txt" \
        --metrics "$RESULT/person_feature_analysis/person_feature_metrics.csv" \
        --out "$RESULT/person_feature_analysis/ape_person_metrics.png"

The trajectory and feature metrics MUST come from the same ORB-SLAM3 run.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evo.core import metrics, sync
from evo.core.metrics import PoseRelation
from evo.tools import file_interface


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, type=Path,
                   help="TUM RGB-D groundtruth.txt")
    p.add_argument("--traj", required=True, type=Path,
                   help="ORB-SLAM3 CameraTrajectory.txt from the SAME run")
    p.add_argument("--metrics", required=True, type=Path,
                   help="person_feature_metrics.csv from the SAME run")
    p.add_argument("--out", required=True, type=Path,
                   help="Output PNG")
    p.add_argument("--max-diff", type=float, default=0.01,
                   help="Max timestamp association difference in seconds")
    return p.parse_args()


def main():
    args = parse_args()

    # Load TUM-format trajectories.
    traj_ref = file_interface.read_tum_trajectory_file(str(args.gt))
    traj_est = file_interface.read_tum_trajectory_file(str(args.traj))

    # Associate timestamps exactly as evo does.
    traj_ref, traj_est = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=args.max_diff
    )

    # Match `evo_ape tum GT EST -a`:
    # SE(3) Umeyama alignment, no scale correction.
    traj_est.align(traj_ref, correct_scale=False)

    ape_metric = metrics.APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    ape = np.asarray(ape_metric.error, dtype=float)

    # The timestamps after association are the estimate timestamps.
    # Use elapsed time from the first associated pose so the x-axis matches
    # the visual inspection convention.
    t_ape = np.asarray(traj_est.timestamps, dtype=float)
    t_ape = t_ape - t_ape[0]

    df = pd.read_csv(args.metrics).sort_values("time")
    t = df["time"].to_numpy(dtype=float)

    # Main semantic-feature quantities.
    person_map = df["person_map_inliers"].to_numpy(dtype=float)
    bg_map = df["background_map_inliers"].to_numpy(dtype=float)
    person_frac = df["person_inlier_fraction"].to_numpy(dtype=float)

    # --- Figure: three vertically aligned panels ---
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]}
    )

    # 1) APE
    axes[0].plot(t_ape, ape, linewidth=1.2)
    axes[0].set_ylabel("APE (m)")
    axes[0].set_title("Trajectory error and accepted person-feature support")

    # 2) Accepted map features
    axes[1].plot(t, person_map, label="Person map inliers")
    axes[1].plot(t, bg_map, label="Background map inliers")
    axes[1].set_ylabel("Accepted map features")
    axes[1].legend(loc="upper right")

    # 3) Person share of accepted map inliers
    axes[2].plot(t, person_frac)
    axes[2].set_ylabel("Person inlier fraction")
    axes[2].set_xlabel("Elapsed time (s)")
    axes[2].set_ylim(0, 1)

    # Mark the intervals already used in the experiment.
    events = [
        ("C foreground", 2.0, 5.0),
        ("C + blur", 14.5, 15.5),
        ("B returns", 19.5, 20.5),
        ("sitting transition", 23.0, 25.0),
        ("seated", 25.0, 28.7),
    ]
    for ax in axes:
        for _, a, b in events:
            ax.axvspan(a, b, alpha=0.08)

    # Put labels only on the top panel to avoid clutter.
    ymax = axes[0].get_ylim()[1]
    for label, a, b in events:
        axes[0].text(
            (a + b) / 2, ymax * 0.96, label,
            ha="center", va="top", fontsize=8, rotation=90
        )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")

    # Also save a time-aligned CSV for later statistics/correlation.
    # Interpolate semantic metrics to APE timestamps.
    valid = np.isfinite(person_frac)
    aligned = pd.DataFrame({
        "time": t_ape,
        "ape_m": ape,
        "person_map_inliers": np.interp(t_ape, t, person_map),
        "background_map_inliers": np.interp(t_ape, t, bg_map),
        "person_inlier_fraction": np.interp(
            t_ape, t[valid], person_frac[valid]
        ),
    })

    csv_out = args.out.with_suffix(".csv")
    aligned.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    # Simple descriptive correlations. Treat these as exploratory, because
    # adjacent video frames are temporally correlated and APE is cumulative.
    print()
    print("Exploratory Pearson correlations:")
    print(aligned[[
        "ape_m",
        "person_map_inliers",
        "person_inlier_fraction"
    ]].corr()["ape_m"])


if __name__ == "__main__":
    main()
