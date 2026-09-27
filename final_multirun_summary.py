#!/usr/bin/env python3
"""
Create a final summary figure across multiple inspection runs, combining:

1. APE over time
2. Person vs background geometric residual over time
3. Moving-person vs static-like-person map-inlier acceptance over time

Dependencies:
    pip install evo pandas matplotlib numpy

Example:
    source ~/evo-env/bin/activate

    ROOT="$ORB/results/tum_rgbd_dynamic/inspection_runs"
    SEQ="$TUMRGBD/rgbd_dataset_freiburg3_walking_xyz"

    python final_multirun_summary.py \
        --root "$ROOT" \
        --gt "$SEQ/groundtruth.txt" \
        --out "$ROOT/final_multirun_summary.png"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evo.core import metrics, sync
from evo.core.metrics import PoseRelation
from evo.tools import file_interface


EVENTS = [
    ("C foreground", 2.0, 5.0),
    ("C + blur", 14.5, 15.5),
    ("B returns", 19.5, 20.5),
    ("Sitting transition", 23.0, 25.0),
    ("Seated", 25.0, 28.7),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path,
                   help="Directory containing inspection_run1, inspection_run2, ...")
    p.add_argument("--gt", required=True, type=Path,
                   help="TUM groundtruth.txt")
    p.add_argument("--out", required=True, type=Path,
                   help="Output PNG path")
    p.add_argument("--max-diff", type=float, default=0.01,
                   help="APE timestamp association max diff")
    p.add_argument("--dt", type=float, default=0.05,
                   help="Common time-grid step in seconds")
    p.add_argument("--rolling-window", type=int, default=9,
                   help="Centered rolling-median window (frames) for frame metrics")
    return p.parse_args()


def load_ape_series(gt_path: Path, traj_path: Path, max_diff: float):
    traj_ref = file_interface.read_tum_trajectory_file(str(gt_path))
    traj_est = file_interface.read_tum_trajectory_file(str(traj_path))
    traj_ref, traj_est = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=max_diff
    )
    traj_est.align(traj_ref, correct_scale=False)

    ape_metric = metrics.APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))

    ape = np.asarray(ape_metric.error, dtype=float)
    t = np.asarray(traj_est.timestamps, dtype=float)
    t = t - t[0]
    return t, ape


def nan_interp(grid, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if len(x) == 1:
        out = np.full_like(grid, np.nan, dtype=float)
        idx = np.argmin(np.abs(grid - x[0]))
        out[idx] = y[0]
        return out

    out = np.interp(grid, x, y, left=np.nan, right=np.nan)
    out[grid < x[0]] = np.nan
    out[grid > x[-1]] = np.nan
    return out


def rolling_median(series, window):
    return pd.Series(series).rolling(window, center=True, min_periods=1).median().to_numpy()


def add_event_shading(ax):
    for _, a, b in EVENTS:
        ax.axvspan(a, b, alpha=0.07)


def main():
    args = parse_args()

    run_dirs = sorted([
        p for p in args.root.iterdir()
        if p.is_dir() and p.name.startswith("inspection_run")
    ])
    if not run_dirs:
        raise RuntimeError(f"No inspection_run* directories found under {args.root}")

    runs = []
    max_time = 0.0

    print("Loading runs...")
    for run_dir in run_dirs:
        traj = run_dir / "CameraTrajectory.txt"
        frame_csv = run_dir / "geometric_motion_analysis" / "frame_motion_metrics.csv"

        if not traj.is_file():
            print(f"Skipping {run_dir.name}: missing {traj}")
            continue
        if not frame_csv.is_file():
            print(f"Skipping {run_dir.name}: missing {frame_csv}")
            continue

        t_ape, ape = load_ape_series(args.gt, traj, args.max_diff)
        frame = pd.read_csv(frame_csv).sort_values("time")

        t_frame = frame["time"].to_numpy(dtype=float)
        person_res = rolling_median(frame["person_residual_median_px"], args.rolling_window)
        bg_res = rolling_median(frame["background_residual_median_px"], args.rolling_window)
        moving_acc = pd.Series(frame["moving_person_map_acceptance_rate"]).rolling(
            args.rolling-window if False else args.rolling_window, center=True, min_periods=1
        ).mean().to_numpy()
        static_acc = pd.Series(frame["static_person_map_acceptance_rate"]).rolling(
            args.rolling_window, center=True, min_periods=1
        ).mean().to_numpy()

        max_time = max(max_time, float(np.nanmax(t_ape)), float(np.nanmax(t_frame)))

        runs.append({
            "name": run_dir.name,
            "t_ape": t_ape,
            "ape": ape,
            "t_frame": t_frame,
            "person_res": person_res,
            "bg_res": bg_res,
            "moving_acc": moving_acc,
            "static_acc": static_acc,
            "rmse": float(np.sqrt(np.mean(ape**2))),
            "peak_ape": float(np.nanmax(ape)),
            "peak_time": float(t_ape[np.nanargmax(ape)]),
        })

        print(f"  {run_dir.name}: rmse={runs[-1]['rmse']:.3f} m, "
              f"peak={runs[-1]['peak_ape']:.3f} m at {runs[-1]['peak_time']:.2f}s")

    if not runs:
        raise RuntimeError("No valid run directories were loaded.")

    grid = np.arange(0.0, max_time + args.dt / 2, args.dt, dtype=float)

    ape_stack = []
    person_res_stack = []
    bg_res_stack = []
    moving_acc_stack = []
    static_acc_stack = []

    for r in runs:
        ape_stack.append(nan_interp(grid, r["t_ape"], r["ape"]))
        person_res_stack.append(nan_interp(grid, r["t_frame"], r["person_res"]))
        bg_res_stack.append(nan_interp(grid, r["t_frame"], r["bg_res"]))
        moving_acc_stack.append(nan_interp(grid, r["t_frame"], r["moving_acc"]))
        static_acc_stack.append(nan_interp(grid, r["t_frame"], r["static_acc"]))

    ape_stack = np.vstack(ape_stack)
    person_res_stack = np.vstack(person_res_stack)
    bg_res_stack = np.vstack(bg_res_stack)
    moving_acc_stack = np.vstack(moving_acc_stack)
    static_acc_stack = np.vstack(static_acc_stack)

    def mean_std(stack):
        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)

    ape_mean, ape_std = mean_std(ape_stack)
    person_res_mean, person_res_std = mean_std(person_res_stack)
    bg_res_mean, bg_res_std = mean_std(bg_res_stack)
    moving_acc_mean, moving_acc_std = mean_std(moving_acc_stack)
    static_acc_mean, static_acc_std = mean_std(static_acc_stack)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    out_csv = args.out.with_suffix(".csv")
    aligned = pd.DataFrame({
        "time": grid,
        "ape_mean_m": ape_mean,
        "ape_std_m": ape_std,
        "person_residual_mean_px": person_res_mean,
        "person_residual_std_px": person_res_std,
        "background_residual_mean_px": bg_res_mean,
        "background_residual_std_px": bg_res_std,
        "moving_person_acceptance_mean": moving_acc_mean,
        "moving_person_acceptance_std": moving_acc_std,
        "static_person_acceptance_mean": static_acc_mean,
        "static_person_acceptance_std": static_acc_std,
    })
    aligned.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    for r in runs:
        axes[0].plot(r["t_ape"], r["ape"], alpha=0.35, linewidth=1.0)
    axes[0].plot(grid, ape_mean, linewidth=2.0, label="Mean APE")
    axes[0].fill_between(grid, ape_mean - ape_std, ape_mean + ape_std, alpha=0.2)
    axes[0].set_ylabel("APE (m)")
    axes[0].set_title("Final multirun summary: APE, geometric residual, and acceptance")

    for r in runs:
        axes[1].plot(r["t_frame"], r["person_res"], alpha=0.22, linewidth=1.0)
        axes[1].plot(r["t_frame"], r["bg_res"], alpha=0.18, linewidth=1.0, linestyle="--")
    axes[1].plot(grid, person_res_mean, linewidth=2.0, label="Person residual")
    axes[1].fill_between(grid, person_res_mean - person_res_std, person_res_mean + person_res_std, alpha=0.18)
    axes[1].plot(grid, bg_res_mean, linewidth=2.0, linestyle="--", label="Background residual")
    axes[1].fill_between(grid, bg_res_mean - bg_res_std, bg_res_mean + bg_res_std, alpha=0.12)
    axes[1].set_ylabel("Median residual (px)")
    axes[1].legend(loc="upper right")

    for r in runs:
        axes[2].plot(r["t_frame"], r["moving_acc"], alpha=0.22, linewidth=1.0)
        axes[2].plot(r["t_frame"], r["static_acc"], alpha=0.18, linewidth=1.0, linestyle="--")
    axes[2].plot(grid, moving_acc_mean, linewidth=2.0, label="Moving-person acceptance")
    axes[2].fill_between(grid, moving_acc_mean - moving_acc_std, moving_acc_mean + moving_acc_std, alpha=0.18)
    axes[2].plot(grid, static_acc_mean, linewidth=2.0, linestyle="--",
                 label="Static-like-person acceptance")
    axes[2].fill_between(grid, static_acc_mean - static_acc_std, static_acc_mean + static_acc_std, alpha=0.12)
    axes[2].set_ylabel("Map-inlier acceptance")
    axes[2].set_xlabel("Elapsed time (s)")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right")

    for ax in axes:
        add_event_shading(ax)

    ymax = np.nanmax(ape_mean + ape_std)
    for label, a, b in EVENTS:
        axes[0].text((a + b) / 2, ymax * 0.98, label,
                     ha="center", va="top", fontsize=8, rotation=90)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Saved figure: {args.out}")
    print(f"Saved aligned data: {out_csv}")

    out_stats = args.out.with_suffix(".txt")
    with out_stats.open("w") as f:
        f.write("Per-run APE summary\n")
        f.write("-------------------\n")
        for r in runs:
            f.write(
                f"{r['name']}: rmse={r['rmse']:.6f} m, "
                f"peak={r['peak_ape']:.6f} m at {r['peak_time']:.3f} s\n"
            )

        f.write("\nCross-run averages\n")
        f.write("------------------\n")
        f.write(f"Mean RMSE: {np.mean([r['rmse'] for r in runs]):.6f} m\n")
        f.write(f"Std RMSE:  {np.std([r['rmse'] for r in runs]):.6f} m\n")
        f.write(f"Mean peak APE: {np.mean([r['peak_ape'] for r in runs]):.6f} m\n")
        f.write(f"Mean peak time: {np.mean([r['peak_time'] for r in runs]):.6f} s\n")

        f.write("\nEvent summaries from mean curves\n")
        f.write("-------------------------------\n")
        for label, a, b in EVENTS:
            mask = (grid >= a) & (grid < b)
            if not np.any(mask):
                continue
            f.write(
                f"{label}: "
                f"APE={np.nanmean(ape_mean[mask]):.6f} m, "
                f"person_res={np.nanmean(person_res_mean[mask]):.6f} px, "
                f"bg_res={np.nanmean(bg_res_mean[mask]):.6f} px, "
                f"moving_accept={np.nanmean(moving_acc_mean[mask]):.6f}, "
                f"static_accept={np.nanmean(static_acc_mean[mask]):.6f}\n"
            )

    print(f"Saved summary stats: {out_stats}")


if __name__ == "__main__":
    main()
