#!/usr/bin/env python3
"""
Offline person-mask / ORB-feature analysis for TUM RGB-D + ORB-SLAM3 logs.

Inputs
------
1. feature_points.csv generated from the modified FrameDrawer.cc
2. TUM RGB-D rgb/ directory containing timestamp-named PNG files

Outputs
-------
- person_feature_metrics.csv
    Per-frame summary statistics.
- person_feature_points.csv
    Original per-feature records augmented with on_person + matched RGB frame.
- event_summary.csv
    Summary over the experiment intervals used in the walking_xyz inspection.
- map_inliers_person_vs_background.png
- person_inlier_fraction.png
- acceptance_rate_person_vs_background.png
- optional person_feature_overlay.mp4

Important
---------
"person" is a semantic class, NOT a motion label. A seated person remains "person".
This is useful for studying the dynamic -> stationary transition, but the script
does not by itself decide whether a person is currently moving.
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO


EVENTS = [
    ("C_foreground",       2.0,  5.0),
    ("reference",          9.0, 11.0),
    ("C_blur",            14.5, 15.5),
    ("B_returns",         19.5, 20.5),
    ("sitting_transition",23.0, 25.0),
    ("seated",            25.0, 28.7),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Intersect ORB-SLAM3 feature locations with YOLO person masks."
    )
    p.add_argument(
        "--features",
        required=True,
        type=Path,
        help="Path to feature_points.csv",
    )
    p.add_argument(
        "--rgb-dir",
        required=True,
        type=Path,
        help="TUM RGB-D rgb directory containing timestamp-named PNG files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("person_feature_analysis"),
        help="Output directory (default: ./person_feature_analysis)",
    )
    p.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="Ultralytics segmentation model (default: yolo11n-seg.pt)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Person detection confidence threshold (default: 0.20)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size (default: 640)",
    )
    p.add_argument(
        "--device",
        default=None,
        help='Ultralytics device, e.g. "cpu", "0", "cuda". Default: auto',
    )
    p.add_argument(
        "--max-time-diff",
        type=float,
        default=0.020,
        help="Maximum feature/RGB timestamp difference in seconds (default: 0.020)",
    )
    p.add_argument(
        "--mask-dilate",
        type=int,
        default=2,
        help="Dilate union person mask by this many pixels (default: 2; use 0 to disable)",
    )
    p.add_argument(
        "--save-overlay-video",
        action="store_true",
        help="Write a diagnostic MP4 showing person masks and ORB feature statuses",
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="FPS for optional diagnostic video (default: 30)",
    )
    return p.parse_args()


def load_rgb_index(rgb_dir: Path):
    rows = []
    for path in sorted(rgb_dir.glob("*.png")):
        try:
            ts = float(path.stem)
        except ValueError:
            continue
        rows.append((ts, path))

    if not rows:
        raise RuntimeError(f"No timestamp-named PNG images found in {rgb_dir}")

    rows.sort(key=lambda x: x[0])
    timestamps = np.array([r[0] for r in rows], dtype=np.float64)
    paths = [r[1] for r in rows]
    return timestamps, paths


def nearest_rgb(timestamp, rgb_timestamps, rgb_paths, max_diff):
    idx = int(np.searchsorted(rgb_timestamps, timestamp))

    candidates = []
    if idx < len(rgb_timestamps):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)

    if not candidates:
        return None, None, None

    best = min(candidates, key=lambda i: abs(rgb_timestamps[i] - timestamp))
    diff = abs(rgb_timestamps[best] - timestamp)

    if diff > max_diff:
        return None, None, diff

    return float(rgb_timestamps[best]), rgb_paths[best], diff


def union_person_mask(result, image_shape, dilate_pixels=0):
    h, w = image_shape[:2]

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return np.zeros((h, w), dtype=np.uint8), 0

    masks = result.masks.data.detach().cpu().numpy()
    union = np.any(masks > 0.5, axis=0).astype(np.uint8)

    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)

    if dilate_pixels > 0:
        k = 2 * dilate_pixels + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        union = cv2.dilate(union, kernel, iterations=1)

    n_instances = 0
    if result.boxes is not None:
        n_instances = len(result.boxes)

    return union, n_instances


def safe_ratio(num, den):
    return float(num) / float(den) if den else np.nan


def make_overlay(image, person_mask, frame_features):
    out = image.copy()

    # Semantic person mask overlay.
    overlay = out.copy()
    overlay[person_mask.astype(bool)] = (0, 255, 255)
    out = cv2.addWeighted(out, 0.72, overlay, 0.28, 0)

    status_style = {
        "orb_only":    ((160, 160, 160), 1),
        "map_inlier":  ((0, 255, 0),     3),
        "vo_inlier":   ((255, 0, 0),     3),
        "outlier":     ((0, 0, 255),     3),
    }

    h, w = out.shape[:2]
    for row in frame_features.itertuples(index=False):
        x = int(round(row.x))
        y = int(round(row.y))
        if not (0 <= x < w and 0 <= y < h):
            continue

        color, radius = status_style.get(row.status, ((255, 255, 255), 1))
        if row.status == "outlier":
            cv2.drawMarker(
                out, (x, y), color,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=7, thickness=1
            )
        else:
            cv2.circle(out, (x, y), radius, color, -1)

    return out


def event_summary(metrics):
    rows = []
    for label, t0, t1 in EVENTS:
        x = metrics[(metrics["time"] >= t0) & (metrics["time"] < t1)]
        if x.empty:
            continue

        rows.append({
            "event": label,
            "t_start": t0,
            "t_end": t1,
            "frames": len(x),
            "mean_orb_total": x["orb_total"].mean(),
            "mean_person_orb": x["person_orb"].mean(),
            "mean_map_inliers": x["map_inliers"].mean(),
            "mean_person_map_inliers": x["person_map_inliers"].mean(),
            "mean_background_map_inliers": x["background_map_inliers"].mean(),
            "mean_outliers": x["outliers"].mean(),
            "mean_person_outliers": x["person_outliers"].mean(),
            "mean_person_inlier_fraction": x["person_inlier_fraction"].mean(),
            "mean_person_acceptance_rate": x["person_acceptance_rate"].mean(),
            "mean_background_acceptance_rate": x["background_acceptance_rate"].mean(),
            "mean_person_area_ratio": x["person_area_ratio"].mean(),
        })
    return pd.DataFrame(rows)


def make_plots(metrics, out_dir):
    # 1. Accepted map features: person vs background.
    plt.figure(figsize=(11, 5))
    plt.plot(
        metrics["time"],
        metrics["person_map_inliers"],
        label="Person map inliers",
    )
    plt.plot(
        metrics["time"],
        metrics["background_map_inliers"],
        label="Background map inliers",
    )
    plt.xlabel("Time from start (s)")
    plt.ylabel("Accepted map features")
    plt.title("ORB-SLAM3 accepted map features by semantic region")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "map_inliers_person_vs_background.png", dpi=180)
    plt.close()

    # 2. Fraction of accepted map inliers lying on people.
    plt.figure(figsize=(11, 5))
    plt.plot(
        metrics["time"],
        metrics["person_inlier_fraction"],
    )
    plt.xlabel("Time from start (s)")
    plt.ylabel("Person map inliers / all map inliers")
    plt.title("Fraction of accepted ORB-SLAM3 map inliers on people")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "person_inlier_fraction.png", dpi=180)
    plt.close()

    # 3. Acceptance rates on person and background regions.
    plt.figure(figsize=(11, 5))
    plt.plot(
        metrics["time"],
        metrics["person_acceptance_rate"],
        label="Person-feature acceptance rate",
    )
    plt.plot(
        metrics["time"],
        metrics["background_acceptance_rate"],
        label="Background-feature acceptance rate",
    )
    plt.xlabel("Time from start (s)")
    plt.ylabel("Map inliers / detected ORB features")
    plt.title("ORB feature acceptance rate by semantic region")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acceptance_rate_person_vs_background.png", dpi=180)
    plt.close()


def main():
    args = parse_args()

    if not args.features.is_file():
        raise FileNotFoundError(args.features)
    if not args.rgb_dir.is_dir():
        raise NotADirectoryError(args.rgb_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features: {args.features}")
    features = pd.read_csv(args.features)

    required = {
        "time", "timestamp", "feature_id", "x", "y",
        "status", "map_point_id", "observations",
    }
    missing = required - set(features.columns)
    if missing:
        raise RuntimeError(
            "feature_points.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    features = features.sort_values(["timestamp", "feature_id"]).reset_index(drop=True)

    print(f"Feature rows: {len(features):,}")
    print(f"Frames: {features['timestamp'].nunique():,}")

    rgb_timestamps, rgb_paths = load_rgb_index(args.rgb_dir)
    print(f"RGB images indexed: {len(rgb_paths):,}")

    print(f"Loading segmentation model: {args.model}")
    model = YOLO(args.model)

    metrics_rows = []
    augmented_path = args.out_dir / "person_feature_points.csv"
    metrics_path = args.out_dir / "person_feature_metrics.csv"

    # Remove old augmented output so repeated runs do not append duplicates.
    if augmented_path.exists():
        augmented_path.unlink()

    video_writer = None

    grouped = features.groupby("timestamp", sort=True)

    for frame_index, (timestamp, frame) in enumerate(
        tqdm(grouped, total=features["timestamp"].nunique(), desc="Frames")
    ):
        elapsed_time = float(frame["time"].iloc[0])

        rgb_ts, rgb_path, timestamp_diff = nearest_rgb(
            float(timestamp),
            rgb_timestamps,
            rgb_paths,
            args.max_time_diff,
        )

        if rgb_path is None:
            print(
                f"\nWARNING: no RGB frame within {args.max_time_diff:.3f}s "
                f"of feature timestamp {timestamp:.6f}; nearest diff={timestamp_diff}",
                file=sys.stderr,
            )
            continue

        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"\nWARNING: failed to read {rgb_path}", file=sys.stderr)
            continue

        predict_kwargs = dict(
            source=image,
            classes=[0],          # COCO class 0 = person
            conf=args.conf,
            imgsz=args.imgsz,
            retina_masks=True,
            verbose=False,
        )
        if args.device is not None:
            predict_kwargs["device"] = args.device

        result = model.predict(**predict_kwargs)[0]
        person_mask, person_instances = union_person_mask(
            result,
            image.shape,
            args.mask_dilate,
        )

        h, w = image.shape[:2]

        xs = np.rint(frame["x"].to_numpy()).astype(np.int32)
        ys = np.rint(frame["y"].to_numpy()).astype(np.int32)

        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        on_person = np.zeros(len(frame), dtype=bool)
        on_person[valid] = person_mask[ys[valid], xs[valid]].astype(bool)

        frame_aug = frame.copy()
        frame_aug["rgb_timestamp"] = rgb_ts
        frame_aug["rgb_file"] = rgb_path.name
        frame_aug["rgb_time_diff"] = timestamp_diff
        frame_aug["on_person"] = on_person.astype(np.uint8)

        frame_aug.to_csv(
            augmented_path,
            mode="a",
            header=(frame_index == 0),
            index=False,
        )

        status = frame["status"].to_numpy()

        is_map = status == "map_inlier"
        is_vo = status == "vo_inlier"
        is_outlier = status == "outlier"

        orb_total = len(frame)
        person_orb = int(on_person.sum())
        background_orb = orb_total - person_orb

        map_inliers = int(is_map.sum())
        person_map_inliers = int((is_map & on_person).sum())
        background_map_inliers = int((is_map & ~on_person).sum())

        vo_inliers = int(is_vo.sum())
        person_vo_inliers = int((is_vo & on_person).sum())

        outliers = int(is_outlier.sum())
        person_outliers = int((is_outlier & on_person).sum())
        background_outliers = int((is_outlier & ~on_person).sum())

        metrics_rows.append({
            "time": elapsed_time,
            "timestamp": float(timestamp),
            "rgb_timestamp": rgb_ts,
            "rgb_time_diff": timestamp_diff,
            "rgb_file": rgb_path.name,
            "person_instances": person_instances,
            "person_area_ratio": float(person_mask.mean()),

            "orb_total": orb_total,
            "person_orb": person_orb,
            "background_orb": background_orb,

            "map_inliers": map_inliers,
            "person_map_inliers": person_map_inliers,
            "background_map_inliers": background_map_inliers,

            "vo_inliers": vo_inliers,
            "person_vo_inliers": person_vo_inliers,

            "outliers": outliers,
            "person_outliers": person_outliers,
            "background_outliers": background_outliers,

            # Of all accepted established-map features, how many lie on a person?
            "person_inlier_fraction": safe_ratio(
                person_map_inliers, map_inliers
            ),

            # Of all ORB keypoints detected on a person, how many are accepted
            # as established map-point matches?
            "person_acceptance_rate": safe_ratio(
                person_map_inliers, person_orb
            ),

            # Equivalent acceptance rate for the non-person region.
            "background_acceptance_rate": safe_ratio(
                background_map_inliers, background_orb
            ),

            # How much of the image's ORB feature budget lies on people?
            "person_orb_fraction": safe_ratio(
                person_orb, orb_total
            ),
        })

        if args.save_overlay_video:
            overlay = make_overlay(image, person_mask, frame_aug)

            text = (
                f"t={elapsed_time:.2f}s  "
                f"person map={person_map_inliers}/{map_inliers}  "
                f"person ORB={person_orb}/{orb_total}"
            )
            cv2.putText(
                overlay,
                text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(args.out_dir / "person_feature_overlay.mp4"),
                    fourcc,
                    args.video_fps,
                    (w, h),
                )

            video_writer.write(overlay)

    if video_writer is not None:
        video_writer.release()

    metrics = pd.DataFrame(metrics_rows).sort_values("time")
    metrics.to_csv(metrics_path, index=False)

    summary = event_summary(metrics)
    summary.to_csv(args.out_dir / "event_summary.csv", index=False)

    make_plots(metrics, args.out_dir)

    print()
    print("Done.")
    print(f"Per-frame metrics:       {metrics_path}")
    print(f"Per-feature augmented:   {augmented_path}")
    print(f"Event summary:           {args.out_dir / 'event_summary.csv'}")
    print(f"Plots:                   {args.out_dir}/*.png")
    if args.save_overlay_video:
        print(f"Diagnostic video:        {args.out_dir / 'person_feature_overlay.mp4'}")

    if not summary.empty:
        print()
        print("Event summary:")
        cols = [
            "event",
            "frames",
            "mean_person_orb",
            "mean_person_map_inliers",
            "mean_person_inlier_fraction",
            "mean_person_acceptance_rate",
            "mean_background_acceptance_rate",
        ]
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
