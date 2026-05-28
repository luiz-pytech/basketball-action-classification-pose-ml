import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

KP = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}

SELECTED_KEYPOINTS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

MOTION_KEYPOINTS = [
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
]

ANGLE_DEFINITIONS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "torso_tilt_abs": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
}

SERIES_STATS = ["mean", "std", "min", "max", "range", "first_to_last_delta"]
DEFAULT_FRAMES = [0, 4, 8, 12, 15]
CONF_THRESHOLD = 0.25
RANDOM_STATE = 42
TEST_SIZE = 0.20
EPS = 1e-8
POSTURE_FEATURE_COUNT = 13


def find_root_dir():
    root_dir = Path.cwd().resolve()
    while root_dir != root_dir.parent and not (root_dir / "data").exists():
        root_dir = root_dir.parent
    return root_dir


def resolve_path(root_dir, path_value):
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return root_dir / path


def finite_rows(values):
    values = np.asarray(values, dtype=float)
    return np.isfinite(values).all(axis=1)


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def safe_stat(values, func):
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    return float(func(values))


def series_range(values):
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.max(values) - np.min(values))


def first_to_last_delta(values):
    values = np.asarray(values, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if valid_idx.size == 0:
        return np.nan
    if valid_idx.size == 1:
        return 0.0
    return float(values[valid_idx[-1]] - values[valid_idx[0]])


def add_series_stats(features, prefix, values):
    values = np.asarray(values, dtype=float)
    features[f"{prefix}_mean"] = safe_stat(values, np.mean)
    features[f"{prefix}_std"] = safe_stat(values, np.std)
    features[f"{prefix}_min"] = safe_stat(values, np.min)
    features[f"{prefix}_max"] = safe_stat(values, np.max)
    features[f"{prefix}_range"] = series_range(values)
    features[f"{prefix}_first_to_last_delta"] = first_to_last_delta(values)


def pair_distance(frame_coords, point_a, point_b):
    a = frame_coords[KP[point_a]]
    b = frame_coords[KP[point_b]]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    return float(np.linalg.norm(a - b))


def midpoint(frame_coords, point_a, point_b):
    a = frame_coords[KP[point_a]]
    b = frame_coords[KP[point_b]]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return (a + b) / 2.0


def bbox_scale(frame_coords, keypoints):
    indexes = [KP[name] for name in keypoints]
    values = frame_coords[indexes]
    valid = finite_rows(values)
    if valid.sum() < 2:
        return np.nan

    points = values[valid]
    width = np.nanmax(points[:, 0]) - np.nanmin(points[:, 0])
    height = np.nanmax(points[:, 1]) - np.nanmin(points[:, 1])
    scale = np.sqrt(width**2 + height**2)
    if not np.isfinite(scale) or scale <= EPS:
        return np.nan
    return float(scale)


def frame_body_center(frame_coords):
    hip_center = midpoint(frame_coords, "left_hip", "right_hip")
    if hip_center is not None:
        return hip_center

    shoulder_center = midpoint(frame_coords, "left_shoulder", "right_shoulder")
    if shoulder_center is not None:
        return shoulder_center

    torso_points = frame_coords[
        [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
    ]
    valid = finite_rows(torso_points)
    if valid.sum() >= 2:
        return np.nanmean(torso_points[valid], axis=0)

    selected_points = frame_coords[[KP[name] for name in SELECTED_KEYPOINTS]]
    valid = finite_rows(selected_points)
    if valid.sum() >= 2:
        return np.nanmean(selected_points[valid], axis=0)

    return np.array([np.nan, np.nan], dtype=float)


def frame_body_scale(frame_coords):
    hip_center = midpoint(frame_coords, "left_hip", "right_hip")
    shoulder_center = midpoint(frame_coords, "left_shoulder", "right_shoulder")
    if hip_center is not None and shoulder_center is not None:
        torso_scale = float(np.linalg.norm(shoulder_center - hip_center))
        if np.isfinite(torso_scale) and torso_scale > EPS:
            return torso_scale

    shoulder_scale = pair_distance(frame_coords, "left_shoulder", "right_shoulder")
    if np.isfinite(shoulder_scale) and shoulder_scale > EPS:
        return shoulder_scale

    hip_scale = pair_distance(frame_coords, "left_hip", "right_hip")
    if np.isfinite(hip_scale) and hip_scale > EPS:
        return hip_scale

    torso_scale = bbox_scale(
        frame_coords,
        ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    )
    if np.isfinite(torso_scale) and torso_scale > EPS:
        return torso_scale

    return bbox_scale(frame_coords, SELECTED_KEYPOINTS)


def interpolate_2d(values):
    values = np.asarray(values, dtype=float).copy()
    frame_indexes = np.arange(values.shape[0])

    for col in range(values.shape[1]):
        col_values = values[:, col]
        valid = np.isfinite(col_values)
        if valid.sum() >= 2:
            values[:, col] = np.interp(frame_indexes, frame_indexes[valid], col_values[valid])
        elif valid.sum() == 1:
            values[:, col] = col_values[valid][0]

    return values


def clean_pose(pose, conf_threshold):
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 3 or pose.shape[1] < len(COCO_KEYPOINTS) or pose.shape[2] < 3:
        raise ValueError(f"shape inesperado para pose: {pose.shape}")

    coords = pose[:, : len(COCO_KEYPOINTS), :2].astype(float)
    confidence = pose[:, : len(COCO_KEYPOINTS), 2].astype(float)
    coords[(~np.isfinite(confidence)) | (confidence < conf_threshold)] = np.nan
    return coords, confidence


def normalize_pose(pose, conf_threshold):
    coords, confidence = clean_pose(pose, conf_threshold)

    centers = np.stack([frame_body_center(frame) for frame in coords])
    centers = interpolate_2d(centers)

    scales = np.array([frame_body_scale(frame) for frame in coords], dtype=float)
    valid_scales = scales[np.isfinite(scales) & (scales > EPS)]
    fallback_scale = float(np.median(valid_scales)) if valid_scales.size else 1.0
    scales = np.where(np.isfinite(scales) & (scales > EPS), scales, fallback_scale)

    normalized = (coords - centers[:, None, :]) / scales[:, None, None]
    return {
        "normalized": normalized,
        "confidence": confidence,
        "centers": centers,
        "scales": scales,
        "global_scale": fallback_scale,
    }


def add_snapshot_features(features, normalized, frame_indexes):
    for frame_idx in frame_indexes:
        for keypoint_name in SELECTED_KEYPOINTS:
            kp_idx = KP[keypoint_name]
            for axis_idx, axis_name in enumerate(["x", "y"]):
                col = f"frame_{frame_idx:02d}_{keypoint_name}_{axis_name}_norm"
                if frame_idx >= normalized.shape[0]:
                    features[col] = np.nan
                    continue

                value = normalized[frame_idx, kp_idx, axis_idx]
                features[col] = float(value) if np.isfinite(value) else np.nan


def add_motion_delta_features(features, norm_data, frame_indexes):
    normalized = norm_data["normalized"]
    centers = norm_data["centers"]
    global_scale = max(float(norm_data["global_scale"]), EPS)

    for start_frame, end_frame in zip(frame_indexes[:-1], frame_indexes[1:]):
        for keypoint_name in MOTION_KEYPOINTS:
            kp_idx = KP[keypoint_name]
            for axis_idx, axis_name in enumerate(["x", "y"]):
                col = (
                    f"delta_{start_frame:02d}_{end_frame:02d}_"
                    f"{keypoint_name}_{axis_name}_norm"
                )
                if start_frame >= normalized.shape[0] or end_frame >= normalized.shape[0]:
                    features[col] = np.nan
                    continue

                start_value = normalized[start_frame, kp_idx, axis_idx]
                end_value = normalized[end_frame, kp_idx, axis_idx]
                if np.isfinite(start_value) and np.isfinite(end_value):
                    features[col] = float(end_value - start_value)
                else:
                    features[col] = np.nan

        for axis_idx, axis_name in enumerate(["x", "y"]):
            col = f"delta_{start_frame:02d}_{end_frame:02d}_body_center_{axis_name}_norm"
            if start_frame >= centers.shape[0] or end_frame >= centers.shape[0]:
                features[col] = np.nan
                continue

            start_value = centers[start_frame, axis_idx]
            end_value = centers[end_frame, axis_idx]
            if np.isfinite(start_value) and np.isfinite(end_value):
                features[col] = float((end_value - start_value) / global_scale)
            else:
                features[col] = np.nan


def first_last_point_distance(points):
    points = np.asarray(points, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(points).all(axis=1))
    if valid_idx.size == 0:
        return np.nan
    if valid_idx.size == 1:
        return 0.0
    start = points[valid_idx[0]]
    end = points[valid_idx[-1]]
    return float(np.linalg.norm(end - start))


def add_motion_stat_features(features, norm_data):
    normalized = norm_data["normalized"]
    centers = norm_data["centers"] / max(float(norm_data["global_scale"]), EPS)

    for keypoint_name in MOTION_KEYPOINTS:
        kp_points = normalized[:, KP[keypoint_name], :]
        speed = np.linalg.norm(np.diff(kp_points, axis=0), axis=1)
        finite_speed = finite_values(speed)

        features[f"motion_{keypoint_name}_mean_speed"] = safe_stat(speed, np.mean)
        features[f"motion_{keypoint_name}_max_speed"] = safe_stat(speed, np.max)
        features[f"motion_{keypoint_name}_total_displacement"] = (
            float(np.sum(finite_speed)) if finite_speed.size else np.nan
        )
        features[f"motion_{keypoint_name}_net_displacement"] = first_last_point_distance(kp_points)

    center_speed = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    finite_center_speed = finite_values(center_speed)
    features["motion_body_center_mean_speed"] = safe_stat(center_speed, np.mean)
    features["motion_body_center_max_speed"] = safe_stat(center_speed, np.max)
    features["motion_body_center_total_displacement"] = (
        float(np.sum(finite_center_speed)) if finite_center_speed.size else np.nan
    )
    features["motion_body_center_net_displacement"] = first_last_point_distance(centers)


def angle_three_points(frame_coords, point_a, point_b, point_c):
    a = frame_coords[KP[point_a]]
    b = frame_coords[KP[point_b]]
    c = frame_coords[KP[point_c]]
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(c).all():
        return np.nan

    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if not np.isfinite(norm_product) or norm_product <= EPS:
        return np.nan

    cosine = np.clip(np.dot(ba, bc) / norm_product, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def torso_tilt_abs(frame_coords):
    shoulder_center = midpoint(frame_coords, "left_shoulder", "right_shoulder")
    hip_center = midpoint(frame_coords, "left_hip", "right_hip")
    if shoulder_center is None or hip_center is None:
        return np.nan

    vector = shoulder_center - hip_center
    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= EPS:
        return np.nan

    # Eixo y da imagem cresce para baixo; usamos a inclinacao absoluta em relacao ao eixo vertical.
    return float(abs(np.degrees(np.arctan2(vector[0], -vector[1]))))


def add_angle_stat_features(features, norm_data):
    normalized = norm_data["normalized"]

    for angle_name, points in ANGLE_DEFINITIONS.items():
        if angle_name == "torso_tilt_abs":
            values = [torso_tilt_abs(frame) for frame in normalized]
        else:
            values = [angle_three_points(frame, *points) for frame in normalized]

        add_series_stats(features, f"angle_{angle_name}", values)


def safe_ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return np.nan
    return float(numerator / denominator)


def safe_sum(values):
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.sum(values))


def safe_asymmetry(left_value, right_value):
    total = safe_sum([left_value, right_value])
    if not np.isfinite(total) or abs(total) <= EPS:
        return np.nan
    return float(abs(left_value - right_value) / total)


def ratio_true(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.nan
    return float(np.sum(values[valid] > 0.0) / valid.sum())


def frame_torso_center(frame_coords):
    torso_points = frame_coords[
        [KP["left_shoulder"], KP["right_shoulder"], KP["left_hip"], KP["right_hip"]]
    ]
    valid = finite_rows(torso_points)
    if valid.sum() == 0:
        return np.array([np.nan, np.nan], dtype=float)
    return np.nanmean(torso_points[valid], axis=0)


def highest_hand_relative_height(frame_coords):
    torso_center = frame_torso_center(frame_coords)
    if not np.isfinite(torso_center).all():
        return np.nan

    heights = []
    for wrist_name in ["left_wrist", "right_wrist"]:
        wrist = frame_coords[KP[wrist_name]]
        if np.isfinite(wrist).all():
            heights.append(torso_center[1] - wrist[1])

    if not heights:
        return np.nan
    return float(np.nanmax(heights))


def wrist_above_reference(frame_coords, reference_y):
    if not np.isfinite(reference_y):
        return np.nan

    wrist_y = []
    for wrist_name in ["left_wrist", "right_wrist"]:
        wrist = frame_coords[KP[wrist_name]]
        if np.isfinite(wrist).all():
            wrist_y.append(wrist[1])

    if not wrist_y:
        return np.nan
    return 1.0 if float(np.nanmin(wrist_y)) < float(reference_y) else 0.0


def add_posture_stat_features(features, norm_data):
    normalized = norm_data["normalized"]

    hands_dist = []
    feet_dist = []
    arm_opening = []
    leg_opening = []
    hands_relative_height = []
    wrist_above_shoulder = []
    wrist_above_head = []

    for frame in normalized:
        frame_hands_dist = pair_distance(frame, "left_wrist", "right_wrist")
        frame_feet_dist = pair_distance(frame, "left_ankle", "right_ankle")
        shoulder_dist = pair_distance(frame, "left_shoulder", "right_shoulder")
        hip_dist = pair_distance(frame, "left_hip", "right_hip")

        hands_dist.append(frame_hands_dist)
        feet_dist.append(frame_feet_dist)
        arm_opening.append(safe_ratio(frame_hands_dist, shoulder_dist))
        leg_opening.append(safe_ratio(frame_feet_dist, hip_dist))
        hands_relative_height.append(highest_hand_relative_height(frame))

        shoulder_center = midpoint(frame, "left_shoulder", "right_shoulder")
        shoulder_y = shoulder_center[1] if shoulder_center is not None else np.nan
        wrist_above_shoulder.append(wrist_above_reference(frame, shoulder_y))

        nose = frame[KP["nose"]]
        nose_y = nose[1] if np.isfinite(nose).all() else np.nan
        wrist_above_head.append(wrist_above_reference(frame, nose_y))

    features["dist_hands_mean"] = safe_stat(hands_dist, np.mean)
    features["dist_hands_range"] = series_range(hands_dist)
    features["dist_feet_mean"] = safe_stat(feet_dist, np.mean)
    features["dist_feet_range"] = series_range(feet_dist)
    features["arm_opening_mean"] = safe_stat(arm_opening, np.mean)
    features["arm_opening_range"] = series_range(arm_opening)
    features["leg_opening_mean"] = safe_stat(leg_opening, np.mean)
    features["leg_opening_range"] = series_range(leg_opening)
    features["hands_relative_height_mean"] = safe_stat(hands_relative_height, np.mean)
    features["hands_relative_height_max"] = safe_stat(hands_relative_height, np.max)
    features["hands_relative_height_range"] = series_range(hands_relative_height)
    features["wrist_above_shoulder_ratio"] = ratio_true(wrist_above_shoulder)
    features["wrist_above_head_ratio"] = ratio_true(wrist_above_head)


def point_speed_series(points):
    points = np.asarray(points, dtype=float)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def point_motion_total(points):
    return safe_sum(point_speed_series(points))


def point_position_std(points):
    points = np.asarray(points, dtype=float)
    valid = finite_rows(points)
    if valid.sum() == 0:
        return np.nan
    std_xy = np.nanstd(points[valid], axis=0)
    if not np.isfinite(std_xy).all():
        return np.nan
    return float(np.linalg.norm(std_xy))


def opposite_sign_ratio(left_values, right_values):
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)
    valid = (
        np.isfinite(left_values)
        & np.isfinite(right_values)
        & (np.abs(left_values) > EPS)
        & (np.abs(right_values) > EPS)
    )
    if valid.sum() == 0:
        return np.nan
    return float(np.sum((left_values[valid] * right_values[valid]) < 0.0) / valid.sum())


def safe_corr(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    if valid.sum() < 2:
        return np.nan
    a = values_a[valid]
    b = values_b[valid]
    if np.nanstd(a) <= EPS or np.nanstd(b) <= EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def abs_safe_corr(values_a, values_b):
    corr = safe_corr(values_a, values_b)
    return abs(corr) if np.isfinite(corr) else np.nan


def sign_change_count(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > EPS]
    if values.size < 2:
        return np.nan
    signs = np.sign(values)
    return float(np.sum(signs[1:] != signs[:-1]))


def line_deviation(points, start, end):
    vector = end - start
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= EPS:
        return np.nan
    relative = points - start
    cross = np.abs(vector[0] * relative[:, 1] - vector[1] * relative[:, 0])
    return float(np.nanmax(cross / norm))


def trajectory_summary(points):
    points = np.asarray(points, dtype=float)
    valid = finite_rows(points)
    if valid.sum() < 2:
        return {
            "total_path": np.nan,
            "net_displacement": np.nan,
            "linearity": np.nan,
            "curvature_ratio": np.nan,
            "max_line_deviation": np.nan,
        }

    valid_points = points[valid]
    steps = np.linalg.norm(np.diff(valid_points, axis=0), axis=1)
    total_path = safe_sum(steps)
    net_displacement = float(np.linalg.norm(valid_points[-1] - valid_points[0]))
    linearity = safe_ratio(net_displacement, total_path)
    curvature_ratio = safe_ratio(total_path, net_displacement)
    max_deviation = line_deviation(valid_points, valid_points[0], valid_points[-1])
    return {
        "total_path": total_path,
        "net_displacement": net_displacement,
        "linearity": linearity,
        "curvature_ratio": curvature_ratio,
        "max_line_deviation": max_deviation,
    }


def highest_wrist_y(frame_coords):
    values = []
    for wrist_name in ["left_wrist", "right_wrist"]:
        wrist = frame_coords[KP[wrist_name]]
        if np.isfinite(wrist).all():
            values.append(wrist[1])
    if not values:
        return np.nan
    return float(np.nanmin(values))


def side_elbow_angle(frame_coords, side):
    return angle_three_points(
        frame_coords,
        f"{side}_shoulder",
        f"{side}_elbow",
        f"{side}_wrist",
    )


def side_knee_angle(frame_coords, side):
    return angle_three_points(
        frame_coords,
        f"{side}_hip",
        f"{side}_knee",
        f"{side}_ankle",
    )


def add_semantic_stat_features(features, norm_data):
    normalized = norm_data["normalized"]
    centers = norm_data["centers"] / max(float(norm_data["global_scale"]), EPS)

    motion_totals = {}
    for keypoint_name in [
        "left_wrist",
        "right_wrist",
        "left_elbow",
        "right_elbow",
        "left_ankle",
        "right_ankle",
        "left_knee",
        "right_knee",
    ]:
        motion_totals[keypoint_name] = point_motion_total(normalized[:, KP[keypoint_name], :])

    center_motion = point_motion_total(centers)
    wrist_motion = safe_sum([motion_totals["left_wrist"], motion_totals["right_wrist"]])
    ankle_motion = safe_sum([motion_totals["left_ankle"], motion_totals["right_ankle"]])
    upper_motion = safe_sum(
        [
            motion_totals["left_wrist"],
            motion_totals["right_wrist"],
            motion_totals["left_elbow"],
            motion_totals["right_elbow"],
        ]
    )
    lower_motion = safe_sum(
        [
            motion_totals["left_ankle"],
            motion_totals["right_ankle"],
            motion_totals["left_knee"],
            motion_totals["right_knee"],
        ]
    )
    limb_motion = safe_sum([upper_motion, lower_motion])
    overall_motion = safe_sum([upper_motion, lower_motion, center_motion])

    features["semantic_upper_motion_total"] = upper_motion
    features["semantic_lower_motion_total"] = lower_motion
    features["semantic_wrist_motion_total"] = wrist_motion
    features["semantic_ankle_motion_total"] = ankle_motion
    features["semantic_body_center_motion_total"] = center_motion
    features["semantic_overall_motion_energy"] = overall_motion
    features["semantic_upper_lower_motion_ratio"] = safe_ratio(upper_motion, lower_motion)
    features["semantic_wrist_ankle_motion_ratio"] = safe_ratio(wrist_motion, ankle_motion)
    features["semantic_body_limb_motion_ratio"] = safe_ratio(center_motion, limb_motion)
    features["semantic_wrist_motion_share"] = safe_ratio(wrist_motion, overall_motion)
    features["semantic_ankle_motion_share"] = safe_ratio(ankle_motion, overall_motion)

    features["semantic_wrist_motion_asymmetry"] = safe_asymmetry(
        motion_totals["left_wrist"],
        motion_totals["right_wrist"],
    )
    features["semantic_ankle_motion_asymmetry"] = safe_asymmetry(
        motion_totals["left_ankle"],
        motion_totals["right_ankle"],
    )
    features["semantic_elbow_motion_asymmetry"] = safe_asymmetry(
        motion_totals["left_elbow"],
        motion_totals["right_elbow"],
    )
    features["semantic_knee_motion_asymmetry"] = safe_asymmetry(
        motion_totals["left_knee"],
        motion_totals["right_knee"],
    )

    left_ankle_y_velocity = np.diff(normalized[:, KP["left_ankle"], 1])
    right_ankle_y_velocity = np.diff(normalized[:, KP["right_ankle"], 1])
    left_wrist_y_velocity = np.diff(normalized[:, KP["left_wrist"], 1])
    right_wrist_y_velocity = np.diff(normalized[:, KP["right_wrist"], 1])
    features["semantic_ankle_y_opposition_ratio"] = opposite_sign_ratio(
        left_ankle_y_velocity,
        right_ankle_y_velocity,
    )
    features["semantic_wrist_y_opposition_ratio"] = opposite_sign_ratio(
        left_wrist_y_velocity,
        right_wrist_y_velocity,
    )

    wrist_height_over_shoulder = []
    wrist_height_over_head = []
    high_wrist_elbow_angles = []
    highest_wrist_y_values = []
    knee_bend = []
    low_stance = []
    leg_opening_when_bent = []
    torso_tilt_when_legs_open = []

    for frame in normalized:
        shoulder_center = midpoint(frame, "left_shoulder", "right_shoulder")
        shoulder_y = shoulder_center[1] if shoulder_center is not None else np.nan
        nose = frame[KP["nose"]]
        nose_y = nose[1] if np.isfinite(nose).all() else np.nan
        wrist_y = highest_wrist_y(frame)
        highest_wrist_y_values.append(wrist_y)

        wrist_over_shoulder = shoulder_y - wrist_y if np.isfinite(shoulder_y) and np.isfinite(wrist_y) else np.nan
        wrist_over_head = nose_y - wrist_y if np.isfinite(nose_y) and np.isfinite(wrist_y) else np.nan
        wrist_height_over_shoulder.append(wrist_over_shoulder)
        wrist_height_over_head.append(wrist_over_head)

        for side in ["left", "right"]:
            wrist = frame[KP[f"{side}_wrist"]]
            if np.isfinite(wrist).all() and np.isfinite(shoulder_y) and wrist[1] < shoulder_y:
                high_wrist_elbow_angles.append(side_elbow_angle(frame, side))

        left_knee_angle = side_knee_angle(frame, "left")
        right_knee_angle = side_knee_angle(frame, "right")
        mean_knee_angle = safe_stat([left_knee_angle, right_knee_angle], np.mean)
        bend = 180.0 - mean_knee_angle if np.isfinite(mean_knee_angle) else np.nan
        knee_bend.append(bend)

        feet_dist = pair_distance(frame, "left_ankle", "right_ankle")
        hip_dist = pair_distance(frame, "left_hip", "right_hip")
        leg_opening = safe_ratio(feet_dist, hip_dist)
        torso_tilt = torso_tilt_abs(frame)

        low_stance.append(
            1.0
            if np.isfinite(bend)
            and np.isfinite(leg_opening)
            and bend > 25.0
            and leg_opening > 1.2
            else 0.0
            if np.isfinite(bend) and np.isfinite(leg_opening)
            else np.nan
        )
        if np.isfinite(bend) and bend > 20.0:
            leg_opening_when_bent.append(leg_opening)
        if np.isfinite(leg_opening) and leg_opening > 1.2:
            torso_tilt_when_legs_open.append(torso_tilt)

    upward_delta = np.asarray(highest_wrist_y_values[:-1]) - np.asarray(highest_wrist_y_values[1:])
    features["semantic_wrist_height_over_shoulder_mean"] = safe_stat(wrist_height_over_shoulder, np.mean)
    features["semantic_wrist_height_over_shoulder_max"] = safe_stat(wrist_height_over_shoulder, np.max)
    features["semantic_wrist_height_over_head_max"] = safe_stat(wrist_height_over_head, np.max)
    features["semantic_high_wrist_elbow_angle_mean"] = safe_stat(high_wrist_elbow_angles, np.mean)
    features["semantic_wrist_upward_displacement_max"] = safe_stat(upward_delta, np.max)
    features["semantic_knee_bend_mean"] = safe_stat(knee_bend, np.mean)
    features["semantic_knee_bend_max"] = safe_stat(knee_bend, np.max)
    features["semantic_low_stance_ratio"] = ratio_true(low_stance)
    features["semantic_leg_opening_when_bent_mean"] = safe_stat(leg_opening_when_bent, np.mean)
    features["semantic_torso_tilt_when_legs_open_mean"] = safe_stat(torso_tilt_when_legs_open, np.mean)

    wrist_position_std = safe_stat(
        [
            point_position_std(normalized[:, KP["left_wrist"], :]),
            point_position_std(normalized[:, KP["right_wrist"], :]),
        ],
        np.mean,
    )
    ankle_position_std = safe_stat(
        [
            point_position_std(normalized[:, KP["left_ankle"], :]),
            point_position_std(normalized[:, KP["right_ankle"], :]),
        ],
        np.mean,
    )
    features["semantic_body_center_position_std"] = point_position_std(centers)
    features["semantic_wrist_position_std"] = wrist_position_std
    features["semantic_ankle_position_std"] = ankle_position_std


def elbow_sequence_features(angle_values):
    angle_values = np.asarray(angle_values, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(angle_values))
    if valid_idx.size == 0:
        return {
            "flexion_before_extension": np.nan,
            "extension_after_flexion": np.nan,
            "max_extension_velocity": np.nan,
        }

    values = angle_values[valid_idx]
    min_pos = int(np.nanargmin(values))
    min_value = values[min_pos]
    first_value = values[0]
    after_min = values[min_pos:]
    diffs = np.diff(values)
    return {
        "flexion_before_extension": float(first_value - min_value) if np.isfinite(first_value) else np.nan,
        "extension_after_flexion": float(np.nanmax(after_min) - min_value) if after_min.size else np.nan,
        "max_extension_velocity": safe_stat(diffs, np.max),
    }


def wrist_torso_distances(normalized, wrist_name):
    values = []
    for frame in normalized:
        wrist = frame[KP[wrist_name]]
        torso = frame_torso_center(frame)
        if np.isfinite(wrist).all() and np.isfinite(torso).all():
            values.append(float(np.linalg.norm(wrist - torso)))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def leg_opening_series(normalized):
    values = []
    for frame in normalized:
        feet_dist = pair_distance(frame, "left_ankle", "right_ankle")
        hip_dist = pair_distance(frame, "left_hip", "right_hip")
        values.append(safe_ratio(feet_dist, hip_dist))
    return np.asarray(values, dtype=float)


def ratio_above_threshold(values, threshold):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.nan
    return float(np.sum(values[valid] > threshold) / valid.sum())


def normalized_pair_distance_series(normalized, point_a, point_b, reference_a, reference_b):
    values = []
    for frame in normalized:
        distance = pair_distance(frame, point_a, point_b)
        reference = pair_distance(frame, reference_a, reference_b)
        values.append(safe_ratio(distance, reference))
    return np.asarray(values, dtype=float)


def positive_peak(values):
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.max(np.maximum(values, 0.0)))


def motion_entropy(motion_values):
    values = finite_values(motion_values)
    total = safe_sum(values)
    if values.size == 0 or not np.isfinite(total) or total <= EPS:
        return 0.0
    shares = values / total
    entropy = -np.sum(shares * np.log(shares + EPS))
    return float(entropy / np.log(values.size)) if values.size > 1 else 0.0


def add_refined_action_features(features, norm_data):
    normalized = norm_data["normalized"]
    centers = norm_data["centers"] / max(float(norm_data["global_scale"]), EPS)

    left_ankle = normalized[:, KP["left_ankle"], :]
    right_ankle = normalized[:, KP["right_ankle"], :]
    left_ankle_y_velocity = np.diff(left_ankle[:, 1])
    right_ankle_y_velocity = np.diff(right_ankle[:, 1])
    left_ankle_y_range = series_range(left_ankle[:, 1])
    right_ankle_y_range = series_range(right_ankle[:, 1])
    left_ankle_direction_changes = sign_change_count(left_ankle_y_velocity)
    right_ankle_direction_changes = sign_change_count(right_ankle_y_velocity)
    ankle_direction_changes = [left_ankle_direction_changes, right_ankle_direction_changes]
    ankle_vertical_ranges = [left_ankle_y_range, right_ankle_y_range]

    ankle_x_gap = left_ankle[:, 0] - right_ankle[:, 0]
    features["refined_ankle_x_cross_count"] = sign_change_count(ankle_x_gap)
    features["refined_ankle_y_direction_change_mean"] = safe_stat(ankle_direction_changes, np.mean)
    features["refined_ankle_y_direction_change_max"] = safe_stat(ankle_direction_changes, np.max)
    features["refined_ankle_y_direction_change_asymmetry"] = safe_asymmetry(
        left_ankle_direction_changes,
        right_ankle_direction_changes,
    )
    features["refined_ankle_vertical_range_mean"] = safe_stat(ankle_vertical_ranges, np.mean)
    features["refined_ankle_vertical_range_asymmetry"] = safe_asymmetry(
        left_ankle_y_range,
        right_ankle_y_range,
    )

    center_x_motion = safe_sum(np.abs(np.diff(centers[:, 0])))
    center_y_motion = safe_sum(np.abs(np.diff(centers[:, 1])))
    center_summary = trajectory_summary(centers)
    features["refined_body_center_low_displacement_flag_010"] = (
        1.0
        if np.isfinite(center_summary["net_displacement"])
        and center_summary["net_displacement"] < 0.10
        else 0.0
        if np.isfinite(center_summary["net_displacement"])
        else np.nan
    )
    features["refined_body_center_vertical_range"] = series_range(centers[:, 1])
    features["refined_body_center_vertical_std"] = safe_stat(centers[:, 1], np.std)
    features["refined_body_center_vertical_horizontal_motion_ratio"] = safe_ratio(
        center_y_motion,
        center_x_motion,
    )

    leg_opening = leg_opening_series(normalized)
    knee_opening = normalized_pair_distance_series(
        normalized,
        "left_knee",
        "right_knee",
        "left_hip",
        "right_hip",
    )
    leg_opening_mean = safe_stat(leg_opening, np.mean)
    leg_opening_std = safe_stat(leg_opening, np.std)
    knee_opening_mean = safe_stat(knee_opening, np.mean)
    knee_opening_std = safe_stat(knee_opening, np.std)
    wide_base_ratio = ratio_above_threshold(leg_opening, 1.2)
    features["refined_leg_opening_std"] = leg_opening_std
    features["refined_leg_opening_stability_score"] = safe_ratio(leg_opening_mean, 1.0 + leg_opening_std)
    features["refined_leg_opening_range_to_mean_ratio"] = safe_ratio(series_range(leg_opening), leg_opening_mean)
    features["refined_knee_opening_mean"] = knee_opening_mean
    features["refined_knee_opening_std"] = knee_opening_std
    features["refined_knee_opening_range"] = series_range(knee_opening)
    features["refined_wide_base_stability_score"] = safe_ratio(wide_base_ratio, 1.0 + leg_opening_std)

    wrist_points = {
        "left": normalized[:, KP["left_wrist"], :],
        "right": normalized[:, KP["right_wrist"], :],
    }
    wrist_motion = {
        side: point_motion_total(points)
        for side, points in wrist_points.items()
    }
    active_side = "left"
    if np.nan_to_num(wrist_motion["right"], nan=-1.0) > np.nan_to_num(wrist_motion["left"], nan=-1.0):
        active_side = "right"

    left_wrist_y_velocity = np.diff(wrist_points["left"][:, 1])
    right_wrist_y_velocity = np.diff(wrist_points["right"][:, 1])
    left_wrist_y_std = safe_stat(left_wrist_y_velocity, np.std)
    right_wrist_y_std = safe_stat(right_wrist_y_velocity, np.std)
    active_wrist_y_velocity = np.diff(wrist_points[active_side][:, 1])
    active_wrist_y_acceleration = np.diff(active_wrist_y_velocity)
    features["refined_left_wrist_y_velocity_std"] = left_wrist_y_std
    features["refined_right_wrist_y_velocity_std"] = right_wrist_y_std
    features["refined_wrist_y_velocity_asymmetry"] = safe_asymmetry(left_wrist_y_std, right_wrist_y_std)
    features["refined_active_wrist_y_acceleration_abs_max"] = safe_stat(
        np.abs(active_wrist_y_acceleration),
        np.max,
    )
    features["refined_active_wrist_y_acceleration_std"] = safe_stat(active_wrist_y_acceleration, np.std)
    features["refined_active_wrist_vertical_peak_count"] = sign_change_count(active_wrist_y_velocity)

    left_wrist_torso = wrist_torso_distances(normalized, "left_wrist")
    right_wrist_torso = wrist_torso_distances(normalized, "right_wrist")
    active_wrist_torso = wrist_torso_distances(normalized, f"{active_side}_wrist")
    left_torso_velocity = np.diff(left_wrist_torso)
    right_torso_velocity = np.diff(right_wrist_torso)
    active_torso_velocity = np.diff(active_wrist_torso)
    wrist_torso_ranges = [series_range(left_wrist_torso), series_range(right_wrist_torso)]
    wrist_torso_deltas = [
        first_to_last_delta(left_wrist_torso),
        first_to_last_delta(right_wrist_torso),
    ]
    features["refined_active_wrist_torso_distance_peak_velocity"] = safe_stat(
        np.abs(active_torso_velocity),
        np.max,
    )
    features["refined_active_wrist_torso_extension_speed_max"] = positive_peak(active_torso_velocity)
    features["refined_both_wrists_torso_distance_delta_mean"] = safe_stat(wrist_torso_deltas, np.mean)
    features["refined_both_wrists_torso_distance_delta_max"] = safe_stat(wrist_torso_deltas, np.max)
    features["refined_both_wrists_extension_speed_max"] = safe_stat(
        [
            positive_peak(left_torso_velocity),
            positive_peak(right_torso_velocity),
        ],
        np.max,
    )
    features["refined_wrist_torso_distance_range_mean"] = safe_stat(wrist_torso_ranges, np.mean)
    features["refined_wrist_torso_distance_range_max"] = safe_stat(wrist_torso_ranges, np.max)

    selected_motions = [
        point_motion_total(normalized[:, KP[keypoint_name], :])
        for keypoint_name in SELECTED_KEYPOINTS
    ]
    selected_motion_total = safe_sum(selected_motions)
    selected_motion_mean = safe_stat(selected_motions, np.mean)
    selected_motion_std = safe_stat(selected_motions, np.std)
    features["refined_selected_keypoint_motion_total"] = selected_motion_total
    features["refined_selected_keypoint_motion_mean"] = selected_motion_mean
    features["refined_selected_keypoint_motion_std"] = selected_motion_std
    features["refined_selected_keypoint_motion_max"] = safe_stat(selected_motions, np.max)
    features["refined_motion_entropy"] = motion_entropy(selected_motions)
    features["refined_pose_stability_score"] = safe_ratio(1.0, 1.0 + selected_motion_total)
    features["refined_low_motion_wide_base_score"] = safe_ratio(wide_base_ratio, 1.0 + selected_motion_total)
    features["refined_pick_stability_score"] = safe_ratio(
        wide_base_ratio * safe_ratio(leg_opening_mean, 1.0 + leg_opening_std),
        1.0 + selected_motion_total,
    )


def add_pair_critical_features(features, norm_data):
    normalized = norm_data["normalized"]
    centers = norm_data["centers"] / max(float(norm_data["global_scale"]), EPS)

    wrist_points = {
        "left": normalized[:, KP["left_wrist"], :],
        "right": normalized[:, KP["right_wrist"], :],
    }
    wrist_motion = {
        side: point_motion_total(points)
        for side, points in wrist_points.items()
    }
    active_side = "left"
    if np.nan_to_num(wrist_motion["right"], nan=-1.0) > np.nan_to_num(wrist_motion["left"], nan=-1.0):
        active_side = "right"

    active_wrist_points = wrist_points[active_side]
    active_wrist_summary = trajectory_summary(active_wrist_points)
    for metric, value in active_wrist_summary.items():
        features[f"pair_active_wrist_{metric}"] = value

    active_wrist_speed = point_speed_series(active_wrist_points)
    active_wrist_y_velocity = np.diff(active_wrist_points[:, 1])
    upward_velocity = -active_wrist_y_velocity
    features["pair_active_wrist_peak_speed"] = safe_stat(active_wrist_speed, np.max)
    features["pair_active_wrist_peak_to_mean_speed_ratio"] = safe_ratio(
        safe_stat(active_wrist_speed, np.max),
        safe_stat(active_wrist_speed, np.mean),
    )
    features["pair_active_wrist_upward_speed_max"] = safe_stat(upward_velocity, np.max)
    features["pair_active_wrist_vertical_range"] = series_range(active_wrist_points[:, 1])
    features["pair_active_wrist_y_direction_change_count"] = sign_change_count(active_wrist_y_velocity)

    active_elbow_angles = [
        side_elbow_angle(frame, active_side)
        for frame in normalized
    ]
    elbow_features = elbow_sequence_features(active_elbow_angles)
    for metric, value in elbow_features.items():
        features[f"pair_active_elbow_{metric}"] = value

    left_wrist_torso = wrist_torso_distances(normalized, "left_wrist")
    right_wrist_torso = wrist_torso_distances(normalized, "right_wrist")
    active_wrist_torso = wrist_torso_distances(normalized, f"{active_side}_wrist")
    left_wrist_speed = point_speed_series(wrist_points["left"])
    right_wrist_speed = point_speed_series(wrist_points["right"])
    features["pair_active_wrist_torso_distance_range"] = series_range(active_wrist_torso)
    features["pair_active_wrist_torso_distance_delta"] = first_to_last_delta(active_wrist_torso)
    features["pair_wrist_torso_distance_corr"] = safe_corr(left_wrist_torso, right_wrist_torso)
    features["pair_wrist_torso_distance_range_asymmetry"] = safe_asymmetry(
        series_range(left_wrist_torso),
        series_range(right_wrist_torso),
    )
    features["pair_wrist_speed_corr"] = safe_corr(left_wrist_speed, right_wrist_speed)
    features["pair_wrist_y_velocity_corr"] = safe_corr(
        np.diff(wrist_points["left"][:, 1]),
        np.diff(wrist_points["right"][:, 1]),
    )

    ankle_y_velocities = [
        np.diff(normalized[:, KP["left_ankle"], 1]),
        np.diff(normalized[:, KP["right_ankle"], 1]),
    ]
    ankle_y_corrs = [
        abs_safe_corr(active_wrist_y_velocity, ankle_velocity)
        for ankle_velocity in ankle_y_velocities
    ]
    ankle_y_std = safe_stat([safe_stat(values, np.std) for values in ankle_y_velocities], np.mean)
    features["pair_active_wrist_ankle_y_corr_abs_max"] = safe_stat(ankle_y_corrs, np.max)
    features["pair_active_wrist_ankle_y_corr_abs_mean"] = safe_stat(ankle_y_corrs, np.mean)
    features["pair_active_wrist_to_ankle_y_std_ratio"] = safe_ratio(
        safe_stat(active_wrist_y_velocity, np.std),
        ankle_y_std,
    )

    lower_lateral_motion = safe_sum(
        [
            safe_sum(np.abs(np.diff(normalized[:, KP["left_ankle"], 0]))),
            safe_sum(np.abs(np.diff(normalized[:, KP["right_ankle"], 0]))),
            safe_sum(np.abs(np.diff(normalized[:, KP["left_knee"], 0]))),
            safe_sum(np.abs(np.diff(normalized[:, KP["right_knee"], 0]))),
        ]
    )
    center_lateral_motion = safe_sum(np.abs(np.diff(centers[:, 0])))
    center_summary = trajectory_summary(centers)
    leg_opening = leg_opening_series(normalized)
    features["pair_lower_body_lateral_motion_total"] = lower_lateral_motion
    features["pair_body_center_lateral_motion_total"] = center_lateral_motion
    features["pair_lower_lateral_center_ratio"] = safe_ratio(lower_lateral_motion, center_lateral_motion)
    features["pair_wide_base_ratio"] = ratio_above_threshold(leg_opening, 1.2)
    features["pair_wide_base_low_center_motion_score"] = safe_ratio(
        safe_stat(leg_opening, np.mean),
        1.0 + center_summary["total_path"] if np.isfinite(center_summary["total_path"]) else np.nan,
    )

    for metric, value in center_summary.items():
        features[f"pair_body_center_{metric}"] = value
    features["pair_body_center_x_net_abs"] = abs(first_to_last_delta(centers[:, 0]))
    features["pair_body_center_y_net_abs"] = abs(first_to_last_delta(centers[:, 1]))


def extract_temporal_features(
    pose,
    frame_indexes,
    conf_threshold,
    include_snapshots=True,
    include_motion_deltas=False,
    include_motion_stats=False,
    include_angle_stats=False,
    include_posture_stats=False,
    include_semantic_stats=False,
    include_pair_critical_stats=False,
    include_refined_action_stats=False,
):
    norm_data = normalize_pose(pose, conf_threshold)
    features = {}

    if include_snapshots:
        add_snapshot_features(features, norm_data["normalized"], frame_indexes)

    if include_motion_deltas:
        add_motion_delta_features(features, norm_data, frame_indexes)

    if include_motion_stats:
        add_motion_stat_features(features, norm_data)

    if include_angle_stats:
        add_angle_stat_features(features, norm_data)

    if include_posture_stats:
        add_posture_stat_features(features, norm_data)

    if include_semantic_stats:
        add_semantic_stat_features(features, norm_data)

    if include_pair_critical_stats:
        add_pair_critical_features(features, norm_data)

    if include_refined_action_stats:
        add_refined_action_features(features, norm_data)

    return features


def build_dataset(
    manifest,
    root_dir,
    frame_indexes,
    conf_threshold,
    include_snapshots=True,
    include_motion_deltas=False,
    include_motion_stats=False,
    include_angle_stats=False,
    include_posture_stats=False,
    include_semantic_stats=False,
    include_pair_critical_stats=False,
    include_refined_action_stats=False,
):
    rows = []
    missing_files = []
    errors = []

    for _, item in manifest.iterrows():
        sample_id = str(item["sample_id"])
        pose_path = resolve_path(root_dir, item["pose_path"])

        if not pose_path.exists():
            missing_files.append(sample_id)
            continue

        try:
            pose = np.load(pose_path)
            features = extract_temporal_features(
                pose=pose,
                frame_indexes=frame_indexes,
                conf_threshold=conf_threshold,
                include_snapshots=include_snapshots,
                include_motion_deltas=include_motion_deltas,
                include_motion_stats=include_motion_stats,
                include_angle_stats=include_angle_stats,
                include_posture_stats=include_posture_stats,
                include_semantic_stats=include_semantic_stats,
                include_pair_critical_stats=include_pair_critical_stats,
                include_refined_action_stats=include_refined_action_stats,
            )
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": str(exc)})
            continue

        row = {
            "sample_id": sample_id,
            "label_id": int(item["label_id"]),
            "label": item["label"],
            "frame_count": item.get("frame_count", np.nan),
            "detected_frame_count": item.get("detected_frame_count", np.nan),
            "missing_frame_ratio": item.get("missing_frame_ratio", np.nan),
            "mean_keypoint_conf": item.get("mean_keypoint_conf", np.nan),
        }
        if "group_id" in item and pd.notna(item["group_id"]):
            row["group_id"] = int(item["group_id"])
        if "group_label" in item and pd.notna(item["group_label"]):
            row["group_label"] = item["group_label"]
        for optional_col in ["source_sample_id", "augmentation", "split"]:
            if optional_col in item and pd.notna(item[optional_col]):
                row[optional_col] = item[optional_col]
        row.update(features)
        rows.append(row)

    dataset = pd.DataFrame(rows)
    return dataset, missing_files, errors


def add_quality_flag(dataset):
    dataset = dataset.copy()
    missing_ratio = pd.to_numeric(dataset["missing_frame_ratio"], errors="coerce")
    mean_conf = pd.to_numeric(dataset["mean_keypoint_conf"], errors="coerce")

    dataset["quality_flag"] = (
        (missing_ratio > 0.25)
        | (mean_conf < 0.50)
        | missing_ratio.isna()
        | mean_conf.isna()
    )
    return dataset


def save_splits(dataset, feature_cols, output_prefix, test_size, random_state):
    data = dataset[~dataset["quality_flag"]].copy()
    X = data[feature_cols].copy()
    y = data["label_id"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    processed_dir = output_prefix.parent
    output_name = output_prefix.name
    X_train.to_csv(processed_dir / f"X_train_{output_name}.csv", index=False)
    X_test.to_csv(processed_dir / f"X_test_{output_name}.csv", index=False)
    y_train.to_frame("label_id").to_csv(
        processed_dir / f"y_train_{output_name}.csv",
        index=False,
    )
    y_test.to_frame("label_id").to_csv(
        processed_dir / f"y_test_{output_name}.csv",
        index=False,
    )

    return {
        "quality_filtered_rows": int(len(data)),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": list(y_train.shape),
        "y_test_shape": list(y_test.shape),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extrai features temporais normalizadas de keypoints corporais."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Caminho do manifest CSV. Padrao: data/processed/yolo_pose_manifest_balanced_400.csv",
    )
    parser.add_argument(
        "--output-name",
        default="dataset_basquet_temporal_keypoints",
        help="Prefixo dos arquivos gerados em data/processed.",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=DEFAULT_FRAMES,
        help="Frames usados como snapshots temporais.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONF_THRESHOLD,
        help="Confianca minima para aceitar um keypoint.",
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Nao salva X_train/X_test/y_train/y_test.",
    )
    parser.add_argument(
        "--include-motion-deltas",
        action="store_true",
        help="Inclui deltas de punhos, tornozelos e centro do corpo entre os frames.",
    )
    parser.add_argument(
        "--include-motion-stats",
        action="store_true",
        help="Inclui velocidade media/maxima e deslocamento total/liquido.",
    )
    parser.add_argument(
        "--include-angle-stats",
        action="store_true",
        help="Inclui estatisticas de angulos de cotovelo, ombro, joelho e tronco.",
    )
    parser.add_argument(
        "--include-posture-stats",
        action="store_true",
        help="Inclui distancias, aberturas e altura relativa das maos.",
    )
    parser.add_argument(
        "--include-semantic-stats",
        action="store_true",
        help="Inclui features semanticas de energia relativa, assimetria, bracos altos, base defensiva e estabilidade.",
    )
    parser.add_argument(
        "--include-pair-critical-stats",
        action="store_true",
        help="Inclui features focadas nos pares criticos: shoot/block, pass/ball in hand, dribble/locomocao, defense/pick e walk/no_action.",
    )
    parser.add_argument(
        "--include-refined-action-stats",
        action="store_true",
        help="Inclui features refinadas para locomocao, defesa/walk, drible, passe e estabilidade de pick.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Nao inclui as coordenadas normalizadas dos frames como features.",
    )
    args = parser.parse_args()

    root_dir = find_root_dir()
    processed_dir = root_dir / "data" / "processed"
    manifest_path = args.manifest or (processed_dir / "yolo_pose_manifest_balanced_400.csv")
    output_prefix = processed_dir / args.output_name

    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
    dataset, missing_files, errors = build_dataset(
        manifest=manifest,
        root_dir=root_dir,
        frame_indexes=args.frames,
        conf_threshold=args.conf_threshold,
        include_snapshots=not args.no_snapshots,
        include_motion_deltas=args.include_motion_deltas,
        include_motion_stats=args.include_motion_stats,
        include_angle_stats=args.include_angle_stats,
        include_posture_stats=args.include_posture_stats,
        include_semantic_stats=args.include_semantic_stats,
        include_pair_critical_stats=args.include_pair_critical_stats,
        include_refined_action_stats=args.include_refined_action_stats,
    )
    if dataset.empty:
        raise SystemExit(
            "Nenhuma pose foi carregada. Confira se os pose_path do manifest existem "
            "ou rode a extracao YOLO antes de gerar features."
        )

    id_cols = ["sample_id", "label_id", "label"]
    if "group_id" in dataset.columns:
        id_cols.append("group_id")
    if "group_label" in dataset.columns:
        id_cols.append("group_label")
    for optional_col in ["source_sample_id", "augmentation", "split"]:
        if optional_col in dataset.columns:
            id_cols.append(optional_col)
    metadata_cols = [
        "frame_count",
        "detected_frame_count",
        "missing_frame_ratio",
        "mean_keypoint_conf",
        "quality_flag",
    ]
    feature_cols = [col for col in dataset.columns if col not in id_cols + metadata_cols]

    for col in metadata_cols[:-1] + feature_cols:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    dataset = add_quality_flag(dataset)
    snapshot_cols = [col for col in feature_cols if col.startswith("frame_")]
    delta_cols = [col for col in feature_cols if col.startswith("delta_")]
    body_center_delta_cols = [col for col in delta_cols if "body_center" in col]
    motion_stat_cols = [col for col in feature_cols if col.startswith("motion_")]
    angle_cols = [col for col in feature_cols if col.startswith("angle_")]
    semantic_cols = [col for col in feature_cols if col.startswith("semantic_")]
    pair_critical_cols = [col for col in feature_cols if col.startswith("pair_")]
    refined_action_cols = [col for col in feature_cols if col.startswith("refined_")]
    posture_cols = [
        col
        for col in feature_cols
        if col.startswith("dist_")
        or col.startswith("arm_opening")
        or col.startswith("leg_opening")
        or col.startswith("hands_relative_height")
        or col.startswith("wrist_above")
    ]

    missing_report = pd.DataFrame(
        {
            "column": feature_cols,
            "missing_count": [int(dataset[col].isna().sum()) for col in feature_cols],
        }
    )
    missing_report["missing_pct"] = (
        missing_report["missing_count"] / len(dataset) * 100
    ).round(4)
    missing_report = missing_report.sort_values("missing_count", ascending=False)

    class_counts = dataset["label"].value_counts().sort_index()
    feature_report = pd.DataFrame(
        [
            {"metric": "rows", "value": len(dataset)},
            {"metric": "columns_total", "value": dataset.shape[1]},
            {"metric": "feature_columns", "value": len(feature_cols)},
            {"metric": "snapshot_features", "value": len(snapshot_cols)},
            {"metric": "motion_delta_features", "value": len(delta_cols)},
            {"metric": "body_center_delta_features", "value": len(body_center_delta_cols)},
            {"metric": "motion_stat_features", "value": len(motion_stat_cols)},
            {"metric": "angle_features", "value": len(angle_cols)},
            {"metric": "posture_features", "value": len(posture_cols)},
            {"metric": "semantic_features", "value": len(semantic_cols)},
            {"metric": "pair_critical_features", "value": len(pair_critical_cols)},
            {"metric": "refined_action_features", "value": len(refined_action_cols)},
            {"metric": "classes", "value": dataset["label"].nunique()},
            {"metric": "min_class_count", "value": int(class_counts.min())},
            {"metric": "max_class_count", "value": int(class_counts.max())},
            {"metric": "quality_flag_count", "value": int(dataset["quality_flag"].sum())},
            {
                "metric": "quality_flag_pct",
                "value": round(float(dataset["quality_flag"].mean() * 100), 2),
            },
            {"metric": "missing_files", "value": len(missing_files)},
            {"metric": "extraction_errors", "value": len(errors)},
        ]
    )

    class_info = (
        dataset[["label_id", "label"]]
        .drop_duplicates()
        .sort_values("label_id")
        .set_index("label_id")["label"]
        .to_dict()
    )

    split_info = None
    if not args.no_splits:
        split_info = save_splits(
            dataset=dataset,
            feature_cols=feature_cols,
            output_prefix=output_prefix,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    metadata = {
        "feature_columns": feature_cols,
        "drop_columns": id_cols + metadata_cols,
        "selected_frames": args.frames,
        "selected_keypoints": SELECTED_KEYPOINTS,
        "motion_keypoints": MOTION_KEYPOINTS,
        "angle_definitions": ANGLE_DEFINITIONS,
        "include_snapshots": not args.no_snapshots,
        "include_motion_deltas": args.include_motion_deltas,
        "include_motion_stats": args.include_motion_stats,
        "include_angle_stats": args.include_angle_stats,
        "include_posture_stats": args.include_posture_stats,
        "include_semantic_stats": args.include_semantic_stats,
        "include_pair_critical_stats": args.include_pair_critical_stats,
        "include_refined_action_stats": args.include_refined_action_stats,
        "feature_groups": {
            "snapshot_features": len(snapshot_cols),
            "motion_delta_features": len(delta_cols),
            "body_center_delta_features": len(body_center_delta_cols),
            "motion_stat_features": len(motion_stat_cols),
            "angle_features": len(angle_cols),
            "posture_features": len(posture_cols),
            "semantic_features": len(semantic_cols),
            "pair_critical_features": len(pair_critical_cols),
            "refined_action_features": len(refined_action_cols),
        },
        "normalization": {
            "center": "hip midpoint; fallback shoulder midpoint; fallback torso/selected mean",
            "scale": "torso length; fallback shoulder distance; fallback hip distance; fallback bbox diagonal",
            "confidence_threshold": args.conf_threshold,
        },
        "class_ids": [int(k) for k in class_info.keys()],
        "class_names": list(class_info.values()),
        "id_to_label": {str(k): v for k, v in class_info.items()},
        "missing_files": missing_files,
        "errors": errors,
        "split_info": split_info,
    }

    dataset.to_csv(output_prefix.with_suffix(".csv"), index=False)
    missing_report.to_csv(output_prefix.with_name(output_prefix.name + "_missing_report.csv"), index=False)
    feature_report.to_csv(output_prefix.with_name(output_prefix.name + "_feature_report.csv"), index=False)

    with output_prefix.with_name(output_prefix.name + "_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("Dataset salvo em:", output_prefix.with_suffix(".csv").relative_to(root_dir))
    print("Features:", len(feature_cols))
    print("Linhas:", len(dataset))
    print("Quality flags:", int(dataset["quality_flag"].sum()))
    print("Arquivos ausentes:", len(missing_files))
    print("Erros de extracao:", len(errors))


if __name__ == "__main__":
    main()
