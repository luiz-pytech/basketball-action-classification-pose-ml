from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data/processed/dataset_basquet_motion_stats_keypoints_semantic_pair_all_v26.csv"
KEYPOINT_DIR = PROJECT_ROOT / "data/processed/yolo_pose_keypoints_all_v26"
OUTPUT_DIR = PROJECT_ROOT / "results/figures/keypoint_slide_examples"

FRAME_INDEX = 8
CONF_THRESHOLD = 0.25
MAX_SAMPLES_PER_CLASS = 1200

COCO_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def normalize_pose(keypoints: np.ndarray) -> np.ndarray | None:
    frame = keypoints[min(FRAME_INDEX, len(keypoints) - 1)].astype(float).copy()
    xy = frame[:, :2]
    conf = frame[:, 2]

    valid = conf >= CONF_THRESHOLD
    if valid.sum() < 8:
        return None

    left_hip, right_hip = xy[11], xy[12]
    left_shoulder, right_shoulder = xy[5], xy[6]

    if conf[11] >= CONF_THRESHOLD and conf[12] >= CONF_THRESHOLD:
        center = (left_hip + right_hip) / 2
    elif conf[5] >= CONF_THRESHOLD and conf[6] >= CONF_THRESHOLD:
        center = (left_shoulder + right_shoulder) / 2
    else:
        center = xy[valid].mean(axis=0)

    torso_len = np.linalg.norm(((left_shoulder + right_shoulder) / 2) - ((left_hip + right_hip) / 2))
    if not np.isfinite(torso_len) or torso_len < 1e-6:
        torso_len = np.linalg.norm(left_shoulder - right_shoulder)
    if not np.isfinite(torso_len) or torso_len < 1e-6:
        torso_len = np.linalg.norm(xy[valid].max(axis=0) - xy[valid].min(axis=0))
    if not np.isfinite(torso_len) or torso_len < 1e-6:
        return None

    normalized = frame.copy()
    normalized[:, :2] = (xy - center) / torso_len
    normalized[~valid, :2] = np.nan
    return normalized


def load_class_poses(data: pd.DataFrame, label: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rows = data[data["label"].eq(label)].head(MAX_SAMPLES_PER_CLASS)
    poses = []
    for sample_id in rows["sample_id"].astype(str):
        path = KEYPOINT_DIR / f"{sample_id}.npy"
        if not path.exists():
            continue
        normalized = normalize_pose(np.load(path))
        if normalized is None:
            continue
        vector = normalized[:, :2].copy()
        vector = np.nan_to_num(vector, nan=0.0).reshape(-1)
        poses.append((sample_id, normalized, vector))
    return poses


def closest_pair(
    left_poses: list[tuple[str, np.ndarray, np.ndarray]],
    right_poses: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    right_matrix = np.vstack([pose[2] for pose in right_poses])
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(right_matrix)
    distances, indices = nn.kneighbors(np.vstack([pose[2] for pose in left_poses]))
    best_left_index = int(np.argmin(distances[:, 0]))
    best_right_index = int(indices[best_left_index, 0])
    left_id, left_pose, _ = left_poses[best_left_index]
    right_id, right_pose, _ = right_poses[best_right_index]
    return (left_id, left_pose), (right_id, right_pose)


def draw_pose(ax, pose: np.ndarray, title: str) -> None:
    xy = pose[:, :2]
    for start, end in COCO_SKELETON:
        if np.any(np.isnan(xy[[start, end]])):
            continue
        ax.plot(
            xy[[start, end], 0],
            -xy[[start, end], 1],
            color="#2563eb",
            linewidth=4,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(xy[:, 0], -xy[:, 1], s=60, color="#f97316", edgecolor="#111827", linewidth=0.7, zorder=2)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=14)
    ax.set_aspect("equal")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.2, 2.0)
    ax.axis("off")


def save_pair_image(left_label: str, right_label: str, output_name: str) -> None:
    data = pd.read_csv(DATASET_PATH, dtype={"sample_id": str})
    data = data[data["quality_flag"].astype(str).str.lower().eq("false")].copy()

    left_poses = load_class_poses(data, left_label)
    right_poses = load_class_poses(data, right_label)
    if not left_poses or not right_poses:
        raise RuntimeError(f"Sem poses suficientes para {left_label} x {right_label}")

    (left_id, left_pose), (right_id, right_pose) = closest_pair(left_poses, right_poses)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")
    draw_pose(axes[0], left_pose, left_label)
    draw_pose(axes[1], right_pose, right_label)
    fig.suptitle("Distribuição do Dataset", fontsize=20, y=0.98)
    fig.text(
        0.5,
        0.04,
        "Dataset - Total: 37085, Após Filtro: 35586",
        ha="center",
        fontsize=14,
        color="#4b5563",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_pair_image("walk", "no_action", "keypoints_walk_vs_no_action.png")
    save_pair_image("run", "dribble", "keypoints_run_vs_dribble.png")


if __name__ == "__main__":
    main()
