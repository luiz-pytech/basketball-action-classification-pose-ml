from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data/processed/dataset_basquet_defense_walk_shoot_pruned80_all_v26.csv"
KEYPOINT_DIR = PROJECT_ROOT / "data/processed/yolo_pose_keypoints_all_v26"
OUTPUT_DIR = PROJECT_ROOT / "results/figures/keypoint_slide_examples"

FRAME_INDEX = 8
CONF_THRESHOLD = 0.25
MAX_SAMPLES_PER_CLASS = 1600
LABELS = ["defense", "walk", "shoot"]

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


def closest_triplet(
    defense_poses: list[tuple[str, np.ndarray, np.ndarray]],
    walk_poses: list[tuple[str, np.ndarray, np.ndarray]],
    shoot_poses: list[tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, tuple[str, np.ndarray]]:
    walk_matrix = np.vstack([pose[2] for pose in walk_poses])
    shoot_matrix = np.vstack([pose[2] for pose in shoot_poses])

    walk_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(walk_matrix)
    shoot_nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(shoot_matrix)

    defense_matrix = np.vstack([pose[2] for pose in defense_poses])
    walk_distances, walk_indices = walk_nn.kneighbors(defense_matrix)
    shoot_distances, shoot_indices = shoot_nn.kneighbors(defense_matrix)

    best_defense_index = int(np.argmin(walk_distances[:, 0] + shoot_distances[:, 0]))
    best_walk_index = int(walk_indices[best_defense_index, 0])
    best_shoot_index = int(shoot_indices[best_defense_index, 0])

    defense_id, defense_pose, _ = defense_poses[best_defense_index]
    walk_id, walk_pose, _ = walk_poses[best_walk_index]
    shoot_id, shoot_pose, _ = shoot_poses[best_shoot_index]

    return {
        "defense": (defense_id, defense_pose),
        "walk": (walk_id, walk_pose),
        "shoot": (shoot_id, shoot_pose),
    }


def draw_pose(ax, pose: np.ndarray, title: str, sample_id: str) -> None:
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
    ax.scatter(xy[:, 0], -xy[:, 1], s=48, color="#f97316", edgecolor="#111827", linewidth=0.6, zorder=2)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
    ax.text(0.5, -0.03, f"sample_id: {sample_id}", ha="center", transform=ax.transAxes, fontsize=8, color="#4b5563")
    ax.set_aspect("equal")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.2, 2.0)
    ax.axis("off")


def main() -> None:
    data = pd.read_csv(DATASET_PATH, dtype={"sample_id": str})
    if "quality_flag" in data.columns:
        data = data[~data["quality_flag"].astype(bool)].copy()

    poses_by_label = {label: load_class_poses(data, label) for label in LABELS}
    missing = [label for label, poses in poses_by_label.items() if not poses]
    if missing:
        raise RuntimeError(f"Sem poses suficientes para: {missing}")

    triplet = closest_triplet(
        poses_by_label["defense"],
        poses_by_label["walk"],
        poses_by_label["shoot"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), facecolor="white")
    fig.suptitle("Keypoints normalizados - exemplos visualmente parecidos", fontsize=18, y=0.96)
    for ax, label in zip(axes, LABELS):
        sample_id, pose = triplet[label]
        draw_pose(ax, pose, label, sample_id)

    fig.text(
        0.5,
        0.035,
        "Poses muito parecidas (defense, walk e shoot)",
        ha="center",
        fontsize=26,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0.02, 0.16, 0.98, 0.9])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "keypoints_3_classes_defense_walk_shoot_similar.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura salva em: {output_path.relative_to(PROJECT_ROOT)}")
    for label in LABELS:
        print(f"{label}: {triplet[label][0]}")


if __name__ == "__main__":
    main()
