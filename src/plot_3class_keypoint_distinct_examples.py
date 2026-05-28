from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_STATE, TEST_SIZE  # noqa: E402
from src.run_classification_experiment import load_feature_list, prepare_data  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data/processed/dataset_basquet_defense_walk_shoot_pruned80_all_v26.csv"
FEATURE_LIST_PATH = (
    PROJECT_ROOT / "results/feature_lists/feature_audit_semantic_pair_all_v26_original_pruned_80.txt"
)
MODEL_PATH = PROJECT_ROOT / "results/models/best_classification_model_final_3_classes_defense_walk_shoot_pruned80_all_v26.joblib"
KEYPOINT_DIR = PROJECT_ROOT / "data/processed/yolo_pose_keypoints_all_v26"
OUTPUT_DIR = PROJECT_ROOT / "results/figures/keypoint_slide_examples"

FRAME_INDEX = 8
CONF_THRESHOLD = 0.25
TOP_CORRECT_PER_CLASS = 80
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


def pose_vector(pose: np.ndarray) -> np.ndarray:
    vector = pose[:, :2].copy()
    return np.nan_to_num(vector, nan=0.0).reshape(-1)


def load_pose(sample_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = KEYPOINT_DIR / f"{sample_id}.npy"
    if not path.exists():
        return None
    pose = normalize_pose(np.load(path))
    if pose is None:
        return None
    return pose, pose_vector(pose)


def select_correct_high_confidence_candidates() -> dict[str, list[dict[str, object]]]:
    selected_features = load_feature_list(FEATURE_LIST_PATH)
    X, y, class_ids, class_names, data = prepare_data(DATASET_PATH, "original", selected_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    del X_train, y_train

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    class_name_by_id = dict(zip(class_ids, class_names))
    sample_rows = data.loc[X_test.index, ["sample_id", "label"]].copy()
    sample_rows["true_id"] = y_test.to_numpy()
    sample_rows["pred_id"] = y_pred
    sample_rows["confidence"] = probabilities.max(axis=1)
    sample_rows["correct"] = sample_rows["true_id"].eq(sample_rows["pred_id"])

    candidates: dict[str, list[dict[str, object]]] = {}
    for class_id, label in class_name_by_id.items():
        class_rows = sample_rows[
            sample_rows["correct"] & sample_rows["true_id"].eq(class_id)
        ].sort_values("confidence", ascending=False)
        class_candidates = []
        for row in class_rows.itertuples(index=False):
            loaded = load_pose(str(row.sample_id))
            if loaded is None:
                continue
            pose, vector = loaded
            class_candidates.append(
                {
                    "sample_id": str(row.sample_id),
                    "label": label,
                    "confidence": float(row.confidence),
                    "pose": pose,
                    "vector": vector,
                }
            )
            if len(class_candidates) >= TOP_CORRECT_PER_CLASS:
                break
        candidates[label] = class_candidates

    return candidates


def select_distinct_triplet(candidates: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, object]]:
    for label in LABELS:
        if not candidates.get(label):
            raise RuntimeError(f"Sem candidatos corretos com pose valida para {label}")

    best_score = -np.inf
    best_triplet: tuple[dict[str, object], dict[str, object], dict[str, object]] | None = None
    for defense, walk, shoot in product(candidates["defense"], candidates["walk"], candidates["shoot"]):
        d_vec = defense["vector"]
        w_vec = walk["vector"]
        s_vec = shoot["vector"]
        pose_distance = (
            np.linalg.norm(d_vec - w_vec)
            + np.linalg.norm(d_vec - s_vec)
            + np.linalg.norm(w_vec - s_vec)
        ) / 3
        confidence = (
            float(defense["confidence"])
            + float(walk["confidence"])
            + float(shoot["confidence"])
        ) / 3
        score = pose_distance + 0.25 * confidence
        if score > best_score:
            best_score = score
            best_triplet = (defense, walk, shoot)

    if best_triplet is None:
        raise RuntimeError("Nao foi possivel selecionar trio distinto.")
    return dict(zip(LABELS, best_triplet))


def draw_pose(ax, pose: np.ndarray, title: str, sample_id: str, confidence: float) -> None:
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
    ax.text(
        0.5,
        -0.03,
        f"sample_id: {sample_id} | conf: {confidence:.2f}",
        ha="center",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 2.6)
    ax.axis("off")


def main() -> None:
    candidates = select_correct_high_confidence_candidates()
    triplet = select_distinct_triplet(candidates)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), facecolor="white")
    fig.suptitle("keypoints - exemplos de amostras", fontsize=18, y=0.96)
    for ax, label in zip(axes, LABELS):
        item = triplet[label]
        draw_pose(
            ax,
            item["pose"],
            label,
            str(item["sample_id"]),
            float(item["confidence"]),
        )

    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.9])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "keypoints_3_classes_defense_walk_shoot_distinct.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura salva em: {output_path.relative_to(PROJECT_ROOT)}")
    for label in LABELS:
        item = triplet[label]
        print(f"{label}: {item['sample_id']} conf={float(item['confidence']):.4f}")


if __name__ == "__main__":
    main()
