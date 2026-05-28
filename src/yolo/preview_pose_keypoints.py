"""Create a contact-sheet preview of extracted pose keypoints over video frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, VIDEOS_DIR, YOLO_POSE_DIR  # noqa: E402


COCO_SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview extracted YOLO pose keypoints.")
    parser.add_argument("sample_id", help="Example: 0000000 or 0000000_flipped")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--pose-dir", type=Path, default=YOLO_POSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR / "pose_previews")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    return parser.parse_args()


def read_video_frames(video_path: Path, limit: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def draw_pose(frame: np.ndarray, keypoints: np.ndarray, conf_threshold: float) -> np.ndarray:
    out = frame.copy()

    for a, b in COCO_SKELETON:
        if np.any(np.isnan(keypoints[[a, b], :2])):
            continue
        if keypoints[a, 2] < conf_threshold or keypoints[b, 2] < conf_threshold:
            continue
        pa = tuple(np.round(keypoints[a, :2]).astype(int))
        pb = tuple(np.round(keypoints[b, :2]).astype(int))
        cv2.line(out, pa, pb, (0, 220, 255), 2, lineType=cv2.LINE_AA)

    for x, y, conf in keypoints:
        if np.isnan(x) or np.isnan(y) or conf < conf_threshold:
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), 3, (20, 255, 80), -1, lineType=cv2.LINE_AA)

    return out


def contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("No frames to render.")

    h, w = frames[0].shape[:2]
    resized = [cv2.resize(frame, (w, h)) for frame in frames]
    return np.concatenate(resized, axis=1)


def main() -> None:
    args = parse_args()
    video_path = args.videos_dir / f"{args.sample_id}.mp4"
    pose_path = args.pose_dir / f"{args.sample_id}.npy"

    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not pose_path.exists():
        raise SystemExit(f"Pose file not found: {pose_path}")

    frames = read_video_frames(video_path, args.frames)
    poses = np.load(pose_path)
    count = min(len(frames), len(poses), args.frames)
    rendered = [draw_pose(frames[i], poses[i], args.conf_threshold) for i in range(count)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.sample_id}_pose_preview.jpg"
    cv2.imwrite(str(output_path), contact_sheet(rendered))
    print(f"Saved preview: {output_path}")


if __name__ == "__main__":
    main()
