"""Extract YOLO Pose keypoints from SpaceJam clips and save them as .npy files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ANNOTATION_FILE,
    GROUP_MAP,
    GROUP_TO_ID,
    LABELS,
    PROCESSED_DIR,
    VIDEOS_DIR,
    YOLO_POSE_DIR,
    YOLO_POSE_MANIFEST,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 17 COCO body keypoints from SpaceJam .mp4 clips using YOLO Pose."
    )
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--annotations", type=Path, default=ANNOTATION_FILE)
    parser.add_argument("--output-dir", type=Path, default=YOLO_POSE_DIR)
    parser.add_argument("--manifest", type=Path, default=YOLO_POSE_MANIFEST)
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Ultralytics pose model. Use yolo11s-pose.pt for better quality, slower extraction.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N selected videos.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional JSON or txt list of sample ids.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Example: cpu, 0, 0,1. Default lets Ultralytics choose.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Frame stride for extraction.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write the manifest with expected pose paths. Does not run YOLO.",
    )
    parser.add_argument(
        "--person-strategy",
        choices=["largest_box", "highest_keypoint_conf"],
        default="largest_box",
        help="How to choose one player if YOLO detects more than one person.",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(sample_id): int(label_id) for sample_id, label_id in data.items()}


def load_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()

    if text[0] == "[":
        return {str(item) for item in json.loads(text)}

    return {line.strip() for line in text.splitlines() if line.strip()}


def selected_videos(
    videos_dir: Path,
    annotations: dict[str, int],
    ids: set[str] | None,
    start_index: int,
    limit: int | None,
) -> list[Path]:
    videos = []
    for path in sorted(videos_dir.glob("*.mp4")):
        sample_id = path.stem
        if sample_id not in annotations:
            continue
        if ids is not None and sample_id not in ids:
            continue
        videos.append(path)

    videos = videos[start_index:]
    if limit is not None:
        videos = videos[:limit]
    return videos


def grouped_label(label: str) -> tuple[int | str, str]:
    group = GROUP_MAP.get(label, "")
    if not group:
        return "", ""
    return GROUP_TO_ID[group], group


def result_keypoints(result) -> np.ndarray | None:
    if result.keypoints is None:
        return None

    data = result.keypoints.data
    if data is None or len(data) == 0:
        return None

    keypoints = data.cpu().numpy().astype(np.float32)
    if keypoints.shape[-1] == 2:
        conf = result.keypoints.conf
        if conf is None:
            conf_arr = np.ones(keypoints.shape[:2] + (1,), dtype=np.float32)
        else:
            conf_arr = conf.cpu().numpy().astype(np.float32)[..., None]
        keypoints = np.concatenate([keypoints, conf_arr], axis=-1)

    return keypoints


def choose_person_index(result, keypoints: np.ndarray, strategy: str) -> int:
    if strategy == "largest_box" and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == len(keypoints):
            widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
            heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
            areas = widths * heights
            return int(np.nanargmax(areas))

    scores = np.nanmean(keypoints[:, :, 2], axis=1)
    return int(np.nanargmax(scores))


def empty_frame(num_keypoints: int = 17) -> np.ndarray:
    return np.full((num_keypoints, 3), np.nan, dtype=np.float32)


def pose_stats(pose: np.ndarray) -> dict[str, float]:
    if pose.size == 0:
        return {
            "frame_count": int(pose.shape[0]),
            "detected_frame_count": 0,
            "missing_frame_ratio": math.nan,
            "mean_keypoint_conf": math.nan,
        }

    confidence = pose[:, :, 2]
    detected_frames = np.any(np.isfinite(confidence), axis=1)
    detected = int(detected_frames.sum())
    mean_conf = float(np.nanmean(confidence)) if confidence.size else math.nan
    return {
        "frame_count": int(pose.shape[0]),
        "detected_frame_count": detected,
        "missing_frame_ratio": float(1.0 - detected / pose.shape[0]) if pose.shape[0] else math.nan,
        "mean_keypoint_conf": mean_conf,
    }


def extract_video_keypoints(model, video_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float]]:
    predict_kwargs = {
        "source": str(video_path),
        "stream": True,
        "verbose": False,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "vid_stride": args.vid_stride,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    frames = []
    detected = 0

    for result in model.predict(**predict_kwargs):
        keypoints = result_keypoints(result)
        if keypoints is None:
            frames.append(empty_frame())
            continue

        person_idx = choose_person_index(result, keypoints, args.person_strategy)
        frames.append(keypoints[person_idx])
        detected += 1

    if not frames:
        pose = np.empty((0, 17, 3), dtype=np.float32)
    else:
        pose = np.stack(frames).astype(np.float32)

    stats = pose_stats(pose)
    stats["detected_frame_count"] = int(detected)
    return pose, stats


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "label_id",
        "label",
        "group_id",
        "group_label",
        "video_path",
        "pose_path",
        "frame_count",
        "detected_frame_count",
        "missing_frame_ratio",
        "mean_keypoint_conf",
        "status",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    ids = load_ids(args.ids_file)
    videos = selected_videos(args.videos_dir, annotations, ids, args.start_index, args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    needs_model = (
        not args.manifest_only
        and (args.overwrite or any(not (args.output_dir / f"{path.stem}.npy").exists() for path in videos))
    )
    model = None
    if needs_model:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc
        model = YOLO(args.model)

    rows: list[dict[str, object]] = []

    progress_label = "Writing manifest" if args.manifest_only else "Extracting YOLO pose"
    for video_path in tqdm(videos, desc=progress_label):
        sample_id = video_path.stem
        label_id = annotations[sample_id]
        label = LABELS.get(label_id, str(label_id))
        group_id, group_label = grouped_label(label)
        pose_path = args.output_dir / f"{sample_id}.npy"
        start = time.perf_counter()

        if args.manifest_only:
            if pose_path.exists():
                try:
                    stats = pose_stats(np.load(pose_path))
                    status = "existing_pose"
                except Exception as exc:
                    stats = {
                        "frame_count": "",
                        "detected_frame_count": "",
                        "missing_frame_ratio": "",
                        "mean_keypoint_conf": "",
                    }
                    status = f"error: {type(exc).__name__}: {exc}"
            else:
                stats = {
                    "frame_count": "",
                    "detected_frame_count": "",
                    "missing_frame_ratio": "",
                    "mean_keypoint_conf": "",
                }
                status = "missing_pose"

            rows.append(
                {
                    "sample_id": sample_id,
                    "label_id": label_id,
                    "label": label,
                    "group_id": group_id,
                    "group_label": group_label,
                    "video_path": str(video_path),
                    "pose_path": str(pose_path),
                    **stats,
                    "status": status,
                    "seconds": 0.0,
                }
            )
            continue

        if pose_path.exists() and not args.overwrite:
            try:
                stats = pose_stats(np.load(pose_path))
            except Exception:
                stats = {
                    "frame_count": "",
                    "detected_frame_count": "",
                    "missing_frame_ratio": "",
                    "mean_keypoint_conf": "",
                }
            rows.append(
                {
                    "sample_id": sample_id,
                    "label_id": label_id,
                    "label": label,
                    "group_id": group_id,
                    "group_label": group_label,
                    "video_path": str(video_path),
                    "pose_path": str(pose_path),
                    **stats,
                    "status": "skipped_existing",
                    "seconds": 0.0,
                }
            )
            continue

        try:
            if model is None:
                raise RuntimeError("YOLO model was not loaded.")
            pose, stats = extract_video_keypoints(model, video_path, args)
            np.save(pose_path, pose)
            status = "ok"
        except Exception as exc:  # keep long runs moving and report failures in the manifest
            stats = {
                "frame_count": "",
                "detected_frame_count": "",
                "missing_frame_ratio": "",
                "mean_keypoint_conf": "",
            }
            status = f"error: {type(exc).__name__}: {exc}"

        rows.append(
            {
                "sample_id": sample_id,
                "label_id": label_id,
                "label": label,
                "group_id": group_id,
                "group_label": group_label,
                "video_path": str(video_path),
                "pose_path": str(pose_path),
                **stats,
                "status": status,
                "seconds": round(time.perf_counter() - start, 3),
            }
        )

    write_manifest(args.manifest, rows)
    print(f"Processed entries: {len(rows)}")
    print(f"Pose output: {args.output_dir}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
