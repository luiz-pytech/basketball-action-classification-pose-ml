from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SPACEJAM_DIR = RAW_DIR / "joints"
VIDEOS_DIR = SPACEJAM_DIR / "examples"
ANNOTATION_FILE = SPACEJAM_DIR / "annotation_dict.json"
LABELS_FILE = SPACEJAM_DIR / "labels_dict.json"
TEST_KEYS_FILE = SPACEJAM_DIR / "testset_keys_1lug2020.txt"

YOLO_POSE_DIR = PROCESSED_DIR / "yolo_pose_keypoints"
YOLO_POSE_MANIFEST = PROCESSED_DIR / "yolo_pose_manifest.csv"
YOLO_BALL_DIR = PROCESSED_DIR / "yolo_ball_detections_all_v26"
YOLO_BALL_MANIFEST = PROCESSED_DIR / "yolo_ball_manifest_all_v26.csv"
YOLO_BALL_FEATURES = PROCESSED_DIR / "ball_features_all_v26.csv"

RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2

LABELS = {
    0: "block",
    1: "pass",
    2: "run",
    3: "dribble",
    4: "shoot",
    5: "ball in hand",
    6: "defense",
    7: "pick",
    8: "no_action",
    9: "walk",
    10: "discard",
}

GROUP_MAP = {
    "pass": "ball_control",
    "dribble": "ball_control",
    "ball in hand": "ball_control",
    "block": "defensive_action",
    "defense": "defensive_action",
    "pick": "defensive_action",
    "run": "locomotion",
    "walk": "locomotion",
    "shoot": "shoot",
    "no_action": "no_action",
}

GROUP_ORDER = [
    "ball_control",
    "defensive_action",
    "locomotion",
    "no_action",
    "shoot",
]

GROUP_TO_ID = {label: idx for idx, label in enumerate(GROUP_ORDER)}
