from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GROUP_MAP, GROUP_ORDER, GROUP_TO_ID, RANDOM_STATE, TEST_SIZE  # noqa: E402


MODEL_ALIASES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svc": "SVM (SVC)",
    "mlp": "MLP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a classification experiment for one tabular dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--target", choices=["original", "grouped"], required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "decision_tree", "random_forest", "xgboost", "svc", "mlp"],
        choices=list(MODEL_ALIASES),
        help="Models to train. The svc alias uses sklearn.svm.SVC.",
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--no-sample-weight", action="store_true")
    parser.add_argument(
        "--feature-list",
        type=Path,
        default=None,
        help="Optional .txt, .csv or .json file with feature columns to keep.",
    )
    parser.add_argument(
        "--split-column",
        default=None,
        help="Optional dataset column with predefined split labels.",
    )
    parser.add_argument("--train-split-value", default="train")
    parser.add_argument("--test-split-value", default="test")
    return parser.parse_args()


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def load_feature_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            loaded = loaded.get("features", loaded.get("feature_columns"))
        if not isinstance(loaded, list):
            raise ValueError(f"Lista de features invalida em {path}")
        return [str(item) for item in loaded]

    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
        for column in ["feature", "column", "feature_name"]:
            if column in data.columns:
                return data[column].dropna().astype(str).tolist()
        if data.shape[1] == 1:
            return data.iloc[:, 0].dropna().astype(str).tolist()
        raise ValueError(f"CSV de features precisa ter coluna feature, column ou feature_name: {path}")

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def ensure_group_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "group_label" not in data.columns:
        data["group_label"] = data["label"].map(GROUP_MAP)
    if "group_id" not in data.columns:
        data["group_id"] = data["group_label"].map(GROUP_TO_ID)

    missing = data["group_label"].isna()
    if missing.any():
        unknown = sorted(data.loc[missing, "label"].dropna().unique())
        raise ValueError(f"Labels sem grupo definido: {unknown}")

    data["group_id"] = data["group_id"].astype(int)
    return data


def prepare_data(
    dataset_path: Path,
    target: str,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[int], list[str], pd.DataFrame]:
    data = pd.read_csv(dataset_path, dtype={"sample_id": str})
    if "quality_flag" in data.columns:
        data = data[~data["quality_flag"].astype(bool)].copy()

    if target == "grouped":
        data = ensure_group_columns(data)
        target_col = "group_id"
        class_ids = [idx for idx, label in enumerate(GROUP_ORDER) if label in set(data["group_label"])]
        class_names = [GROUP_ORDER[idx] for idx in class_ids]
    else:
        target_col = "label_id"
        class_info = (
            data[["label_id", "label"]]
            .drop_duplicates()
            .sort_values("label_id")
            .set_index("label_id")["label"]
            .to_dict()
        )
        class_ids = sorted(int(class_id) for class_id in data[target_col].dropna().unique())
        class_names = [class_info[class_id] for class_id in class_ids]

    meta_cols = [
        "sample_id",
        "label",
        "label_id",
        "group_label",
        "group_id",
        "frame_count",
        "detected_frame_count",
        "missing_frame_ratio",
        "mean_keypoint_conf",
        "pose_conf_mean",
        "pose_conf_min",
        "pose_conf_std",
        "quality_flag",
        "source_sample_id",
        "augmentation",
        "split",
    ]
    drop_cols = [col for col in meta_cols if col in data.columns]
    feature_cols = [col for col in data.columns if col not in drop_cols]
    X = data[feature_cols].select_dtypes(include=["number"]).copy()
    if selected_features is not None:
        missing_features = [feature for feature in selected_features if feature not in X.columns]
        if missing_features:
            preview = missing_features[:10]
            raise ValueError(f"Features selecionadas ausentes no dataset: {preview}")
        X = X[selected_features].copy()
    y = data[target_col].astype(int).copy()

    return X, y, class_ids, class_names, data


def build_models(model_keys: list[str], numeric_features: list[str], random_state: int):
    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), numeric_features)],
        remainder="drop",
    )

    available = {
        "logistic_regression": (
            scaled_preprocessor,
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
        ),
        "decision_tree": (
            tree_preprocessor,
            DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
        ),
        "random_forest": (
            tree_preprocessor,
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        "svc": (
            scaled_preprocessor,
            SVC(class_weight="balanced", random_state=random_state, probability=True, cache_size=1000),
        ),
        "mlp": (
            scaled_preprocessor,
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
            ),
        ),
    }

    if XGBClassifier is not None:
        available["xgboost"] = (
            tree_preprocessor,
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
            ),
        )

    return {MODEL_ALIASES[key]: available[key] for key in model_keys if key in available}


def get_scores(estimator, X_data: pd.DataFrame):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_data)
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X_data)
    return None


def compute_auc(y_true, y_score, labels: list[int]) -> float:
    if y_score is None:
        return np.nan
    try:
        return float(
            roc_auc_score(
                y_true,
                y_score,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        return np.nan


def model_supports_sample_weight(model) -> bool:
    try:
        return "sample_weight" in inspect.signature(model.fit).parameters
    except Exception:
        return False


def save_confusion_matrix(
    experiment_name: str,
    model_name: str,
    y_true,
    y_pred,
    class_ids: list[int],
    class_names: list[str],
    root_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, Path]:
    confusion_dir = figures_dir / f"confusion_matrices_{experiment_name}"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    table_path = tables_dir / f"confusion_matrix_{experiment_name}_{safe_name(model_name)}.csv"
    fig_path = confusion_dir / f"confusion_matrix_{experiment_name}_{safe_name(model_name)}.png"
    cm_df.to_csv(table_path)

    size = max(6, min(12, len(class_names) + 2))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name} - Matriz de confusão")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return table_path.relative_to(root_dir), fig_path.relative_to(root_dir)


def main() -> None:
    args = parse_args()
    root_dir = PROJECT_ROOT
    tables_dir = root_dir / "results" / "tables"
    figures_dir = root_dir / "results" / "figures"
    models_dir = root_dir / "results" / "models"
    for path in [tables_dir, figures_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)

    dataset_path = args.dataset if args.dataset.is_absolute() else root_dir / args.dataset
    selected_features = None
    feature_list_path = None
    if args.feature_list is not None:
        feature_list_path = args.feature_list if args.feature_list.is_absolute() else root_dir / args.feature_list
        selected_features = load_feature_list(feature_list_path)

    X, y, class_ids, class_names, data = prepare_data(dataset_path, args.target, selected_features)

    if args.split_column is not None:
        if args.split_column not in data.columns:
            raise ValueError(f"Split column not found in dataset: {args.split_column}")
        split_values = data[args.split_column].astype(str)
        train_mask = split_values == args.train_split_value
        test_mask = split_values == args.test_split_value
        if not train_mask.any() or not test_mask.any():
            raise ValueError(
                f"Split column {args.split_column} must contain "
                f"{args.train_split_value!r} and {args.test_split_value!r} rows."
            )
        X_train, X_test = X.loc[train_mask].copy(), X.loc[test_mask].copy()
        y_train, y_test = y.loc[train_mask].copy(), y.loc[test_mask].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )

    models = build_models(args.models, list(X.columns), args.random_state)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print(f"Experimento: {args.experiment_name}")
    print(f"Dataset: {dataset_path.relative_to(root_dir)}")
    print(f"Target: {args.target}")
    print(f"Rows apos filtro: {len(data)}")
    print(f"Features: {X.shape[1]}")
    if feature_list_path is not None:
        print(f"Feature list: {feature_list_path.relative_to(root_dir)}")
    print(f"X_train: {X_train.shape} X_test: {X_test.shape}")
    print("Classes:", dict(zip(class_ids, class_names)))
    print("Modelos:", list(models.keys()))

    results = []
    trained_models = {}
    predictions = {}

    for model_name, (preprocessor, model) in models.items():
        print(f"Treinando {args.experiment_name}: {model_name}")
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        fit_kwargs = {}
        weighted_model_without_class_weight = model_name in {"XGBoost", "MLP"}
        if (
            not args.no_sample_weight
            and weighted_model_without_class_weight
            and model_supports_sample_weight(model)
        ):
            fit_kwargs["model__sample_weight"] = sample_weight

        start_fit = time.perf_counter()
        pipe.fit(X_train, y_train, **fit_kwargs)
        fit_time = time.perf_counter() - start_fit

        start_pred = time.perf_counter()
        y_pred = pipe.predict(X_test)
        y_score = get_scores(pipe, X_test)
        predict_time = time.perf_counter() - start_pred

        table_path, fig_path = save_confusion_matrix(
            args.experiment_name,
            model_name,
            y_test,
            y_pred,
            class_ids,
            class_names,
            root_dir,
            tables_dir,
            figures_dir,
        )

        results.append(
            {
                "experiment": args.experiment_name,
                "dataset": dataset_path.name,
                "target": args.target,
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "roc_auc_ovr_macro": compute_auc(y_test, y_score, class_ids),
                "fit_time": fit_time,
                "predict_time": predict_time,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": X.shape[1],
                "confusion_matrix_csv": str(table_path),
                "confusion_matrix_png": str(fig_path),
            }
        )
        trained_models[model_name] = pipe
        predictions[model_name] = y_pred

    results_df = pd.DataFrame(results).sort_values(["f1_macro", "accuracy"], ascending=False)
    results_path = tables_dir / f"classification_results_{args.experiment_name}.csv"
    results_df.to_csv(results_path, index=False)

    best_model_name = str(results_df.iloc[0]["model"])
    report_dict = classification_report(
        y_test,
        predictions[best_model_name],
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T
    report_path = tables_dir / f"classification_report_best_model_{args.experiment_name}.csv"
    report_df.to_csv(report_path)

    model_path = models_dir / f"best_classification_model_{args.experiment_name}.joblib"
    joblib.dump(trained_models[best_model_name], model_path)

    metadata = {
        "experiment": args.experiment_name,
        "dataset": str(dataset_path.relative_to(root_dir)),
        "target": args.target,
        "models": list(models.keys()),
        "best_model": best_model_name,
        "class_ids": class_ids,
        "class_names": class_names,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
        "feature_list": str(feature_list_path.relative_to(root_dir)) if feature_list_path else None,
        "feature_columns": list(X.columns),
        "metric_priority": "f1_macro",
    }
    metadata_path = tables_dir / f"classification_metadata_{args.experiment_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Resultados salvos em:", results_path.relative_to(root_dir))
    print("Report salvo em:", report_path.relative_to(root_dir))
    print("Modelo salvo em:", model_path.relative_to(root_dir))
    print("Melhor modelo:", best_model_name)
    print(results_df[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
