from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_STATE  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "data/processed/dataset_basquet_defense_walk_shoot_pruned80_all_v26.csv"
DEFAULT_FEATURE_LIST = (
    PROJECT_ROOT / "results/feature_lists/feature_audit_semantic_pair_all_v26_original_pruned_80.txt"
)
DEFAULT_EXPERIMENT = "final_3_classes_defense_walk_shoot_pruned80_all_v26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA and DBSCAN for the final 3-class dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--feature-list", type=Path, default=DEFAULT_FEATURE_LIST)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--min-samples", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--eps-percentiles", nargs="+", type=float, default=[80, 90, 95])
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_feature_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def cluster_purity(labels_true: pd.Series, labels_pred: np.ndarray) -> float:
    table = pd.crosstab(labels_pred, labels_true)
    if table.empty:
        return float("nan")
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def count_clusters(labels_pred: np.ndarray) -> int:
    return len(set(labels_pred) - {-1})


def safe_silhouette(X_values: np.ndarray, labels_pred: np.ndarray) -> float:
    non_noise = labels_pred != -1
    labels_non_noise = labels_pred[non_noise]
    if len(np.unique(labels_non_noise)) < 2:
        return float("nan")
    return float(silhouette_score(X_values[non_noise], labels_non_noise))


def external_metrics(labels_true: pd.Series, labels_pred: np.ndarray) -> dict[str, float]:
    return {
        "adjusted_rand": adjusted_rand_score(labels_true, labels_pred),
        "normalized_mutual_info": normalized_mutual_info_score(labels_true, labels_pred),
        "homogeneity": homogeneity_score(labels_true, labels_pred),
        "completeness": completeness_score(labels_true, labels_pred),
        "v_measure": v_measure_score(labels_true, labels_pred),
        "purity": cluster_purity(labels_true, labels_pred),
    }


def dbscan_grid(
    X_dbscan: np.ndarray,
    y_dbscan: pd.Series,
    min_samples_candidates: list[int],
    eps_percentiles: list[float],
) -> pd.DataFrame:
    eps_candidates: set[float] = set()
    for min_samples in min_samples_candidates:
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X_dbscan)
        distances, _ = neighbors.kneighbors(X_dbscan)
        kth_distances = np.sort(distances[:, -1])
        for percentile in eps_percentiles:
            eps_candidates.add(round(float(np.percentile(kth_distances, percentile)), 4))

    rows = []
    for eps in sorted(eps_candidates):
        for min_samples in min_samples_candidates:
            clusters = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_dbscan)
            metrics = external_metrics(y_dbscan, clusters)
            rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "sample_rows": len(X_dbscan),
                    "n_clusters": count_clusters(clusters),
                    "noise_ratio": float(np.mean(clusters == -1)),
                    "silhouette_non_noise": safe_silhouette(X_dbscan, clusters),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def choose_best_dbscan(grid: pd.DataFrame) -> pd.Series:
    viable = grid[grid["n_clusters"] >= 2].copy()
    if viable.empty:
        viable = grid.copy()
    return viable.sort_values(
        ["normalized_mutual_info", "adjusted_rand", "noise_ratio"],
        ascending=[False, False, True],
    ).iloc[0]


def save_scatter(
    plot_df: pd.DataFrame,
    color_col: str,
    title: str,
    output_path: Path,
    explained_variance: np.ndarray,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue=color_col,
        palette="tab10",
        s=18,
        alpha=0.72,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(title, fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}%)")
    ax.legend(title=color_col, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.grid(color="#d7dee2", linewidth=1.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pca_cumulative_plot(variance_df: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(
        variance_df["component_number"],
        variance_df["cumulative_explained_variance_ratio"],
        marker="o",
        markersize=4,
        linewidth=2,
        color="#2563eb",
    )
    for threshold in [0.8, 0.9, 0.95]:
        ax.axhline(threshold, color="#6b7280", linestyle="--", linewidth=1)
        ax.text(
            variance_df["component_number"].max(),
            threshold + 0.006,
            f"{threshold:.0%}",
            ha="right",
            va="bottom",
            fontsize=12,
            color="#374151",
        )
    ax.set_title("Variancia explicada acumulada - PCA 3 classes", fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("Numero de componentes")
    ax.set_ylabel("Variancia explicada acumulada")
    ax.set_ylim(0, 1.02)
    ax.grid(color="#d7dee2", linewidth=1.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_3d_scatter(
    plot_df: pd.DataFrame,
    color_col: str,
    title: str,
    output_path: Path,
    explained_variance: np.ndarray,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    categories = sorted(plot_df[color_col].astype(str).unique())
    palette = sns.color_palette("tab10", n_colors=max(len(categories), 3))
    color_map = dict(zip(categories, palette))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    for category in categories:
        subset = plot_df[plot_df[color_col].astype(str) == category]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            subset["PC3"],
            label=category,
            s=14,
            alpha=0.68,
            linewidth=0,
            color=color_map[category],
        )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=18)
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}%)", labelpad=12)
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}%)", labelpad=12)
    ax.set_zlabel(f"PC3 ({explained_variance[2] * 100:.1f}%)", labelpad=12)
    ax.view_init(elev=22, azim=-62)
    ax.legend(title=color_col, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    feature_list_path = resolve_path(args.feature_list)
    tables_dir = PROJECT_ROOT / "results/tables"
    figures_dir = PROJECT_ROOT / "results/figures" / args.experiment_name
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(dataset_path, dtype={"sample_id": str})
    if "quality_flag" in data.columns:
        data = data[~data["quality_flag"].astype(bool)].copy()

    features = load_feature_list(feature_list_path)
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Features ausentes no dataset: {missing_features[:10]}")

    X = data[features].copy()
    y = data["label"].astype(str).copy()

    X_imputed = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputed)

    rng = np.random.default_rng(args.random_state)
    sample_size = min(args.sample_size, X_scaled.shape[0])
    sample_idx = rng.choice(X_scaled.shape[0], size=sample_size, replace=False)

    X_dbscan = X_scaled[sample_idx]
    y_dbscan = y.iloc[sample_idx].reset_index(drop=True)
    grid = dbscan_grid(X_dbscan, y_dbscan, args.min_samples, args.eps_percentiles)
    best = choose_best_dbscan(grid)
    dbscan_clusters = DBSCAN(
        eps=float(best["eps"]),
        min_samples=int(best["min_samples"]),
        n_jobs=-1,
    ).fit_predict(X_dbscan)

    dbscan_metrics = external_metrics(y_dbscan, dbscan_clusters)
    dbscan_metrics.update(
        {
            "eps": float(best["eps"]),
            "min_samples": int(best["min_samples"]),
            "rows": sample_size,
            "n_clusters": count_clusters(dbscan_clusters),
            "noise_ratio": float(np.mean(dbscan_clusters == -1)),
            "silhouette_non_noise": safe_silhouette(X_dbscan, dbscan_clusters),
        }
    )

    pca = PCA(n_components=2, random_state=args.random_state)
    pca_coords = pca.fit_transform(X_scaled)
    pca_3d = PCA(n_components=3, random_state=args.random_state)
    pca_3d_coords = pca_3d.fit_transform(X_scaled)
    pca_full = PCA(random_state=args.random_state)
    pca_full.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    variance_df = pd.DataFrame(
        {
            "component_number": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
            "component": [f"PC{idx}" for idx in range(1, len(pca_full.explained_variance_ratio_) + 1)],
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": cumulative_variance,
        }
    )
    class_plot_df = pd.DataFrame(
        {
            "PC1": pca_coords[sample_idx, 0],
            "PC2": pca_coords[sample_idx, 1],
            "label": y.iloc[sample_idx].to_numpy(),
        }
    )
    dbscan_plot_df = class_plot_df.copy()
    dbscan_plot_df["cluster_dbscan"] = dbscan_clusters.astype(str)
    class_3d_plot_df = pd.DataFrame(
        {
            "PC1": pca_3d_coords[sample_idx, 0],
            "PC2": pca_3d_coords[sample_idx, 1],
            "PC3": pca_3d_coords[sample_idx, 2],
            "label": y.iloc[sample_idx].to_numpy(),
        }
    )
    dbscan_3d_plot_df = class_3d_plot_df.copy()
    dbscan_3d_plot_df["cluster_dbscan"] = dbscan_clusters.astype(str)

    prefix = args.experiment_name
    grid_path = tables_dir / f"unsupervised_dbscan_grid_{prefix}.csv"
    metrics_path = tables_dir / f"unsupervised_dbscan_metrics_{prefix}.csv"
    pca_variance_path = tables_dir / f"pca_explained_variance_{prefix}.csv"
    pca_full_variance_path = tables_dir / f"pca_full_explained_variance_{prefix}.csv"
    assignments_path = tables_dir / f"unsupervised_dbscan_assignments_{prefix}.csv"
    class_fig_path = figures_dir / f"pca_original_labels_{prefix}.png"
    dbscan_fig_path = figures_dir / f"pca_dbscan_clusters_{prefix}.png"
    class_3d_fig_path = figures_dir / f"pca_3d_original_labels_{prefix}.png"
    dbscan_3d_fig_path = figures_dir / f"pca_3d_dbscan_clusters_{prefix}.png"
    cumulative_fig_path = figures_dir / f"pca_cumulative_explained_variance_{prefix}.png"
    summary_path = tables_dir / f"pca_dbscan_summary_{prefix}.json"

    grid.to_csv(grid_path, index=False)
    pd.DataFrame([dbscan_metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame(
        {
            "component": ["PC1", "PC2", "PC3"],
            "explained_variance_ratio": pca_3d.explained_variance_ratio_,
        }
    ).to_csv(pca_variance_path, index=False)
    variance_df.to_csv(pca_full_variance_path, index=False)
    pd.DataFrame(
        {
            "sample_id": data["sample_id"].iloc[sample_idx].to_numpy(),
            "label": y.iloc[sample_idx].to_numpy(),
            "cluster_dbscan": dbscan_clusters,
        }
    ).to_csv(assignments_path, index=False)

    save_scatter(
        class_plot_df,
        "label",
        "PCA 2D por classe real - 3 classes",
        class_fig_path,
        pca.explained_variance_ratio_,
    )
    save_scatter(
        dbscan_plot_df,
        "cluster_dbscan",
        "PCA 2D por DBSCAN - 3 classes",
        dbscan_fig_path,
        pca.explained_variance_ratio_,
    )
    save_3d_scatter(
        class_3d_plot_df,
        "label",
        "PCA 3D por classe real - 3 classes",
        class_3d_fig_path,
        pca_3d.explained_variance_ratio_,
    )
    save_3d_scatter(
        dbscan_3d_plot_df,
        "cluster_dbscan",
        "PCA 3D por DBSCAN - 3 classes",
        dbscan_3d_fig_path,
        pca_3d.explained_variance_ratio_,
    )
    save_pca_cumulative_plot(variance_df, cumulative_fig_path)

    threshold_components = {
        f"components_for_{int(threshold * 100)}pct": int(np.searchsorted(cumulative_variance, threshold) + 1)
        for threshold in [0.8, 0.9, 0.95]
    }

    summary = {
        "experiment": args.experiment_name,
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "feature_list": str(feature_list_path.relative_to(PROJECT_ROOT)),
        "rows_after_filter": len(data),
        "sample_rows": sample_size,
        "feature_count": len(features),
        "classes": sorted(y.unique().tolist()),
        "dbscan": dbscan_metrics,
        "pca_explained_variance": {
            "PC1": float(pca_3d.explained_variance_ratio_[0]),
            "PC2": float(pca_3d.explained_variance_ratio_[1]),
            "PC3": float(pca_3d.explained_variance_ratio_[2]),
            **threshold_components,
        },
        "outputs": {
            "dbscan_grid": str(grid_path.relative_to(PROJECT_ROOT)),
            "dbscan_metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
            "pca_variance": str(pca_variance_path.relative_to(PROJECT_ROOT)),
            "pca_full_variance": str(pca_full_variance_path.relative_to(PROJECT_ROOT)),
            "dbscan_assignments": str(assignments_path.relative_to(PROJECT_ROOT)),
            "pca_original_labels": str(class_fig_path.relative_to(PROJECT_ROOT)),
            "pca_dbscan_clusters": str(dbscan_fig_path.relative_to(PROJECT_ROOT)),
            "pca_3d_original_labels": str(class_3d_fig_path.relative_to(PROJECT_ROOT)),
            "pca_3d_dbscan_clusters": str(dbscan_3d_fig_path.relative_to(PROJECT_ROOT)),
            "pca_cumulative_variance": str(cumulative_fig_path.relative_to(PROJECT_ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset: {dataset_path.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(data)} Features: {len(features)} Sample DBSCAN/PCA plot: {sample_size}")
    print("PCA explained variance:")
    print(pd.read_csv(pca_variance_path).to_string(index=False))
    print("Componentes necessarios:")
    print(threshold_components)
    print("Best DBSCAN:")
    print(pd.DataFrame([dbscan_metrics]).to_string(index=False))
    print("Figuras:")
    print(class_fig_path.relative_to(PROJECT_ROOT))
    print(dbscan_fig_path.relative_to(PROJECT_ROOT))
    print(class_3d_fig_path.relative_to(PROJECT_ROOT))
    print(dbscan_3d_fig_path.relative_to(PROJECT_ROOT))
    print(cumulative_fig_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
