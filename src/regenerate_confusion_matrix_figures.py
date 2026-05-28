from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def regenerate_figure(confusion_csv: Path, figure_png: Path, model_name: str) -> None:
    cm_df = pd.read_csv(confusion_csv, index_col=0)

    size = max(6, min(12, len(cm_df.index) + 2))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name} - Matriz de confusão")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate confusion matrix PNGs from existing result CSV files."
    )
    parser.add_argument(
        "--results-glob",
        default="classification_results_final*.csv",
        help="Glob used inside results/tables to find result CSV files.",
    )
    args = parser.parse_args()

    regenerated = 0
    for results_path in sorted(TABLES_DIR.glob(args.results_glob)):
        results_df = pd.read_csv(results_path)
        required_columns = {"model", "confusion_matrix_csv", "confusion_matrix_png"}
        if not required_columns.issubset(results_df.columns):
            continue

        for row in results_df.itertuples(index=False):
            confusion_csv = PROJECT_ROOT / getattr(row, "confusion_matrix_csv")
            figure_png = PROJECT_ROOT / getattr(row, "confusion_matrix_png")
            model_name = getattr(row, "model")

            if not confusion_csv.exists():
                print(f"Ignorando CSV ausente: {confusion_csv.relative_to(PROJECT_ROOT)}")
                continue

            regenerate_figure(confusion_csv, figure_png, model_name)
            regenerated += 1

    print(f"Figuras regeneradas: {regenerated}")


if __name__ == "__main__":
    main()
