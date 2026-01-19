import os
import time
import pandas as pd
import numpy as np

from utils import (
    ensure_dir,
    save_roc_pr_curves,
    save_pca_scatter
)

from build_synthetic_dataset import main as build_dataset
from train_eval_iforest import main as run_iforest
from train_eval_kmeans import main as run_kmeans
from train_eval_autoencoder import main as run_autoencoder


def main():
    results_dir = os.path.join("experiments", "results")
    ensure_dir(results_dir)

    # 1) Build dataset (OCR anomalies + sample normals + labels.csv)
    print("\n=== STEP 1: BUILD EXPERIMENT DATASET ===")
    build_dataset()

    # 2) Train/Eval models
    print("\n=== STEP 2: Isolation Forest ===")
    m_if, y_true_if, s_if = run_iforest()

    print("\n=== STEP 3: KMeans Distance ===")
    m_km, y_true_km, s_km = run_kmeans()

    print("\n=== STEP 4: Autoencoder (PyTorch) ===")
    m_ae, y_true_ae, s_ae, X_dense = run_autoencoder()

    # 3) Save summary table
    summary = pd.DataFrame([m_if, m_km, m_ae])
    summary_path = os.path.join(results_dir, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Metrics summary saved: {summary_path}")

    # 4) Save ROC/PR curves
    curves = {
        "IsolationForest": {"y_true": y_true_if, "scores": s_if},
        "KMeansDistance": {"y_true": y_true_km, "scores": s_km},
        "AutoEncoder": {"y_true": y_true_ae, "scores": s_ae},
    }
    save_roc_pr_curves(
        curves,
        out_roc=os.path.join(results_dir, "roc_curves.png"),
        out_pr=os.path.join(results_dir, "pr_curves.png")
    )
    print("[OK] ROC/PR curves saved.")

    # 5) PCA scatter (from AE dense X)
    save_pca_scatter(
        X_dense=X_dense,
        y_true=y_true_ae,
        out_png=os.path.join(results_dir, "anomaly_scatter_pca.png")
    )
    print("[OK] PCA scatter saved.")

    print("\nDONE. Check experiments/results/ for figures + metrics.\n")


if __name__ == "__main__":
    main()
