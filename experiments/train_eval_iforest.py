import os
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

from utils import ensure_dir, read_text_file, compute_metrics, save_confusion_matrix

def main():
    labels_csv = os.path.join("experiments", "data", "labels.csv")
    results_dir = os.path.join("experiments", "results")
    ensure_dir(results_dir)

    df = pd.read_csv(labels_csv)
    texts = [read_text_file(p) for p in df["path"].tolist()]
    y_true = df["label"].values.astype(int)

    # TF-IDF (experiment-local)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)

    # Train IF on NORMAL ONLY (recommended)
    X_train = X[y_true == 0]

    t0 = time.time()
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_train)
    train_time = time.time() - t0

    # Scores: higher => more anomalous
    t1 = time.time()
    raw_scores = model.decision_function(X)  # higher = more normal
    scores = -raw_scores                     # higher = more anomalous
    infer_time = time.time() - t1

    # Threshold by percentile of normal scores
    normal_scores = scores[y_true == 0]
    threshold = float(np.percentile(normal_scores, 95))
    y_pred = (scores >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, scores)
    metrics.update({
        "model": "IsolationForest",
        "threshold": threshold,
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "n_samples": len(y_true)
    })

    # Save confusion
    save_confusion_matrix(
        y_true, y_pred,
        out_png=os.path.join(results_dir, "confusion_iforest.png"),
        title="Confusion Matrix - Isolation Forest"
    )

    # Save model artifacts (optional)
    with open(os.path.join(results_dir, "iforest_model.pkl"), "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "model": model, "threshold": threshold}, f)

    # Save per-sample scores (optional)
    out_scores = df.copy()
    out_scores["score"] = scores
    out_scores["pred"] = y_pred
    out_scores.to_csv(os.path.join(results_dir, "scores_iforest.csv"), index=False)

    print(metrics)
    return metrics, y_true, scores

if __name__ == "__main__":
    main()
