import os
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

from utils import ensure_dir, read_text_file, compute_metrics, save_confusion_matrix

def main():
    labels_csv = os.path.join("experiments", "data", "labels.csv")
    results_dir = os.path.join("experiments", "results")
    ensure_dir(results_dir)

    df = pd.read_csv(labels_csv)
    texts = [read_text_file(p) for p in df["path"].tolist()]
    y_true = df["label"].values.astype(int)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)

    # Train KMeans on NORMAL ONLY
    X_train = X[y_true == 0]

    K = 6  # baseline choice; can be an experiment parameter
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=2048)
    km.fit(X_train)
    train_time = time.time() - t0

    # Distance to nearest centroid => anomaly score
    t1 = time.time()
    distances = km.transform(X)  # shape [n, K]
    scores = distances.min(axis=1)  # higher => more anomalous
    infer_time = time.time() - t1

    normal_scores = scores[y_true == 0]
    threshold = float(np.percentile(normal_scores, 95))
    y_pred = (scores >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, scores)
    metrics.update({
        "model": "KMeansDistance",
        "threshold": threshold,
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "n_samples": len(y_true),
        "k": K
    })

    save_confusion_matrix(
        y_true, y_pred,
        out_png=os.path.join(results_dir, "confusion_kmeans.png"),
        title="Confusion Matrix - KMeans Distance"
    )

    with open(os.path.join(results_dir, "kmeans_model.pkl"), "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "model": km, "threshold": threshold, "k": K}, f)

    out_scores = df.copy()
    out_scores["score"] = scores
    out_scores["pred"] = y_pred
    out_scores.to_csv(os.path.join(results_dir, "scores_kmeans.csv"), index=False)

    print(metrics)
    return metrics, y_true, scores

if __name__ == "__main__":
    main()
