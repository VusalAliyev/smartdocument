import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import ensure_dir, read_text_file, compute_metrics, save_confusion_matrix


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # TF-IDF is within [0,1] after normalization-ish; works well enough
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def main():
    labels_csv = os.path.join("experiments", "data", "labels.csv")
    results_dir = os.path.join("experiments", "results")
    ensure_dir(results_dir)

    df = pd.read_csv(labels_csv)
    texts = [read_text_file(p) for p in df["path"].tolist()]
    y_true = df["label"].values.astype(int)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_sparse = vectorizer.fit_transform(texts)
    X = X_sparse.toarray().astype(np.float32)

    # Train AE on NORMAL ONLY
    X_train = X[y_true == 0]
    X_all = X

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoder(input_dim=X.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 128
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=batch_size,
        shuffle=True
    )

    # Training
    epochs = 12
    t0 = time.time()
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(X_train)
        if ep % 2 == 0:
            print(f"[AE] Epoch {ep}/{epochs} - loss: {avg_loss:.6f}")

    train_time = time.time() - t0

    # Inference: reconstruction error per sample
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_all).to(device)
        recon = model(x_tensor).cpu().numpy()
    infer_time = time.time() - t1

    # Score = reconstruction MSE (higher => more anomalous)
    scores = np.mean((X_all - recon) ** 2, axis=1)

    # Threshold by percentile of NORMAL scores
    normal_scores = scores[y_true == 0]
    threshold = float(np.percentile(normal_scores, 95))
    y_pred = (scores >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, scores)
    metrics.update({
        "model": "AutoEncoder",
        "threshold": threshold,
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "n_samples": len(y_true),
        "epochs": epochs
    })

    save_confusion_matrix(
        y_true, y_pred,
        out_png=os.path.join(results_dir, "confusion_autoencoder.png"),
        title="Confusion Matrix - Autoencoder"
    )

    # Save per-sample scores
    out_scores = df.copy()
    out_scores["score"] = scores
    out_scores["pred"] = y_pred
    out_scores.to_csv(os.path.join(results_dir, "scores_autoencoder.csv"), index=False)

    # Save torch model
    torch.save({
        "state_dict": model.state_dict(),
        "threshold": threshold,
        "vectorizer": vectorizer
    }, os.path.join(results_dir, "autoencoder_model.pt"))

    print(metrics)
    return metrics, y_true, scores, X_all

if __name__ == "__main__":
    main()
