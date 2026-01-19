import os
import time
import random
import shutil
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from dotenv import load_dotenv
import pytesseract
import cv2
from pdf2image import convert_from_path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# --------------------------------------------------
# Environment (OCR)
# --------------------------------------------------

load_dotenv()

POPPLER_PATH = os.getenv("POPPLER_PATH", "")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# --------------------------------------------------
# File helpers
# --------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_files_recursive(root: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(r, f))
    return out


# --------------------------------------------------
# OCR extraction
# --------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    OCR-based extraction. Works for scanned PDFs and also for normal PDFs.
    """
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH if POPPLER_PATH else None)
    full_text = []

    for i, page in enumerate(pages):
        temp_img = f"_tmp_page_{os.getpid()}_{i}.jpg"
        page.save(temp_img, "JPEG")

        img = cv2.imread(temp_img)
        text = pytesseract.image_to_string(img, lang="eng")
        full_text.append(text)

        try:
            os.remove(temp_img)
        except:
            pass

    return "\n".join(full_text).strip()


def ocr_pdfs_to_text(
    input_root: str,
    output_root: str,
    max_files: int | None = None
) -> pd.DataFrame:
    """
    Converts all PDFs under input_root to .txt files under output_root, preserving subfolders.
    Returns a dataframe with columns: source_pdf, output_txt, category
    """
    ensure_dir(output_root)

    pdfs = list_files_recursive(input_root, (".pdf",))
    pdfs.sort()

    if max_files is not None:
        pdfs = pdfs[:max_files]

    rows = []

    for idx, pdf in enumerate(pdfs, start=1):
        rel = os.path.relpath(pdf, input_root)
        category = rel.split(os.sep)[0] if os.sep in rel else "Unknown"

        out_txt_path = os.path.join(output_root, os.path.splitext(rel)[0] + ".txt")
        ensure_dir(os.path.dirname(out_txt_path))

        try:
            text = extract_text_from_pdf(pdf)

            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            rows.append({
                "source_pdf": pdf,
                "output_txt": out_txt_path,
                "category": category,
                "ok": True
            })
        except Exception as e:
            rows.append({
                "source_pdf": pdf,
                "output_txt": out_txt_path,
                "category": category,
                "ok": False,
                "error": str(e)
            })

        if idx % 10 == 0:
            print(f"[OCR] Processed {idx}/{len(pdfs)} PDFs...")

    return pd.DataFrame(rows)


# --------------------------------------------------
# Dataset build
# --------------------------------------------------

def sample_normal_texts(dataset_cleaned_root: str, n_samples: int, seed: int = 42) -> List[str]:
    """
    Samples .txt files from dataset_cleaned/**.txt
    """
    random.seed(seed)
    all_txt = list_files_recursive(dataset_cleaned_root, (".txt",))
    if len(all_txt) == 0:
        raise RuntimeError(f"No .txt files found under: {dataset_cleaned_root}")

    if n_samples > len(all_txt):
        n_samples = len(all_txt)

    return random.sample(all_txt, n_samples)


def build_experiment_labels(
    normal_txt_paths: List[str],
    anomaly_txt_paths: List[str],
    output_csv: str
) -> pd.DataFrame:
    """
    Creates labels.csv with columns: path,label,split,source
    label: 0 normal, 1 anomaly
    """
    rows = []
    for p in normal_txt_paths:
        rows.append({"path": p, "label": 0, "source": "normal"})
    for p in anomaly_txt_paths:
        rows.append({"path": p, "label": 1, "source": "anomaly"})

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    ensure_dir(os.path.dirname(output_csv))
    df.to_csv(output_csv, index=False)
    return df


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# --------------------------------------------------
# Metrics + plots
# --------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    scores: higher = more anomalous
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    roc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    pr = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }


def save_confusion_matrix(y_true, y_pred, out_png: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    ensure_dir(os.path.dirname(out_png))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_roc_pr_curves(results: Dict[str, Dict], out_roc: str, out_pr: str):
    """
    results[model] = {"y_true":..., "scores":...}
    """
    from sklearn.metrics import roc_curve, precision_recall_curve

    # ROC
    fig1 = plt.figure(figsize=(7, 5))
    ax1 = fig1.add_subplot(111)
    for name, d in results.items():
        y_true = d["y_true"]
        scores = d["scores"]
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax1.plot(fpr, tpr, label=name)
    ax1.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax1.set_title("ROC Curves (Anomaly Detection)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()
    ensure_dir(os.path.dirname(out_roc))
    fig1.tight_layout()
    fig1.savefig(out_roc, dpi=200)
    plt.close(fig1)

    # PR
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(111)
    for name, d in results.items():
        y_true = d["y_true"]
        scores = d["scores"]
        p, r, _ = precision_recall_curve(y_true, scores)
        ax2.plot(r, p, label=name)
    ax2.set_title("Precision-Recall Curves (Anomaly Detection)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    ensure_dir(os.path.dirname(out_pr))
    fig2.tight_layout()
    fig2.savefig(out_pr, dpi=200)
    plt.close(fig2)


def save_pca_scatter(X_dense: np.ndarray, y_true: np.ndarray, out_png: str):
    """
    Simple 2D PCA visualization: Normal vs Anomaly
    """
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_dense)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111)

    normal = y_true == 0
    anomaly = y_true == 1

    ax.scatter(X2[normal, 0], X2[normal, 1], alpha=0.6, label="Normal")
    ax.scatter(X2[anomaly, 0], X2[anomaly, 1], alpha=0.8, label="Anomaly")

    ax.set_title("PCA Visualization of Document Features")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()

    ensure_dir(os.path.dirname(out_png))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
