import os
import pandas as pd

from utils import (
    ensure_dir,
    ocr_pdfs_to_text,
    sample_normal_texts,
    build_experiment_labels,
    list_files_recursive
)

def main():
    # Paths
    anomaly_pdf_root = os.path.join("experiments", "data", "anomaly")
    anomaly_text_root = os.path.join("experiments", "data", "anomaly_text")
    normal_text_root = os.path.join("experiments", "data", "normal_text")
    labels_csv = os.path.join("experiments", "data", "labels.csv")

    dataset_cleaned_root = "dataset_cleaned"

    ensure_dir(os.path.join("experiments", "data"))
    ensure_dir(anomaly_text_root)
    ensure_dir(normal_text_root)

    # 1) OCR all anomaly PDFs -> anomaly_text/
    print("[1/3] OCR anomaly PDFs -> text ...")
    ocr_df = ocr_pdfs_to_text(anomaly_pdf_root, anomaly_text_root)
    ocr_report_csv = os.path.join("experiments", "data", "ocr_report.csv")
    ocr_df.to_csv(ocr_report_csv, index=False)
    print(f"[OK] OCR report saved: {ocr_report_csv}")

    anomaly_txt_paths = list_files_recursive(anomaly_text_root, (".txt",))

    # If OCR produced empty texts, still keep them for evaluation (these are valid "OCR problematic" anomalies)
    if len(anomaly_txt_paths) == 0:
        # Print a quick diagnostic
        print("[ERROR] No .txt files found under:", anomaly_text_root)
        print("Check whether OCR output path is correct and whether files are being written.")
        raise RuntimeError("No anomaly .txt files produced. Check OCR paths and output directory.")


    # 2) Sample normal texts from dataset_cleaned
    #    For strong evaluation, use 4x normal vs anomaly
    target_normal = min(2000, len(anomaly_txt_paths) * 4)

    print(f"[2/3] Sampling normal texts: {target_normal} samples ...")
    normal_txt_paths = sample_normal_texts(dataset_cleaned_root, n_samples=target_normal)

    # Copy sampled normal texts to experiments/data/normal_text (optional, but keeps experiment self-contained)
    copied_normal = []
    for p in normal_txt_paths:
        rel = os.path.relpath(p, dataset_cleaned_root)
        dst = os.path.join(normal_text_root, rel)
        ensure_dir(os.path.dirname(dst))
        with open(p, "r", encoding="utf-8", errors="ignore") as fsrc:
            content = fsrc.read()
        with open(dst, "w", encoding="utf-8") as fdst:
            fdst.write(content)
        copied_normal.append(dst)

    # 3) Build labels.csv
    print("[3/3] Building labels.csv ...")
    df = build_experiment_labels(
        normal_txt_paths=copied_normal,
        anomaly_txt_paths=anomaly_txt_paths,
        output_csv=labels_csv
    )
    print(f"[OK] labels.csv saved: {labels_csv}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
