from flask import Flask, request, render_template
from dotenv import load_dotenv
import os
import time
import pytesseract
import cv2
from pdf2image import convert_from_path
import pandas as pd
import pickle

from seafile_utils import upload_file_to_seafile
from anomaly_service import detect_anomaly

# --------------------------------------------------
# Environment setup
# --------------------------------------------------

load_dotenv()

POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------------
# Load trained classification model
# --------------------------------------------------

with open("model/trained_model.pkl", "rb") as f:
    vectorizer, classifier = pickle.load(f)

# --------------------------------------------------
# Flask app configuration
# --------------------------------------------------

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------------------------------------------------
# Rule-based document classification
# --------------------------------------------------

def rule_based_classification(text: str):
    text_lower = text.lower()

    if "invoice" in text_lower:
        return "invoices"
    if "financial statement" in text_lower or "annual report" in text_lower:
        return "reports"
    if "agreement" in text_lower or "contract" in text_lower:
        return "contracts"
    if "dear" in text_lower and ("regards" in text_lower or "sincerely" in text_lower):
        return "notifications"

    return None

# --------------------------------------------------
# OCR & text extraction
# --------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    full_text = ""

    for i, page in enumerate(pages):
        temp_img = f"temp_{i}.jpg"
        page.save(temp_img, "JPEG")

        img = cv2.imread(temp_img)
        text = pytesseract.image_to_string(img, lang="eng")

        full_text += f"\n=== Page {i + 1} ===\n{text}\n"
        os.remove(temp_img)

    return full_text


def extract_text_from_excel(excel_path: str) -> str:
    df = pd.read_excel(excel_path)
    text = ""

    for col in df.columns:
        text += " ".join(df[col].astype(str).tolist()) + " "

    return text

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file selected."

    file = request.files["file"]
    if file.filename == "":
        return "No file selected."

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    extension = file.filename.lower().split(".")[-1]

    start_time = time.time()

    # -----------------------------
    # Text extraction
    # -----------------------------

    if extension == "pdf":
        extracted_text = extract_text_from_pdf(file_path)
    elif extension in ["xls", "xlsx"]:
        extracted_text = extract_text_from_excel(file_path)
    else:
        return "Only PDF and Excel files are supported."

    # -----------------------------
    # Document classification (Rule-based + ML)
    # -----------------------------

    label = rule_based_classification(extracted_text)

    X_vec = vectorizer.transform([extracted_text])
    probabilities = classifier.predict_proba(X_vec)[0]
    confidence = float(max(probabilities))
    ml_label = classifier.classes_[probabilities.argmax()]

    if not label:
        label = ml_label

    # -----------------------------
    # Anomaly detection
    # -----------------------------

    anomaly_result = detect_anomaly(
        text=extracted_text,
        classification_confidence=confidence
    )

    # -----------------------------
    # Decide SeaFile upload folder (YOL A: anomalies + others)
    # -----------------------------

    OTHERS_THRESHOLD = 0.65

    if anomaly_result["is_anomaly"]:
        upload_folder = "anomalies"
    elif confidence < OTHERS_THRESHOLD:
        upload_folder = "others"
    else:
        upload_folder = label

    # -----------------------------
    # Upload to SeaFile
    # -----------------------------

    upload_result = upload_file_to_seafile(
        file_path,
        folder_name=upload_folder
    )

    processing_time = round(time.time() - start_time, 2)

    # -----------------------------
    # Render result page
    # -----------------------------

    return render_template(
        "result.html",
        filename=file.filename,
        label=label,
        confidence=f"{confidence:.2f}",
        anomaly=anomaly_result["is_anomaly"],
        anomaly_score=anomaly_result["isolation_score"],
        anomaly_reasons=anomaly_result["reasons"],
        folder=upload_folder,
        time=processing_time,
        seafile_result=upload_result
    )

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
