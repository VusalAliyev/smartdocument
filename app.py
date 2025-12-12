from flask import Flask, request, render_template
from dotenv import load_dotenv
import os
import pytesseract
import cv2
from pdf2image import convert_from_path
import pandas as pd
import pickle
from seafile_utils import upload_file_to_seafile

# .env dosyasını yükle
load_dotenv()

# Poppler ve Tesseract yolları
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ML modeli yükle
with open("model/trained_model.pkl", "rb") as f:
    vectorizer, classifier = pickle.load(f)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def rule_based_class(text):
    lower = text.lower()
    if "invoice" in lower:
        return "invoices"
    if "financial statement" in lower or "annual report" in lower:
        return "reports"
    if "agreement" in lower or "contract" in lower:
        return "contracts"
    if "dear" in lower and ("regards" in lower or "sincerely" in lower):
        return "notifications"
    return None

def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    full_text = ""
    for i, page in enumerate(pages):
        img_name = f"temp_{i}.jpg"
        page.save(img_name, "JPEG")
        img = cv2.imread(img_name)
        text = pytesseract.image_to_string(img, lang="eng")
        full_text += f"\n=== Page {i+1} ===\n{text}\n"
        os.remove(img_name)
    return full_text

def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    text = ""
    for col in df.columns:
        text += " ".join(df[col].astype(str).tolist()) + " "
    return text

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "Dosya seçilmedi."
    file = request.files["file"]
    if file.filename == "":
        return "Dosya seçilmedi."

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    ext = file.filename.lower().split(".")[-1]
    extracted_text = ""

    if ext == "pdf":
        extracted_text = extract_text_from_pdf(filepath)
    elif ext in ["xls", "xlsx"]:
        extracted_text = extract_text_from_excel(filepath)
    else:
        return "Sadece PDF veya Excel yükleyiniz."

    # Rule-based + ML
    label = rule_based_class(extracted_text)
    if not label:
        X_vec = vectorizer.transform([extracted_text])
        label = classifier.predict(X_vec)[0]

    # 🔑 Seafile'a yükle!
    upload_result = upload_file_to_seafile(filepath, folder_name=label)

    return f"""
    ✅ Dosya: {file.filename}<br>
    ➡️ Belge tipi: <strong>{label}</strong><br>
    ☁️ Seafile yükleme sonucu: {upload_result}
    """

if __name__ == "__main__":
    app.run(debug=True)
