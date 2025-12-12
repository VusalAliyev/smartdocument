import os
from pdf2image import convert_from_path
import pytesseract
import cv2

# Poppler ve Tesseract yolu:
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Dataset root:
DATASET_DIR = "dataset"

for root, dirs, files in os.walk(DATASET_DIR):
    for filename in files:
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, filename)
            print(f"Processing: {pdf_path}")

            try:
                # PDF → Image → OCR
                pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
                full_text = ""
                for i, page in enumerate(pages):
                    temp_img = f"temp_{i}.jpg"
                    page.save(temp_img, "JPEG")
                    img = cv2.imread(temp_img)
                    text = pytesseract.image_to_string(img, lang="eng")
                    full_text += f"\n=== Page {i+1} ===\n{text}\n"
                    os.remove(temp_img)

                # Save as .txt next to PDF
                txt_filename = filename.replace(".pdf", ".txt")
                txt_path = os.path.join(root, txt_filename)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"✅ Saved: {txt_path}")
            except Exception as e:
                print(f"❌ Error processing {pdf_path}: {e}")

print("✅✅ All PDFs converted to .txt")
