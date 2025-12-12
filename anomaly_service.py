import pickle
import re

with open("model/anomaly_model.pkl", "rb") as f:
    vectorizer, anomaly_model = pickle.load(f)

def text_quality_check(text):
    words = text.split()
    word_count = len(words)

    alnum_chars = sum(c.isalnum() for c in text)
    total_chars = len(text) if len(text) > 0 else 1
    alnum_ratio = alnum_chars / total_chars

    if word_count < 50 or alnum_ratio < 0.3:
        return True, "Low OCR/Text quality"

    return False, None

def detect_anomaly(text, classification_confidence):
    # 1️⃣ OCR / text quality anomaly
    quality_anomaly, reason = text_quality_check(text)

    # 2️⃣ Distribution-based anomaly
    X = vectorizer.transform([text])
    isolation_pred = anomaly_model.predict(X)[0]  # -1 anomaly, 1 normal
    isolation_score = anomaly_model.decision_function(X)[0]

    distribution_anomaly = isolation_pred == -1

    # 3️⃣ Combined decision
    is_anomaly = (
        quality_anomaly or
        distribution_anomaly or
        (classification_confidence < 0.7 and isolation_score < 0)
    )

    reasons = []
    if quality_anomaly:
        reasons.append(reason)
    if distribution_anomaly:
        reasons.append("Out-of-distribution document")
    if classification_confidence < 0.5:
        reasons.append("Low classification confidence")

    return {
        "is_anomaly": is_anomaly,
        "isolation_score": round(isolation_score, 4),
        "reasons": reasons
    }
