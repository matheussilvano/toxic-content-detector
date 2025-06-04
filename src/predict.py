import joblib
from src.preprocessing import clean_text

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_text(text, threshold=0.71):
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    proba = model.predict_proba(vect)[0][1]
    prediction = int(proba >= threshold)
    return prediction, proba
