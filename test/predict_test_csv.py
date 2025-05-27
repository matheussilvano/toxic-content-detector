import pandas as pd
import joblib
from src.preprocessing import clean_text

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

df_test = pd.read_csv("data/test.csv")
df_test = df_test[['comment_text']].dropna()
df_test['clean_text'] = df_test['comment_text'].apply(clean_text)

X_test_final = vectorizer.transform(df_test['clean_text'])

preds = model.predict(X_test_final)
probas = model.predict_proba(X_test_final)[:, 1]

df_test['toxic_pred'] = preds
df_test['toxic_proba'] = probas

df_test.to_csv("data/test_predictions.csv", index=False)

print("Previsões salvas em data/test_predictions.csv")
