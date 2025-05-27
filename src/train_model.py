import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from src.preprocessing import clean_text

df = pd.read_csv("data/train.csv")
df = df[['comment_text', 'toxic']].dropna()
df['clean_text'] = df['comment_text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['toxic']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
