import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from src.preprocessing import clean_text

df = pd.read_csv("data/train.csv")
df = df[['comment_text', 'toxic']].dropna()
df['clean_text'] = df['comment_text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump((X_test, y_test), "data/test_data.pkl")
