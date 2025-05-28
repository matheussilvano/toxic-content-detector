import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

X_test, y_test = joblib.load("data/test_data.pkl")

X_test_vec = vectorizer.transform(X_test)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
erro = 1 - accuracy

print(f"Acurácia: {accuracy * 100:.2f}%")
print(f"Erros: {erro * 100:.2f}%\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
