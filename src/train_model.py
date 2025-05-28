import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

X_test, y_test = joblib.load("data/test_data.pkl")
X_test_vec = vectorizer.transform(X_test)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%\n")

print("Relatório de Classificação:")
report_dict = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Tóxico', 'Tóxico'], yticklabels=['Não Tóxico', 'Tóxico'])
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

metrics_names = ['precision', 'recall', 'f1-score']
classes = ['0 (Não Tóxico)', '1 (Tóxico)']

values = {metric: [report_dict['0'][metric], report_dict['1'][metric]] for metric in metrics_names}

import numpy as np

x = np.arange(len(classes))
width = 0.2

plt.figure(figsize=(8,5))
for i, metric in enumerate(metrics_names):
    plt.bar(x + i*width, values[metric], width=width, label=metric.capitalize())

plt.xticks(x + width, classes)
plt.ylim(0,1.1)
plt.ylabel('Score')
plt.title('Precision, Recall e F1-Score por Classe')
plt.legend()
plt.show()
