import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve


model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

X_test, y_test = joblib.load("data/test_data.pkl")

probas = model.predict_proba(X_test)[:, 1]
y_pred = (probas >= 0.71).astype(int)


precisions, recalls, thresholds = precision_recall_curve(y_test, probas)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precisão')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precisão vs Recall por Threshold')
plt.legend()
plt.grid()
plt.show()

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

tn, fp, fn, tp = cm.ravel()

labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
values = [tn, fp, fn, tp]
colors = ['green', 'red', 'red', 'green']

plt.figure(figsize=(8,5))
bars = plt.bar(labels, values, color=colors)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200, 
             f'{value:,}', ha='center', va='bottom', fontsize=10)

plt.title('Acertos e Erros da Classificação')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
