# 🧠 Detector de Conteúdo Tóxico

Este é um projeto de Machine Learning para **detecção de comentários tóxicos** usando NLP com Python e uma interface interativa com Streamlit. Ele utiliza **Regressão Logística** e **TF-IDF** para classificar comentários como **tóxicos ou não tóxicos**.

Você pode testar a aplicação online através do seguinte link: [Detector de Conteúdo Tóxico](https://toxic-content-detector.streamlit.app/)

![image](https://github.com/user-attachments/assets/1ad55696-674e-4d16-aa86-0364d5693552)

---

## 🚀 Funcionalidades

- Pré-processamento de texto com NLTK.
- Vetorização com TF-IDF.
- Modelo de Regressão Logística para classificação binária.
- Interface gráfica com Streamlit.
- Treinamento e salvamento de modelo e vetorizador com `joblib`.

---

## 📁 Estrutura do Projeto

```
toxic-content-detector/
├── app/
│   └── streamlit_app.py        # Interface com Streamlit
├── data/
│   └── train.csv               # Base de dados de treino
├── models/
│   ├── logistic_model.pkl      # Modelo treinado
│   └── vectorizer.pkl          # Vetorizador TF-IDF treinado
├── src/
│   ├── preprocessing.py        # Função de limpeza de texto
│   ├── predict.py              # Função de predição
|   └──train_model.py           # Script de treinamento
├── requirements.txt
├── Dockerfile
└── README.md
```
---

## 🧼 Pré-processamento de Texto

O texto é limpo com os seguintes passos:

- Remoção de URLs
- Remoção de caracteres especiais e números
- Conversão para minúsculas
- Remoção de stopwords (palavras comuns como "the", "and", etc.)
- Stemização (reduz palavras à sua raiz)

---

## 📦 Docker

### Build da imagem

```bash
docker build -t toxic-content-detector .
```

### Rodar o container

```bash
docker run -p 8501:8501 toxic-content-detector
```

---

## 📊 Dataset

O modelo foi treinado com um subconjunto do dataset **[Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**.

A base de dados está na pasta `/data`

---

## 🛠 Tecnologias

- Python
- Pandas
- scikit-learn
- NLTK
- Streamlit
- Docker

---

🧠 O modelo (`models/logistic_model.pkl`) já está treinado e pronto para uso.
Caso deseje treiná-lo novamente, execute:

python src/train_model.py

