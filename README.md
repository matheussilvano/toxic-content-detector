# 🧠 Toxic Content Detector

A machine learning project for **toxic comment detection** using NLP with Python and an interactive Streamlit interface. It uses **Logistic Regression** and **TF-IDF** to classify comments as **toxic or non-toxic**.

Try the app online: [Toxic Content Detector](https://toxic-content-detector.streamlit.app/)

![image](https://github.com/user-attachments/assets/751e90e4-1139-445f-8ae5-01e2a4a26bc4)

---

## 🚀 Features

- Text preprocessing with NLTK.
- TF-IDF vectorization.
- Logistic Regression model for binary classification.
- Streamlit web interface.
- Model and vectorizer training and persistence with `joblib`.

---

## 📁 Project structure

```
toxic-content-detector/
├── app/
│   └── streamlit_app.py        # Streamlit interface
├── data/
│   └── train.csv               # Training dataset
├── models/
│   ├── logistic_model.pkl      # Trained model
│   └── vectorizer.pkl          # Trained TF-IDF vectorizer
├── src/
│   ├── preprocessing.py        # Text cleaning function
│   ├── predict.py              # Prediction function
│   └── train_model.py          # Training script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧼 Text preprocessing

Text is cleaned with the following steps:

- URL removal
- Special character and number removal
- Lowercasing
- Stopword removal (common words such as "the", "and", etc.)
- Stemming (reduces words to their root)

---

## 📦 Docker

### Build the image

```bash
docker build -t toxic-content-detector .
```

### Run the container

```bash
docker run -p 8501:8501 toxic-content-detector
```

---

## 📊 Dataset

The model was trained on a subset of the **[Jigsaw Toxic Comment Classification Challenge (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)** dataset.

The data lives in the `/data` folder.

---

## 🛠 Tech stack

- Python
- Pandas
- scikit-learn
- NLTK
- Streamlit
- Docker

---

🧠 The model (`models/logistic_model.pkl`) is already trained and ready to use.
To retrain it, run:

```bash
python src/train_model.py
```
