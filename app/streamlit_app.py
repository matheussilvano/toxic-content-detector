import streamlit as st
from src.predict import predict_text

st.title("Detector de Conteúdo Tóxico")
text = st.text_area("Digite um comentário:")

if st.button("Analisar"):
    prediction, probability = predict_text(text)
    resultado = "Tóxico" if prediction == 1 else "Não tóxico"
    st.write(f"**Resultado:** {resultado}")
    st.write(f"**Probabilidade de ser tóxico:** {probability:.2%}")
