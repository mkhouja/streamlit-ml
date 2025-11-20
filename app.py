import streamlit as st
import joblib
import numpy as np

# Charger le modèle et les colonnes
model = joblib.load('heart_model.joblib')
columns = joblib.load('heart_columns.joblib')

st.title("Prédiction du risque de maladie cardiaque ❤️")

inputs = []
for col in columns:
    value = st.number_input(col, value=0.0)
    inputs.append(value)

if st.button("Prédire"):
    data = np.array([inputs])
    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("Risque élevé de maladie cardiaque détecté.")
    else:
        st.success("Risque faible de maladie cardiaque détecté.")
