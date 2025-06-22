# app.py

import streamlit as st
import pandas as pd
import pickle
import requests


API_URL = "http://localhost:8000/predict"

# Load model and encoders
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

st.title("🚢 Titanic Survival Prediction")

# User input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 25)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Preprocess input
sex_encoded = encoders['Sex'].transform([sex])[0]
embarked_encoded = encoders['Embarked'].transform([embarked])[0]

input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_encoded,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked_encoded
}])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger **survived**.")
    else:
        st.error("💀 The passenger **did not survive**.")
