import streamlit as st
import joblib
import numpy as np

# Load model and encoder
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾")

st.title("🌾 Crop Recommendation System")
st.write("Enter the details of your farm conditions to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", format="%.2f")
humidity = st.number_input("Humidity (%)", format="%.2f")
ph = st.number_input("Soil pH", format="%.2f")
rainfall = st.number_input("Rainfall (mm)", format="%.2f")

# Predict button
if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = le.inverse_transform(prediction)[0]
    st.success(f"🌱 Recommended Crop: **{crop_name.upper()}**")
