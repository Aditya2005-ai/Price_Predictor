import streamlit as st
import joblib
import google.generativeai as genai
import os

# Load model
model = joblib.load('price_model.joblib')

# Gemini API setup (replace with your API key)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

def generate_anomaly_explanation(features, prediction):
    prompt = f"""
    Product features: {features}\nPredicted price: {prediction}\nExplain any anomalies or unusual aspects in this prediction as an e-commerce expert.
    """
    response = genai.generate_text(prompt=prompt, model='models/gemini-pro')
    return response.result if hasattr(response, 'result') else str(response)

st.title('E-commerce Price Prediction')

category = st.number_input('Category (numeric code)', min_value=0)
rating = st.number_input('Rating', min_value=0.0, max_value=5.0, step=0.1)
price = st.number_input('Price (for context)', min_value=0.0)

if st.button('Predict'):
    features = [category, rating, price]
    pred = model.predict([features])[0]
    explanation = generate_anomaly_explanation(features, pred)
    st.success(f'Predicted Price: ${pred:.2f}')
    st.info(f'Anomaly Explanation: {explanation}')
