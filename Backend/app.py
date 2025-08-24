# app.py
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- OPTIONAL: remove these two lines if you don't want LLM analysis ---
import google.generativeai as genai
# ----------------------------------------------------------------------

# ========= Load price model =========
MODEL_PATH = 'price_model.joblib'
model = joblib.load(MODEL_PATH)

# ========= Gemini setup (optional) =========
# NOTE: apna real key .env ya env var me rakho; code me hardcode na karo
GEMINI_API_KEY = "AIzaSyAdPg9vEshFj2Wt8-R51DWV0_bMU25WemY"  # User's provided Gemini API key
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured with provided key.")
except Exception as e:
    print(f"Gemini API configuration error: {e}")

# ========= Mappings (must match training encodings) =========
CATEGORY_MAP = {'electronics': 0, 'clothing': 1, 'home': 2, 'sports': 3, 'books': 4, 'automotive': 5, 'health': 6, 'toys': 7}
BRAND_MAP    = {'premium': 0, 'established': 1, 'emerging': 2, 'generic': 3}
SHIPPING_MAP = {'standard': 0, 'express': 1, 'prime': 2, 'free': 3}
SELLER_MAP   = {'high': 0, 'medium': 1, 'low': 2, 'new': 3}
COMPETITION_MAP = {'low': 0, 'medium': 1, 'high': 2, 'saturated': 3}
DEMAND_MAP      = {'regular': 0, 'seasonal': 1, 'trending': 2, 'declining': 3}
PRODUCT_AGE_MAP = {'new': 0, 'recent': 1, 'established': 2, 'mature': 3}
STOCK_MAP       = {'high': 0, 'medium': 1, 'low': 2, 'limited': 3}

# ========= Feature extraction =========
def extract_features(data):
    # Expecting keys from your form: category, brand, rating, reviews, shipping, seller,
    # competition, demand, productAge, stock
    category     = CATEGORY_MAP.get(str(data.get('category', '')).lower(), 0)
    brand        = BRAND_MAP.get(str(data.get('brand', '')).lower(), 0)
    rating       = float(data.get('rating', 0) or 0)
    reviews      = float(data.get('reviews', 0) or 0)
    shipping     = SHIPPING_MAP.get(str(data.get('shipping', '')).lower(), 0)
    seller       = SELLER_MAP.get(str(data.get('seller', '')).lower(), 0)
    competition  = COMPETITION_MAP.get(str(data.get('competition', '')).lower(), 0)
    demand       = DEMAND_MAP.get(str(data.get('demand', '')).lower(), 0)
    product_age  = PRODUCT_AGE_MAP.get(str(data.get('productAge', '')).lower(), 0)
    stock        = STOCK_MAP.get(str(data.get('stock', '')).lower(), 0)

    # ORDER MUST MATCH TRAINING
    return [category, brand, rating, reviews, shipping, seller, competition, demand, product_age, stock]

# ========= LLM explanation (optional) =========
def generate_anomaly_explanation(features, prediction, data):
    """
    Returns a short explanation via Gemini; if quota/error, returns a safe fallback.
    """
    prompt = (
        "You are an e-commerce pricing analyst.\n"
        f"Product (raw form data): {dict(data)}\n"
        f"Model numeric features: {features}\n"
        f"Predicted price: {round(float(prediction), 2)}\n"
        "Explain potential anomalies or risks in 2–3 concise lines with 1 action item. "
        "If nothing stands out, reply: 'No anomaly detected.'"
    )
    try:
        # Always try to use the provided key
        model_llm = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model_llm.generate_content(prompt)
        # Debug print for LLM response
        print("LLM raw response:", response)
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif getattr(response, "candidates", None):
            return response.candidates[0].content.parts[0].text.strip()
        return "No anomaly detected."
    except Exception as e:
        # If quota exceeded or any error, fallback gracefully
        if '429' in str(e) or 'quota' in str(e).lower():
            return "AI quota exceeded or unavailable. No anomaly detected."
        return f"LLM unavailable: {e}. No anomaly detected."

# ========= Dynamic KPI helpers (feature-based) =========
def calculate_confidence(pred, features, data):
    rating  = float(data.get("rating", 0) or 0)
    reviews = float(data.get("reviews", 0) or 0)
    comp    = str(data.get("competition", "medium")).lower()
    stock   = str(data.get("stock", "medium")).lower()

    score = 0
    # rating & reviews
    if rating >= 4.5: score += 2
    elif rating >= 4: score += 1
    if reviews > 500: score += 2
    elif reviews > 100: score += 1
    # market friction
    if comp in ("high", "saturated"): score -= 1
    if stock == "limited": score -= 1

    if score >= 3:  return "Very High"
    if score == 2:  return "High"
    if score == 1:  return "Medium"
    return "Low"

def market_position(pred, features, data):
    demand = str(data.get("demand", "regular")).lower()
    brand  = str(data.get("brand", "generic")).lower()

    if "premium" in brand or demand == "trending":
        return "Premium"
    if demand == "declining":
        # further split by predicted price
        return "Budget" if pred < 100 else "Mid-range"
    if pred < 100:
        return "Budget"
    if pred < 250:
        return "Mid-range"
    return "Premium"

def price_range(pred, features, data):
    demand = str(data.get("demand", "regular")).lower()
    # dynamic margins by demand state
    margin = 0.10
    if demand == "trending":
        margin = 0.20
    elif demand == "declining":
        margin = 0.05
    lower = round(float(pred) * (1 - margin), 2)
    upper = round(float(pred) * (1 + margin), 2)
    return f"${lower} - ${upper}"

def competitive_index(features, data):
    competition_level = str(data.get("competition", "medium")).lower()
    mapping = {
        "low": "Very Favorable",
        "medium": "Favorable",
        "high": "Competitive",
        "saturated": "Highly Saturated"
    }
    return mapping.get(competition_level, "Unknown")

# ========= Flask app =========
# frontend serve karne ke liye static_folder change kiya
app = Flask(__name__, static_folder='../Frontend', static_url_path='')
CORS(app)

# Serve index.html from Frontend
@app.route('/')
def serve_index():
    return send_from_directory('../Frontend', 'index.html')

# Serve other static files (CSS/JS/images)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../Frontend', path)

# Predict endpoint
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({'message': 'Prediction endpoint is live.'})

    # accept JSON or form-encoded
    data = request.json if request.is_json else request.form

    try:
        features = extract_features(data)
        pred = model.predict([features])[0]
        ai_analysis = generate_anomaly_explanation(features, pred, data)
    except Exception as e:
        # hard fail safety
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    return jsonify({
        'predicted_price' : round(float(pred), 2),
        'anomaly_explanation': ai_analysis,
        'ai_analysis'      : ai_analysis,
        'recommendations'  : ai_analysis,
        'confidence'       : calculate_confidence(pred, features, data),
        'market_position'  : market_position(pred, features, data),
        'price_range'      : price_range(pred, features, data),
        'competitive_index': competitive_index(features, data)
    })

if __name__ == '__main__':
    # set host='0.0.0.0' if you want to test from phone/another PC
    app.run(debug=True)
