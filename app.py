"""
Flask Backend for Fake News Detection Chatbot
Deploy on: Render.com, Replit, or PythonAnywhere
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============= TRAINING & MODEL SAVING =============
# Run this section once to train and save the model

def train_and_save_model():
    """Train the fake news detection model and save it"""
    print("Training model... (this may take a minute)")
    
    # Check if data files exist
    if not os.path.exists("datasets/Fake.csv") or not os.path.exists("datasets/True.csv"):
        print("Error: Fake.csv and True.csv not found!")
        print("Download from: https://www.kaggle.com/datasets/jainpooja/fake-news-detection")
        return False
    
    try:
        # Load data
        fake = pd.read_csv("datasets/Fake.csv")
        real = pd.read_csv("datasets/True.csv")
        
        fake["label"] = 1  # fake
        real["label"] = 0  # real
        
        df = pd.concat([fake, real], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Text cleaning function
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
            text = re.sub(r"[^a-z\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42
        )
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train Logistic Regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Save model and vectorizer
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        
        # Calculate accuracy
        accuracy = model.score(X_test_vec, y_test)
        print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
        print("📁 Saved: model.pkl, vectorizer.pkl")
        
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        return False

# Load trained model and vectorizer
def load_model():
    """Load pre-trained model and vectorizer"""
    global model, vectorizer
    
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        return True
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return False

# Initialize model variables
model = None
vectorizer = None

# ============= TEXT CLEANING =============
def clean_text(text):
    """Clean text for prediction"""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============= API ROUTES =============

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "Fake News Detection API",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict if text is fake or real news
    
    Request body: {"text": "article text here"}
    Response: {"prediction": "FAKE|REAL", "confidence": 0-100}
    """
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get text from request
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Clean and vectorize
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vec)[0]
        confidence = max(model.predict_proba(vec)[0]) * 100
        
        # Format response
        result = "FAKE" if prediction == 1 else "REAL"
        
        return jsonify({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": result,
            "confidence": round(confidence, 2)
        })
        
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    Endpoint to train the model (POST request)
    This is useful for Render.com's free tier which doesn't allow file uploads
    """
    try:
        if train_and_save_model():
            if load_model():
                return jsonify({"status": "success", "message": "Model trained and loaded"})
        return jsonify({"status": "error", "message": "Training failed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ============= STARTUP =============

if __name__ == "__main__":
    # Try to load existing model
    if not load_model():
        print("\n⚠️  Model not found!")
        print("To use this app:")
        print("1. Download data from: https://www.kaggle.com/datasets/jainpooja/fake-news-detection")
        print("2. Place Fake.csv and True.csv in the same directory")
        print("3. Run: python app.py")
        print("\nOr in the app directory, run:")
        print("   curl -X POST http://localhost:5000/train")
    else:
        print("✅ Model loaded successfully!")
    
    # Run Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
