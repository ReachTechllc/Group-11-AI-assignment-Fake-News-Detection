"""
Flask Backend for Fake News Detection Chatbot
Deploy on: Render.com, Replit, or PythonAnywhere

Fixes vs. original:
- Paths are resolved relative to this file (not the current working directory)
- Model is loaded (and auto-trained if missing) at import time, so it also
  works under gunicorn, not just `python app.py`
- Flask reloader/debug no longer trains the model twice
- Safer request handling in /predict
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# ============= PATHS =============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_v1.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer_v1.pkl")

# Set AUTO_TRAIN=false in the environment to disable training on startup
AUTO_TRAIN = os.environ.get("AUTO_TRAIN", "true").lower() == "true"

model = None
vectorizer = None


# ============= TEXT CLEANING =============
def clean_text(text):
    """Clean text (used for both training and prediction)"""
    text = str(text).lower()
    # Remove source "fingerprints" the model could cheat with. In this dataset
    # real articles start with "CITY (Reuters) -" and fake ones contain
    # tags like "21st Century Wire" / "Featured image via Getty".
    text = re.sub(r"^.{0,80}?\(reuters\)\s*-?\s*", " ", text)
    text = re.sub(r"\b(reuters|21st century wire|featured image|getty images|via)\b", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============= TRAINING & MODEL SAVING =============
def train_and_save_model():
    """Train the fake news detection model and save it"""
    print("Training model... (this may take a minute)")

    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        print("Error: Fake.csv and/or True.csv not found!")
        print(f"Looked for:\n  {FAKE_CSV}\n  {TRUE_CSV}")
        print("Download from: https://www.kaggle.com/datasets/jainpooja/fake-news-detection")
        return False

    try:
        fake = pd.read_csv(FAKE_CSV)
        real = pd.read_csv(TRUE_CSV)

        fake["label"] = 1  # fake
        real["label"] = 0  # real

        df = pd.concat([fake, real], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Remove duplicate articles so the test set can't contain copies of training data
        before = len(df)
        df = df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
        print(f"Dropped {before - len(df)} duplicate articles")

        df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
        df = df[df["text"].str.len() > 20].reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )

        vec = TfidfVectorizer(
            max_features=50000, ngram_range=(1, 2), min_df=3,
            sublinear_tf=True, stop_words="english"
        )
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
        clf.fit(X_train_vec, y_train)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vec, f)

        accuracy = clf.score(X_test_vec, y_test)
        print(f"Model trained successfully! Accuracy: {accuracy:.4f}")
        print(f"Saved: {MODEL_PATH}, {VECTORIZER_PATH}")
        return True

    except Exception as e:
        print(f"Error during training: {e}")
        return False


def load_model():
    """Load pre-trained model and vectorizer into globals"""
    global model, vectorizer

    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        print("Model files not found.")
        return False

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def init_model():
    """Load the model; if missing, train it (when enabled) and load again"""
    if load_model():
        return True
    if AUTO_TRAIN:
        print("Model not found. Training now...")
        if train_and_save_model():
            return load_model()
    else:
        print("AUTO_TRAIN is off. POST to /train to train the model.")
    return False


# ============= API ROUTES =============
@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "model_loaded": model is not None and vectorizer is not None,
        "message": "Fake News Detection API",
        "endpoints": {
            "predict": "POST /predict",
            "train": "POST /train",
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
            return jsonify({"error": "Model not loaded. POST to /train first."}), 500

        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        confidence = max(model.predict_proba(vec)[0]) * 100

        result = "FAKE" if prediction == 1 else "REAL"
        if confidence < 70:
            result = "UNCERTAIN"

        return jsonify({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": result,
            "confidence": round(float(confidence), 2),
            "warning": "Very short text: results are unreliable. Paste a full article for better accuracy."
                       if len(cleaned.split()) < 30 else None
        })

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train_endpoint():
    """Train the model on demand (useful on hosts with ephemeral storage)"""
    try:
        if train_and_save_model() and load_model():
            return jsonify({"status": "success", "message": "Model trained and loaded"})
        return jsonify({
            "status": "error",
            "message": "Training failed. Check server logs and that datasets/Fake.csv and datasets/True.csv exist."
        }), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============= STARTUP =============
# Runs at import time so it also works under gunicorn.
# The WERKZEUG_RUN_MAIN check avoids training twice when the debug reloader is on:
# the reloader's parent process skips it, the child process does the work.
_is_reloader_parent = (
    os.environ.get("FLASK_DEBUG") == "1" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
)
if not _is_reloader_parent:
    init_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False prevents the script (and training) from running twice
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)