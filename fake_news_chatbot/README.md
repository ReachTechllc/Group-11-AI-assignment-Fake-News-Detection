# 🔍 Fake News Detection Chatbot

A chatbot that checks whether a news article looks **REAL** or **FAKE**. It uses TF-IDF and Logistic Regression (scikit-learn) with a Flask backend and a plain HTML/CSS/JavaScript frontend.

**How it works:** you paste text into the chat page → the page sends it to the Flask API → the model returns a verdict (`REAL`, `FAKE` or `UNCERTAIN`) with a confidence percentage.

## 📁 Project Structure

```
fake-news-chatbot/
├── datasets/
│	└── dataset.py  		(run this with py dataset.py to download the datasets from kaggle but it will require kaggle.json)
│   ├── Fake.csv            (training data, downloaded from Kaggle)
│   └── True.csv            (training data, downloaded from Kaggle)
│	          
├── app.py                  (Flask backend: trains, saves and serves the model)
├── index.html            (chat interface)
├── requirements.txt        (Python packages)
├── model_v1.pkl            (trained model, created automatically)
├── vectorizer_v1.pkl       (TF-IDF vectorizer, created automatically)
└── README.md
```

## 🚀 Setup

**You need:** Python 3.8+ and the dataset from Kaggle.

**1. Install packages**

```bash
pip install -r requirements.txt
```

`requirements.txt` should contain: `flask`, `flask-cors`, `scikit-learn`, `pandas`.

**2. Get the dataset**

Download it from [Kaggle: Fake News Detection](https://www.kaggle.com/datasets/jainpooja/fake-news-detection), unzip it, and put `Fake.csv` and `True.csv` inside the `datasets/` folder.

**3. Start the backend**

```bash
python app.py
```

On the first run there are no model files, so the app trains one automatically (1-5 minutes). You will see the accuracy in the terminal. After that, the model loads instantly. Keep this terminal open while you use the chatbot.

**4. Open the chatbot**

Open `index.html` in your browser and paste some text. No web server is needed, because the backend allows cross-origin requests.

## 🔌 API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check (shows if the model is loaded) |
| `/predict` | POST | Body `{"text": "..."}` → `{"prediction", "confidence", "warning"}` |
| `/train` | POST | Retrain the model from the CSV files |

## ⚙️ Configuration

- **Backend address:** `BACKEND_URL` near the top of the script in `index.html`. It defaults to `http://localhost:5000/predict`. Change it to your live URL after deploying.
- **Auto-training:** set the environment variable `AUTO_TRAIN=false` to stop the app from training on startup (useful on hosting services).
- **Retraining:** delete `model_v1.pkl` and `vectorizer_v1.pkl`, then restart the app or call `POST /train`.

Once the model files exist, the CSV files are not needed to run the app.

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| "Model not found" / "Model not loaded" | Check that `datasets/Fake.csv` and `datasets/True.csv` exist, then restart. The terminal prints the exact paths it looked for. |
| Chat shows "Connection Error" | Make sure `python app.py` is running and `BACKEND_URL` is correct. Check the browser console (F12) for details. |
| Training is slow | Normal. The dataset has about 45,000 articles, so allow a few minutes. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again. |

## ⚠️ Limitations

This model detects **writing style**, not truth. It learns patterns from the training articles and cannot fact-check claims.

- Short headlines give unreliable results, so paste a full article for better accuracy.
- Results below 70% confidence are shown as `UNCERTAIN`.
- The training data comes from a limited set of sources, so it may not generalize to every publisher.

## ☁️ Deployment

To host it online (Render, Replit, etc.), train the model on your computer first and upload `app.py`, `model_v1.pkl` and `vectorizer_v1.pkl`. Free tiers wipe files on redeploy, and this way the server never needs the large CSVs. Then set `BACKEND_URL` in `index.html` to your live address.

## 🙏 Credits

- Dataset: Kaggle user [jainpooja](https://www.kaggle.com/datasets/jainpooja/fake-news-detection) (check the dataset page for its license)
- Libraries: Flask, scikit-learn, pandas

Open source for educational use.