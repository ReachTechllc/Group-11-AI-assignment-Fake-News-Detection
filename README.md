# 🤖 Fake News Detection Chatbot - Complete Guide

A machine learning chatbot that detects fake news using Logistic Regression and TF-IDF vectorization. Built with Python Flask backend and HTML/CSS/JavaScript frontend. Run locally on XAMPP with no cloud deployment needed.

## 📌 What This Project Does

This chatbot analyzes any news article title or text and tells you if it's likely **FAKE** or **REAL** based on an ML model trained on 45,000+ real and fake news articles from Kaggle.

**Example:**
- Input: "Scientists discover the moon is made of cheese"
- Output: ❌ **FAKE** (98% confidence)

**How it works:**
1. You paste news text in the web interface
2. Text gets sent to Python backend
3. Model analyzes it
4. Returns: FAKE or REAL with confidence percentage

## 🎯 Project Structure

```
fake-news-chatbot/
├── chatbot.html              (Web interface - what you see)
├── app.py                    (Python backend - makes predictions)
├── requirements.txt          (Python packages needed)
├── dataset.py                (Auto-downloads training data)
├── Fake.csv                  (Dataset - downloaded by dataset.py)
├── True.csv                  (Dataset - downloaded by dataset.py)
├── model.pkl                 (Trained ML model - created by app.py)
├── vectorizer.pkl            (TF-IDF vectorizer - created by app.py)
├── KAGGLE_SETUP.md           (How to setup Kaggle credentials)
├── LOCALHOST_QUICKSTART_WITH_DOWNLOAD.txt  (Quick setup guide)
└── README.md                 (This file)
```

## 💻 System Architecture

Your computer runs two services at the same time:

```
┌─────────────────────────────────────────────────┐
│           YOUR WEB BROWSER                      │
│  http://localhost/fake-news-chatbot/chatbot.html│
│                                                 │
│   Beautiful chat interface where you type       │
└────────────────────┬────────────────────────────┘
                     │
                     │ POST request with text
                     ↓
┌─────────────────────────────────────────────────┐
│  XAMPP APACHE (Port 80)                         │
│  Serves: chatbot.html file                      │
└─────────────────────────────────────────────────┘

                     ↕ (different process)

┌─────────────────────────────────────────────────┐
│  PYTHON FLASK (Port 5000)                       │
│  Runs: app.py (your ML prediction engine)       │
│                                                 │
│  When prediction needed:                        │
│  1. Cleans text                                 │
│  2. Applies TF-IDF vectorizer                   │
│  3. Runs through trained model                  │
│  4. Returns: FAKE/REAL + confidence             │
└────────────────────┬────────────────────────────┘
                     │
                     │ JSON response
                     ↓
┌─────────────────────────────────────────────────┐
│  Browser displays result                        │
└─────────────────────────────────────────────────┘
```

## 🚀 Complete Setup Guide

### Prerequisites

Before you start, make sure you have:

1. **XAMPP** - Download from https://www.apachefriends.org/
   - Provides Apache web server
   - Simple to install and use
   - Free and open source

2. **Python 3+** - Download from https://www.python.org/
   - When installing: CHECK "Add Python to PATH"
   - Download from: https://www.python.org/downloads/
   - Verify: Open command prompt, type `python --version`

3. **Kaggle Account** - Create free at https://www.kaggle.com
   - Needed to download training data
   - Just need email and password

4. **Internet Connection** - For downloading dataset (~180 MB)

### Step 1: Get Kaggle API Credentials

This lets you automatically download the dataset.

1. Go to https://www.kaggle.com/settings/account
   - Must be logged into Kaggle

2. Scroll down to "API" section

3. Click "Create New API Token"
   - A file called `kaggle.json` will download

4. Move the file to the right location:

   **Windows:**
   ```
   C:\Users\YOUR_USERNAME\.kaggle\kaggle.json
   ```
   - If `.kaggle` folder doesn't exist, create it

   **Mac/Linux:**
   ```
   ~/.kaggle/kaggle.json
   ```
   - Open Terminal and run:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Why this works:** Kaggle API needs this file to authenticate. It's like your password for accessing Kaggle's servers programmatically.

### Step 2: Create Project Folder

Create a folder where all your files will live.

**Windows:**
```
C:\xampp\htdocs\fake-news-chatbot\
```

**Mac/Linux:**
```
/Applications/XAMPP/htdocs/fake-news-chatbot/
```

The `htdocs` folder is important - Apache automatically serves files from here.

### Step 3: Place Project Files

Put all these files in your project folder:
- chatbot.html
- app.py
- requirements.txt
- dataset.py

The CSV files and model files will be created by the scripts.

### Step 4: Install Python Packages

Open Command Prompt (Windows) or Terminal (Mac/Linux) and navigate to your project folder:

**Windows:**
```bash
cd C:\xampp\htdocs\fake-news-chatbot
```

**Mac/Linux:**
```bash
cd /Applications/XAMPP/htdocs/fake-news-chatbot
```

Install required packages:
```bash
pip install -r requirements.txt
```

This installs:
- Flask (web framework for Python)
- scikit-learn (machine learning library)
- pandas (data processing)
- numpy (numerical computing)

### Step 5: Download Dataset

Run the auto-download script:

**Windows:**
```bash
python dataset.py
```

**Mac/Linux:**
```bash
python3 dataset.py
```

What happens:
1. Installs `kaggle` package
2. Reads your kaggle.json credentials
3. Downloads Fake.csv and True.csv from Kaggle
4. Extracts and verifies the files
5. Shows you the file sizes

You should see:
```
✅ Fake.csv found (88.23 MB)
✅ True.csv found (98.41 MB)
Total: 186.64 MB
✅ SUCCESS! Dataset is ready to use!
```

**Troubleshooting dataset.py:**
- If it says "kaggle.json not found": Re-do Step 1, make sure file is in the right place
- If download is slow: Kaggle servers might be busy, try again later
- If you already have CSV files: It will skip download and verify them

### Step 6: Train the Model

Still in the same terminal, run:

**Windows:**
```bash
python app.py
```

**Mac/Linux:**
```bash
python3 app.py
```

This will:
1. Load Fake.csv and True.csv
2. Clean all the text (remove URLs, special characters, etc.)
3. Convert text to numbers using TF-IDF (term frequency-inverse document frequency)
4. Train Logistic Regression model on the data
5. Save model.pkl and vectorizer.pkl

**Expected output:**
```
Training model... (this may take a minute)
✅ Model trained successfully! Accuracy: 0.9634
📁 Saved: model.pkl, vectorizer.pkl
Running on http://127.0.0.1:5000
```

**Important:** Do NOT close this terminal window! Your Flask server needs to keep running. This terminal is now serving predictions on port 5000.

**Training time:** Usually 2-5 minutes depending on your computer. Be patient!

### Step 7: Start XAMPP

In a **NEW** terminal window (keep the Python one open):

1. Open XAMPP Control Panel
   - Windows: Start Menu → XAMPP Control Panel
   - Mac: Applications → XAMPP

2. You should see a window with "Apache", "MySQL", etc.

3. Find "Apache" and click "Start" button
   - It should turn green and say "Running"
   - Port should show as 80

**What this does:** Apache web server now serves your HTML file on port 80, which is the default web port.

### Step 8: Open the Chatbot

1. Open your web browser (Chrome, Firefox, Safari, Edge, etc.)

2. Go to: `http://localhost/fake-news-chatbot/chatbot.html`

3. You should see a beautiful purple chatbot interface

4. Try typing something:
   ```
   Scientists discover the moon is made of cheese, NASA confirms.
   ```

5. Click "Send"

6. The chatbot should respond: ❌ **FAKE** (with confidence %)

**Congratulations!** Your chatbot is working! 🎉

## 📚 How to Use the Chatbot

### Examples to Try

**FAKE News (should detect as fake):**
- "Aliens found on Mars, governments hiding the truth"
- "Drinking water can make you invisible"
- "Scientists discover moon made of cheese"
- "Trump becomes astronaut and goes to space"

**REAL News (should detect as real):**
- "Apple releases new iPhone with AI features"
- "Climate change report shows warming trends"
- "Stock market rises after earnings reports"
- "New vaccine approved by health authorities"

### What the Output Means

```
❌ FAKE (92.3%)
```

- ❌ = Prediction is FAKE news
- 92.3% = Confidence of prediction (how sure the model is)

Higher confidence = model is more sure about its prediction

### Note on Accuracy

The model is trained to recognize patterns in text that distinguish real from fake news. It's not 100% accurate - no machine learning model is. It's designed to help you think critically about news, not to be the final word.

## 🔧 Troubleshooting

### "Connection Error" in Chatbot

**Symptom:** Chatbot shows "Connection Error" when you try to send a message.

**Solutions:**
1. Check that Python terminal still shows "Running on http://127.0.0.1:5000"
   - If not, your Flask server crashed
   - Open new terminal and run `python app.py` again

2. Check that XAMPP Apache is running (green in Control Panel)
   - If not, click Start button

3. Press F12 in browser, click "Console" tab
   - Look for specific error message
   - Screenshot and search for the error

### "Address Already in Use" Error

**Symptom:** When starting `python app.py`, it says port 5000 is already in use.

**Solutions:**

**Option 1: Use Different Port**
- Edit app.py, find the last line
- Change `port=5000` to `port=5001`
- Edit chatbot.html, find line with `const BACKEND_URL`
- Change `:5000/predict` to `:5001/predict`
- Restart Flask

**Option 2: Kill Process Using Port 5000**

Windows (Command Prompt as Admin):
```bash
netstat -ano | findstr :5000
taskkill /PID [PID_NUMBER] /F
```

Mac/Linux (Terminal):
```bash
lsof -i :5000
kill -9 [PID_NUMBER]
```

### Python Not Found

**Symptom:** "python: command not found" or "ModuleNotFoundError"

**Solutions:**
1. Make sure Python is installed: https://www.python.org/downloads/
2. When installing Python, CHECK "Add Python to PATH"
3. Restart Command Prompt/Terminal after installing Python
4. Verify installation:
   ```bash
   python --version
   ```

### Dataset Download Fails

**Symptom:** dataset.py fails to download Fake.csv and True.csv

**Solutions:**
1. Make sure kaggle.json is in correct location
   - Windows: `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`
   - Mac/Linux: `~/.kaggle/kaggle.json`

2. Check your internet connection

3. If Kaggle is down, download manually:
   - Go to: https://www.kaggle.com/datasets/jainpooja/fake-news-detection
   - Click "Download" button
   - Unzip in your project folder
   - Run: `python app.py`

### Model Training Takes Too Long

**Symptom:** Training seems stuck or very slow.

**This is normal!** The dataset has 45,000 rows of text. Training takes time:
- Slow internet: 5-10 minutes
- Regular computer: 2-5 minutes
- Fast computer: 1-2 minutes

Just let it run. You should see progress in the terminal.

### Browser Says "Cannot GET /fake-news-chatbot/..."

**Symptom:** 404 error when trying to access chatbot.

**Solutions:**
1. Make sure XAMPP Apache is running
2. Make sure chatbot.html is in correct folder:
   - Windows: `C:\xampp\htdocs\fake-news-chatbot\chatbot.html`
   - Mac: `/Applications/XAMPP/htdocs/fake-news-chatbot/chatbot.html`
3. Try URL: `http://localhost/fake-news-chatbot/chatbot.html`

## 📊 Understanding the Code

### chatbot.html
- Provides the user interface
- Written in HTML (structure), CSS (styling), JavaScript (interactivity)
- When you click "Send", JavaScript sends your text to Flask backend
- Displays the response in chat format

### app.py
- Flask web server running on port 5000
- Loads trained model.pkl and vectorizer.pkl
- When chatbot sends text, Flask receives it
- Cleans the text (removes special characters, URLs, etc.)
- Applies TF-IDF vectorizer to convert text to numbers
- Logistic Regression model makes prediction
- Returns JSON response with prediction and confidence

### dataset.py
- Downloads Fake.csv and True.csv from Kaggle
- Uses Kaggle API with your credentials
- Automatic - you don't need to manually download

### model.pkl & vectorizer.pkl
- Created after training with `python app.py`
- model.pkl = Trained Logistic Regression model
- vectorizer.pkl = TF-IDF vectorizer that converts text to numbers
- These are saved so you don't need to retrain every time

### requirements.txt
- Lists all Python packages needed
- When you run `pip install -r requirements.txt`, it installs all of them
- Format: package_name==version_number

## 🎓 Machine Learning Concepts Explained

### TF-IDF (Term Frequency-Inverse Document Frequency)

Machine learning models need numbers, not text. TF-IDF converts text to numbers:

1. **Term Frequency (TF):** How often a word appears in a document
2. **Inverse Document Frequency (IDF):** How rare a word is across all documents
3. **Result:** Important words get high scores, common words get low scores

Example:
- Word "politician" appears 5 times in article = high TF
- "politician" appears in many articles = low IDF
- TF-IDF score = medium (it's somewhat important)

### Logistic Regression

A machine learning algorithm that predicts yes/no (or in our case, REAL/FAKE).

How it works:
1. Training: Show it thousands of examples with labels (this is FAKE, this is REAL)
2. Learning: Find patterns in the text that distinguish REAL from FAKE
3. Prediction: When you give it new text, it compares to learned patterns
4. Output: Returns a probability (0-100%)

### Training vs Prediction

**Training (happens once when you run `python app.py`):**
- Uses all 45,000 examples
- Model learns patterns
- Takes 2-5 minutes
- Saves results to model.pkl

**Prediction (happens every time you use chatbot):**
- Uses the trained model
- Analyzes new text
- Instant (under 1 second)
- No learning, just classification

## 🔒 Security Notes

### kaggle.json

- Contains your Kaggle API key
- Treat it like a password - keep it private
- Don't share, upload to GitHub, or post online

If accidentally shared:
1. Go to https://www.kaggle.com/settings/account
2. Delete the API token
3. Create a new one
4. Replace your kaggle.json file

### Dataset Attribution

The fake news dataset is from Kaggle user "jainpooja":
- Source: https://www.kaggle.com/datasets/jainpooja/fake-news-detection
- Dataset size: ~45,000 articles
- License: Creative Commons (verify on Kaggle page)

Always respect dataset licenses when using public data.

## 🌍 Next Steps After Getting It Working

### Share Your Project

1. **GitHub:**
   - Create repo at github.com
   - Upload all files (except CSV files - too large)
   - Add .gitignore to exclude large files
   - Share the GitHub link

2. **YouTube Tutorial:**
   - Record yourself setting it up
   - Show how it works
   - People can follow along with this README

3. **School Project:**
   - Demo to your class
   - Show accuracy results
   - Discuss how ML detects fake news

### Improve the Model

1. Add more training data from different sources
2. Try different algorithms: SVM, Random Forest, Neural Networks
3. Add confidence threshold (only show results > 85%)
4. Add source validation (check if domain is legitimate)
5. Add URL analysis

### Deploy to Cloud

When ready to share online without needing XAMPP:

1. **Replit:** Upload your project, it auto-runs
2. **Render.com:** Free tier with auto-deployment
3. **Heroku:** (Note: paid now, but still an option)

See deployment guides in separate documents.

## 📞 If You Get Stuck

### Checklist Before Asking for Help

1. ✓ Python installed? (`python --version`)
2. ✓ XAMPP installed and Apache running?
3. ✓ kaggle.json in correct folder?
4. ✓ All project files in `C:\xampp\htdocs\fake-news-chatbot\`?
5. ✓ Ran `pip install -r requirements.txt`?
6. ✓ Ran `python dataset.py` (wait for success)
7. ✓ Ran `python app.py` (don't close terminal)?
8. ✓ Opened browser to `http://localhost/fake-news-chatbot/chatbot.html`?
9. ✓ XAMPP Apache is running?

### Common Fixes in Order

1. Close everything and restart
2. Make sure Python terminal is still running
3. Check browser F12 Console for errors
4. Try incognito/private browsing mode
5. Restart XAMPP
6. Delete model.pkl and vectorizer.pkl, retrain

### Where to Get Help

1. Check the troubleshooting section above
2. Read KAGGLE_SETUP.md for Kaggle issues
3. Read LOCALHOST_ARCHITECTURE.txt to understand the system
4. Search YouTube for similar issues
5. Ask in programming communities (Stack Overflow, Reddit r/learnprogramming)

## 📜 Files Included in This Project

| File | Purpose |
|------|---------|
| chatbot.html | Web interface (HTML/CSS/JavaScript) |
| app.py | Flask backend (Python prediction engine) |
| dataset.py | Kaggle dataset auto-downloader |
| requirements.txt | Python package dependencies |
| README.md | This file - complete documentation |
| KAGGLE_SETUP.md | Detailed Kaggle API setup guide |
| LOCALHOST_QUICKSTART_WITH_DOWNLOAD.txt | Quick 5-step setup |

## ✅ Quick Verification Checklist

After setup, verify everything works:

- [ ] Kaggle API credentials set up
- [ ] Project folder created
- [ ] All files copied to project folder
- [ ] `pip install -r requirements.txt` succeeded
- [ ] `python dataset.py` showed success
- [ ] `python app.py` is running (in terminal)
- [ ] XAMPP Apache is running
- [ ] Browser shows chatbot at http://localhost/fake-news-chatbot/chatbot.html
- [ ] Can type and click Send button
- [ ] Gets REAL or FAKE response
- [ ] Shows confidence percentage

If all checked: **You're done!** 🎉

## 📝 Version Info

- Python: 3.7+
- Flask: 2.3.0
- scikit-learn: 1.2.0
- pandas: 1.5.0
- XAMPP: Latest version

## 🙏 Acknowledgments

- **Dataset:** Kaggle user "jainpooja" for fake news dataset
- **Libraries:** scikit-learn, Flask, pandas
- **XAMPP:** Apache Friends for easy local server setup

## 📄 License

This project is open source. Feel free to use, modify, and share for educational purposes.

---

**Last Updated:** 2026
**Created for:** Educational purposes, school projects, ML learning

**Questions? Refer to the troubleshooting section or re-read the step-by-step guide!**

Happy detecting! 🚀
