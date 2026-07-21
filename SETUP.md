# 🔧 Detailed Setup Guide

This guide walks you through setting up the AQI Predictor project from scratch.

---

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **Git** (check with `git --version`)
- A **GitHub account** (for hosting your code)
- **An internet connection** (to fetch API keys)

---

## Step 1: Prepare Your Machine

### Install Python (if needed)
- **Windows:** Download from [python.org](https://python.org), run installer, **check "Add Python to PATH"**
- **Mac:** `brew install python3.11`
- **Linux:** `sudo apt-get install python3.11 python3.11-venv`

Verify:
```bash
python --version
# Output: Python 3.10+
```

### Install Git (if needed)
- **Windows:** Download from [git-scm.com](https://git-scm.com)
- **Mac:** `brew install git`
- **Linux:** `sudo apt-get install git`

Verify:
```bash
git --version
# Output: git version 2.40+
```

---

## Step 2: Create Your GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. **Repository name:** `aqi-predictor`
3. **Description:** `Production ML system for AQI forecasting with automated CI/CD pipelines`
4. **Visibility:** Public (recruiters need to see it)
5. **Initialize with:** ✅ Add `.gitignore` (Python), ✅ Add a README
6. Click **Create Repository**

---

## Step 3: Clone & Set Up Locally

```bash
# Clone the repo (replace YOUR_USERNAME with your GitHub username)
git clone https://github.com/YOUR_USERNAME/aqi-predictor.git
cd aqi-predictor

# Verify Python version
python --version  # Should be 3.10+
```

---

## Step 4: Set Up Python Virtual Environment

A **virtual environment** isolates your project's dependencies from your system Python. This is essential for professional work.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt. Example:
```
(venv) user@computer aqi-predictor %
```

**Important:** Always activate this before working on the project.

---

## Step 5: Install Dependencies

```bash
# Make sure your venv is activated first!
pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt

# Verify installation (should list many packages)
pip list
```

---

## Step 6: Get API Keys

You need **3 free API keys**. Each takes ~5 minutes to set up.

### 6.1 AQICN API Key (Real-time AQI Data)

1. Visit [aqicn.org/api](https://aqicn.org/api/)
2. **Register** with email
3. Your **API key** appears in your account dashboard
4. Copy it

### 6.2 OpenWeatherMap API Key (Weather Data)

1. Visit [openweathermap.org/api](https://openweathermap.org/api)
2. Click **Sign Up**
3. Create account
4. Go to **API keys** tab
5. Copy the **default key**

### 6.3 Hopsworks API Key (Feature Store & Model Registry)

1. Visit [hopsworks.ai](https://hopsworks.ai)
2. Click **Create Account**
3. Create a **new project** called `aqi-predictor`
4. Go to **Settings** → **API Keys**
5. Generate a new API key and copy it

---

## Step 7: Configure Environment Variables

1. Create a `.env` file in your project root:

```bash
cp .env.example .env
```

2. Edit `.env` and fill in your API keys:

```bash
# On Mac/Linux, use any text editor:
nano .env

# On Windows, use Notepad:
notepad .env
```

3. Paste your API keys:

```env
AQICN_API_KEY=your_actual_aqicn_key_here
OPENWEATHER_API_KEY=your_actual_openweather_key_here
HOPSWORKS_API_KEY=your_actual_hopsworks_key_here
HOPSWORKS_PROJECT_NAME=aqi_predictor
CITY_NAME=Abbottabad
```

4. **Save & close**

**IMPORTANT:** Never commit `.env` to GitHub. It's in `.gitignore` — you're safe. ✅

---

## Step 8: Verify Everything Works

```bash
# Make sure venv is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Test that Python can import the modules
python -c "from src.config import Config; print('✅ Config module works!')"

# Test that dependencies are installed
python -c "import pandas; import sklearn; print('✅ All dependencies installed!')"
```

If you see the ✅ messages, everything is set up correctly.

---

## Step 9: Test the Feature Pipeline (First Real Run!)

This is the first time you'll actually fetch AQI data. Make sure you're in the project root with venv activated:

```bash
# Run the feature pipeline
python -m pipelines.feature_pipeline
```

You should see output like:
```
2024-07-21 14:30:15 | INFO     | pipelines.feature_pipeline | Fetching AQI data for Abbottabad...
2024-07-21 14:30:16 | INFO     | pipelines.feature_pipeline | Successfully fetched data!
```

If you see an error, common issues are:
- **"ModuleNotFoundError"** → Make sure venv is activated
- **"API key not found"** → Check that `.env` file exists and has your keys
- **"Connection error"** → Check your internet connection

---

## Step 10: First Git Commit

Now that everything works, save this milestone:

```bash
# Check what changed
git status

# Add all files to staging
git add .

# Commit with a meaningful message
git commit -m "init: Set up professional Python project structure with config and logging"

# Push to GitHub
git push origin main
```

Your code is now on GitHub! 🎉

---

## Troubleshooting

### "command not found: python"
- Python isn't installed. Follow Step 1 again.
- Or: Python isn't in your PATH. Reinstall and check "Add to PATH" during installation.

### "ModuleNotFoundError: No module named 'src'"
- Your venv isn't activated. Run `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)

### "API key error"
- Your `.env` file doesn't have the keys. Check that `.env` (not `.env.example`) exists in the project root.
- Make sure there are no spaces around the `=` sign: `AQICN_API_KEY=abc123` not `AQICN_API_KEY = abc123`

### "Connection refused"
- Hopsworks might be down (rare). Check [status.hopsworks.ai](https://status.hopsworks.ai)
- Or your internet is down. Check other websites.

---

## Next Steps

Once setup is complete, move to **Sprint 1: Tool Familiarization**:
- Learn Python basics
- Learn Pandas
- Learn how to use APIs with Python
- Practice fetching real data from AQICN

Start with the [Kaggle Learn Pandas course](https://kaggle.com/learn/pandas) (~4 hours).

---

## Quick Reference: Common Commands

```bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install new package (when working on the project)
pip install package_name
pip freeze > requirements.txt  # Update requirements.txt

# Run a pipeline
python -m pipelines.feature_pipeline

# Run the dashboard
streamlit run app/dashboard.py

# Run tests
pytest tests/ -v

# Deactivate virtual environment
deactivate
```

---

**Good luck! 🚀 You're all set to start the project.**
