Absolutely ✅ — here’s your complete **README.md** file, fully formatted and ready to use on GitHub.
Just copy and paste this into your project’s `README.md` file.

---

```markdown
# 🍽️ Food Nutrition Analyzer

A full-featured **AI-powered application** that helps analyze food nutrition from images or manual search, offers dietary and health guidance, and includes an integrated chatbot for nutrition questions!

---

## 🧭 Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [How to Use](#how-to-use)
  - [1. Running the Food Analyzer Web App](#1-running-the-food-analyzer-web-app)
  - [2. Downloading-the-Model](#2-downloading-the-model)
  - [3. Converting Models (Optional)](#3-converting-models-optional)
- [Functionality Details](#functionality-details)
- [Chatbot (OpenRouter Integration)](#chatbot-openrouter-integration)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## 🚀 Features

- **🍔 Food Image Recognition** — Recognize dishes and ingredients from uploaded images using a deep learning model (EfficientNetB0, Food-101).
- **🔍 Manual Food Search** — Search for any food by name and retrieve nutrition details from an integrated CSV database.
- **📊 Track Daily Nutrition** — Add foods (via image or search), log their nutrition values, and view daily totals.
- **💪 Personalized Diet & Exercise Plan** — Enter your weight, height, age, and health condition. The app computes BMI and suggests personalized diet and workout plans (with YouTube exercise videos).
- **🤖 AI Nutrition Chatbot** — Ask any food or nutrition question, powered by **OpenRouter (GPT-powered AI)**.
- **⚙️ Easy Model Download & Conversion** — Scripts to fetch the neural net model and convert between `.h5` and `.keras` formats.

---

## 🗂️ Directory Structure

```

food_analyzer/
│
├── scripts/
│   ├── food_app.py               # Main Streamlit web app
│   ├── convert.py                # Script: Convert .h5 to .keras model format
│   └── download_food_model.py    # Script: Download model (.keras from GitHub)
│
├── models/
│   ├── efficientnetb0_food101.h5
│   ├── efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5
│   ├── tf_model.h5
│   └── tf_model_fixed.keras
│
├── data/
│   └── nutrition_data.csv         # Food and nutrition lookup database
│
├── resources/
│   ├── classes.txt                # Food-101 style class labels
│   └── labels.txt                 # Original label definitions
│
├── venv/                          # Virtual environment (not tracked by Git)
├── README.md

````

---

## ⚙️ Getting Started

### 1️⃣ Clone and Set Up Environment

```bash
git clone <REPO_URL>
cd food_analyzer
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
````

### 2️⃣ Download the Model

```bash
python scripts/download_food_model.py
```

This fetches the required `.keras` model into the `models/` directory.

### 3️⃣ (Optional) Convert Models Between Formats

```bash
python scripts/convert.py efficientnetb0_food101.h5 efficientnetb0_food101.keras
```

---

## 💡 How to Use

### 1. Running the Food Analyzer Web App

```bash
streamlit run scripts/food_app.py
```

Then open the provided URL (usually [http://localhost:8501](http://localhost:8501)) in your browser.

---

### 2. Downloading the Model

If not already present:

```bash
python scripts/download_food_model.py
```

This saves `tf_model_fixed.keras` inside the `models/` folder.

---

### 3. Converting Models (Optional)

Convert `.h5` → `.keras`:

```bash
python scripts/convert.py efficientnetb0_food101.h5 efficientnetb0_food101.keras
```

---

## 🧠 Functionality Details

* **📷 Image Upload & Recognition:**
  Upload any food image to predict its class using a pre-trained EfficientNetB0 model fine-tuned on Food-101.

* **🔎 Manual Lookup:**
  Enter a food name and view its calories and macronutrient breakdown.

* **📆 Daily Log:**
  Add recognized foods to your diary and track total daily intake.

* **🥗 Diet & Exercise Plans:**
  Based on your BMI and health condition, receive a custom plan with foods and exercises (linked to YouTube videos).

* **💬 AI Chatbot:**
  Ask any nutrition-related question using an integrated GPT model via OpenRouter.

---

## 🤖 Chatbot (OpenRouter Integration)

* Powered by GPT-OSS 20B or similar models via [OpenRouter](https://openrouter.ai).
* Obtain an API key from OpenRouter and set it as an environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

> 💡 **Tip:** Never share your personal API keys publicly or commit them to GitHub.

---

## 🖼️ Screenshots

> Add screenshots of your app interface here (upload images and reference them):

* Home Screen
* Food Image Recognition
* Manual Search
* Daily Log
* Diet & Exercise Plan
* Chatbot Interface

---

## 📦 Requirements

* **Python:** 3.8+
* **Packages:**
  `streamlit`, `tensorflow`, `pandas`, `numpy`, `Pillow`, `requests`
* **Browser:** Modern browser for Streamlit interface

---

## 🧩 Troubleshooting

| Problem              | Solution                                                         |
| -------------------- | ---------------------------------------------------------------- |
| **Model Not Found**  | Run `download_food_model.py` and check `models/` folder          |
| **API Key Error**    | Ensure valid OpenRouter API key is exported correctly            |
| **File Path Error**  | Verify model and resource file paths                             |
| **Module Not Found** | Install all dependencies using `pip install -r requirements.txt` |

---

## 🙌 Acknowledgments

* [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* [OpenRouter](https://openrouter.ai)
* [Streamlit](https://streamlit.io/)

---

### 🥳 Enjoy exploring the world of food and nutrition with AI!

```

---

