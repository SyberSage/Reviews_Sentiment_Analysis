# 💬 ChatGPT Reviews Sentiment Analysis

A machine learning and NLP based web application that classifies **ChatGPT app reviews** as **Positive, Neutral, or Negative**, combining a full sentiment analysis pipeline with an interactive dashboard for exploring review data and testing predictions live.

The application is built with **Python and Streamlit** and provides a simple interface where users can explore sentiment trends across the dataset or enter their own review text and get a live sentiment prediction.

## 🚀 Live Demo

**Streamlit App:** https://reviews-sentiment-analysis-sybersage.streamlit.app/
## 📌 Project Overview

The **ChatGPT Reviews Sentiment Analysis** project analyzes user reviews of the ChatGPT app to understand sentiment patterns beyond star ratings alone. Since reviews don't come with a ready-made sentiment label, sentiment is derived from the 1–5 star rating and predicted directly from review text using trained NLP models.

The project covers the complete pipeline:

* 🧹 Text Preprocessing & Cleaning
* 📊 Exploratory Data Analysis (EDA)
* 🔤 TF-IDF Feature Engineering
* 🤖 Multi-Model Training & Comparison (Classical ML + Deep Learning)
* 📈 Model Evaluation (Accuracy, Precision, Recall, F1, AUC-ROC)
* 🖥️ Interactive Streamlit Dashboard with Live Prediction

## ✨ Features

* User-friendly Streamlit interface with left sidebar navigation
* Overview page summarizing the project and dataset
* Insights page covering sentiment distribution, keyword analysis, and breakdowns by platform, location, version, and verified purchase status
* Live sentiment prediction on custom review text
* Side-by-side comparison between the trained model and a pretrained lexicon-based model (VADER)
* Leakage-safe train/test split methodology to ensure honest model evaluation
* Clean and responsive dashboard with word clouds and interactive charts

## 🧠 Machine Learning Models

Multiple machine learning and deep learning algorithms were evaluated, including:

* Multinomial Naive Bayes
* Logistic Regression
* Random Forest
* Linear SVM
* LSTM (Deep Learning)

Each model was trained on TF-IDF text features and evaluated on a **leakage-safe test split** — reviews held out at the unique-text level, not just the row level, to prevent duplicate text from inflating accuracy. The best-performing model, selected using Accuracy, F1-score, and AUC-ROC, was saved as a single deployable `.joblib` pipeline.

### Model Included

| Component          | File                          |
| ------------------- | ------------------------------ |
| Deployed Model       | `sentiment_pipeline.joblib`   |

The saved pipeline bundles the TF-IDF vectorizer, numeric feature scaler, and the calibrated classifier into a single scikit-learn `Pipeline`.

## 🔄 Machine Learning Workflow

The general workflow followed in the project is:

```text
Raw Review Dataset
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering (TF-IDF + Numeric Features)
   ↓
Leakage-Safe Train/Test Split
   ↓
Model Training (Classical ML + LSTM)
   ↓
Model Evaluation & Comparison
   ↓
Best Model Selection
   ↓
Model Serialization (.joblib)
   ↓
Streamlit Deployment
```

## 🛠️ Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Natural Language Processing

* NLTK (tokenization, stopwords, lemmatization, VADER)
* TF-IDF Vectorization

### Machine Learning & Deep Learning

* Naive Bayes
* Logistic Regression
* Random Forest
* Linear SVM
* LSTM (TensorFlow / Keras)

### Visualization

* Matplotlib
* Seaborn
* WordCloud

### Model Deployment

* Streamlit
* Joblib

### Development Tools

* Jupyter Notebook
* VS Code
* GitHub

## 📂 Project Structure

```text
ChatGPT-Reviews-Sentiment-Analysis/
│
├── 01_preprocessing.ipynb
├── 02_eda.ipynb
├── 03_model_building.ipynb
│
├── app.py
│
├── chatgpt_reviews_clean.csv
├── sentiment_pipeline.joblib
│
├── requirements.txt
│
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Navigate to the project directory

```bash
cd your-repository-name
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit
streamlit-option-menu
pandas
scikit-learn
tensorflow-cpu
nltk
matplotlib
seaborn
wordcloud
joblib
scipy
openpyxl
```

If your application uses additional packages, add them to `requirements.txt`.

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**.

Basic deployment steps:

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this repository.
5. Select `app.py` as the main file.
6. Deploy the application.

## ⚠️ Disclaimer

This project is intended **for educational and demonstration purposes only**.

The dataset used contains only 15 unique review templates repeated across 500 rows. On a leakage-safe evaluation, the trained model reaches limited accuracy (~40%), which is documented and explained throughout the notebooks rather than hidden. Predictions generated by this application should be treated as **illustrative, not authoritative**, and should not be used as the sole basis for real-world product decisions.

## 👨‍💻 Author

Yoga Prabhu

Data Science | Machine Learning | Python

Portfolio: https://sybersage.github.io/Portfolio/

GitHub: www.linkedin.com/in/yoga-prabhu

LinkedIn: www.linkedin.com/in/yoga-prabhu

---

⭐ If you found this project useful, consider giving the repository a star!
