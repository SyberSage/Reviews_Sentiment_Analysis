import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import joblib
import re
import nltk
from streamlit_option_menu import option_menu

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="ChatGPT Reviews Sentiment Dashboard", layout="wide")

SENTIMENT_ORDER = ['Negative', 'Neutral', 'Positive']
SENTIMENT_COLORS = {'Negative': '#e74c3c', 'Neutral': '#3498db', 'Positive': '#2ecc71'}

SMALL = (4.5, 3)
MED = (6, 3.2)

# Cached loaders

@st.cache_data
def load_data():
    return pd.read_csv('chatgpt_reviews_clean.csv')

@st.cache_resource
def load_pipeline():
    return joblib.load('sentiment_pipeline.joblib')

@st.cache_resource
def load_cleaning_tools():
    stop_words = set(stopwords.words('english'))
    negation_words = {"not", "no", "nor", "never", "cannot", "n't", "don", "didn", "doesn",
                       "isn", "wasn", "weren", "won", "wouldn", "couldn", "shouldn", "aren"}
    stop_words = stop_words - negation_words
    lemmatizer = WordNetLemmatizer()
    sia = SentimentIntensityAnalyzer()
    return stop_words, lemmatizer, sia

df = load_data()
pipeline = load_pipeline()
stop_words, lemmatizer, sia = load_cleaning_tools()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    tokens = [lemmatizer.lemmatize(t, pos='n') for t in tokens]
    return ' '.join(tokens)

def vader_predict(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        label = 'Positive'
    elif score <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    intensity = min(abs(score) * 100, 100)
    return label, intensity, score

def top_words(text_series, n=15):
    words = ' '.join(text_series.dropna()).split()
    return pd.DataFrame(Counter(words).most_common(n), columns=['word', 'count'])

# Left sidebar navigation

with st.sidebar:
    st.title("ChatGPT Reviews")
    st.caption("Sentiment Analysis Dashboard")
    page = option_menu(
        menu_title=None,
        options=["Overview", "Predictions", "Insights"],
        icons=["info-circle", "cpu", "search"],
        default_index=0,
    )

# PAGE 1: OVERVIEW  (about page)

if page == "Overview":
    st.title("About This Dashboard")
    st.markdown("### Abstract")
    st.markdown("""
This project applies natural language processing and machine learning 
to classify the sentiment of ChatGPT app reviews as Positive, 
Neutral, or Negative, moving beyond star ratings to understand 
user feedback directly from review text. The pipeline covers text 
preprocessing, exploratory analysis, TF-IDF feature engineering, and 
training five models — Naive Bayes, Logistic Regression, Random 
Forest, Linear SVM, and an LSTM — evaluated using accuracy, precision, 
recall, F1-score, and AUC-ROC on a leakage-safe test split to ensure results 
reflect genuine generalization. A calibrated Linear SVM was selected as the deployed model and is presented alongside a pretrained lexicon-based tool (VADER) for comparison.

""")
    st.markdown("**Approach:** data cleaning, exploratory analysis, model comparison, and a live prediction tool")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(df)}")
    col2.metric("Unique Review Texts", f"{df['review_clean'].nunique()}")
    col3.metric("Average Rating", f"{df['rating'].mean():.2f} / 5")
    col4.metric("Positive Share", f"{(df['sentiment']=='Positive').mean()*100:.0f}%")

    st.markdown("### What's in each tab")
    st.markdown("""
- **Predictions** — type in any review text and get a live sentiment prediction from the deployed model, shown alongside pretrained VADER for comparison.
- **Insights** — the full set of analysis questions: sentiment distribution, rating alignment, keywords per class, trends over time, and breakdowns by platform, location, version, and verification status.
""")

    st.markdown("### How the sentiment labels were built")
    st.markdown("""
Reviews don't come with a ready-made sentiment label, only a 1–5 star rating. Sentiment was derived
from that rating: **1–2 stars → Negative, 3 stars → Neutral, 4–5 stars → Positive.**
""")

    st.markdown("### Model & a key limitation to know before using Predictions")
    st.info(
        "This dataset contains only **15 unique review texts** repeated across 500 rows (synthetic/templated data). "
        "A properly leakage-safe evaluation — testing the model on review templates it never trained on — shows "
        "the deployed **Linear SVM** reaches only **~40% accuracy** (macro AUC ~0.55). Pretrained **VADER**, which "
        "wasn't trained on this dataset at all, reaches **~78%** on the same test set. Both are shown together on "
        "the Predictions page so you can weigh them against each other rather than trust either blindly."
    )

# PAGE 2: PREDICTIONS

elif page == "Predictions":
    st.title("Predictions")
    st.subheader("Predict Sentiment for New Review Text")

    st.warning(
        "**Accuracy caveat:** on a leakage-safe test split, our trained Linear SVM scored ~40% accuracy "
        "(macro AUC ~0.55), while pretrained VADER scored ~78% on the same test set -- though VADER also "
        "has its own blind spot on neutral/ambiguous language. This dataset has only 15 unique training "
        "templates, which limits how well the trained model generalizes to new phrasing. Treat both "
        "predictions below as informative, not authoritative -- especially the trained model's."
    )

    user_text = st.text_area("Enter review text:", placeholder="e.g. The app crashes every time I open it, very frustrating.")

    if st.button("Predict Sentiment") and user_text.strip():
        cleaned = clean_text(user_text)
        review_length_clean = len(user_text)
        clean_token_count = len(cleaned.split())

        input_row = pd.DataFrame([{
            'review_clean': cleaned,
            'review_length_clean': review_length_clean,
            'clean_token_count': clean_token_count,
        }])

        pred = pipeline.predict(input_row)[0]
        proba = pipeline.predict_proba(input_row)[0]
        proba_dict = dict(zip(pipeline.classes_, proba))
        svm_conf = proba_dict[pred] * 100

        vlabel, vconf, vcompound = vader_predict(user_text)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Trained Model (Linear SVM)")
            st.markdown(f"## {svm_conf:.0f}% {pred}")
            st.bar_chart(pd.Series(proba_dict).reindex(SENTIMENT_ORDER))

        with col2:
            st.markdown("#### Pretrained VADER")
            st.markdown(f"## {vconf:.0f}% {vlabel}")
            st.caption(f"(compound score: {vcompound:.3f} -- intensity shown as a confidence proxy, not a calibrated probability)")
            st.progress(min(max((vcompound + 1) / 2, 0.0), 1.0))

# PAGE 3: INSIGHTS

elif page == "Insights":
    st.title("Insights")
    st.caption("All analysis questions, from overall sentiment down to category-level breakdowns.")

    # Q1
    with st.expander("1. What is the overall sentiment of user reviews?", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=SMALL)
            counts = df['sentiment'].value_counts().reindex(SENTIMENT_ORDER)
            ax.pie(counts, labels=counts.index, autopct='%1.0f%%',
                   colors=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER], startangle=90,
                   textprops={'fontsize': 8})
            st.pyplot(fig, use_container_width=False)
        with col2:
            st.metric("Positive", f"{(df['sentiment']=='Positive').mean()*100:.1f}%")
            st.metric("Neutral", f"{(df['sentiment']=='Neutral').mean()*100:.1f}%")
            st.metric("Negative", f"{(df['sentiment']=='Negative').mean()*100:.1f}%")

    # Q2
    with st.expander("2. How does sentiment vary by rating?"):
        st.caption(
            "`sentiment` is derived directly from `rating` (1-2\u2192Negative, 3\u2192Neutral, 4-5\u2192Positive), "
            "so this mapping is exact by construction."
        )
        ct = pd.crosstab(df['rating'], df['sentiment']).reindex(columns=SENTIMENT_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=SMALL)
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={'size': 8}, cbar=False)
        ax.set_xlabel('Sentiment', fontsize=8); ax.set_ylabel('Rating', fontsize=8)
        ax.tick_params(labelsize=7)
        st.pyplot(fig, use_container_width=False)

    # Q3
    with st.expander("3. Which keywords are most associated with each sentiment class?"):
        cols = st.columns(3)
        for c, label, cmap in zip(cols, SENTIMENT_ORDER, ['Reds', 'Blues', 'Greens']):
            with c:
                st.markdown(f"**{label}**")
                text = ' '.join(df.loc[df['sentiment'] == label, 'review_clean'].dropna())
                if text.strip():
                    wc = WordCloud(width=300, height=200, background_color='white', colormap=cmap).generate(text)
                    fig, ax = plt.subplots(figsize=(3, 2))
                    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                    st.pyplot(fig, use_container_width=False)
                st.dataframe(top_words(df.loc[df['sentiment'] == label, 'review_clean'], n=6), hide_index=True, height=180)

    # Q4
    if 'date_clean' in df.columns:
        with st.expander("4. How has sentiment changed over time?"):
            df_time = df.copy()
            df_time['date_clean'] = pd.to_datetime(df_time['date_clean'], errors='coerce')
            df_time = df_time.dropna(subset=['date_clean'])

            if len(df_time) == 0:
                st.info("No valid, parseable dates available to show a time trend.")
            else:
                st.caption(f"Using {len(df_time)} / {len(df)} rows with a valid date.")
                df_time['month'] = df_time['date_clean'].dt.to_period('M').astype(str)
                monthly = pd.crosstab(df_time['month'], df_time['sentiment'])
                monthly = monthly.reindex(columns=SENTIMENT_ORDER, fill_value=0).sort_index()
                fig, ax = plt.subplots(figsize=SMALL)
                monthly.plot(kind='line', marker='o', ax=ax, color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER], linewidth=1.5, markersize=4)
                ax.set_ylabel('Reviews', fontsize=8); ax.set_xlabel('Month', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)
                plt.xticks(rotation=45)
                st.pyplot(fig, use_container_width=False)

    # Q5
    with st.expander("5. Do verified users leave more positive or negative reviews?"):
        ct = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize='index')
        ct = ct.reindex(columns=SENTIMENT_ORDER, fill_value=0) * 100
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=SMALL)
            ct.plot(kind='bar', ax=ax, color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
            ax.set_ylabel('% of reviews', fontsize=8); ax.set_xlabel('')
            ax.tick_params(labelsize=7); ax.legend(fontsize=6)
            plt.xticks(rotation=0)
            st.pyplot(fig, use_container_width=False)
        with col2:
            st.dataframe(ct.round(1))

    # Q6
    with st.expander("6. Are longer reviews more likely to be negative or positive?"):
        length_col = 'review_length_clean' if 'review_length_clean' in df.columns else 'review_length'
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=SMALL)
            sns.boxplot(data=df, x='sentiment', y=length_col, order=SENTIMENT_ORDER, ax=ax,
                        palette=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
            ax.set_ylabel('Length', fontsize=8); ax.set_xlabel('')
            ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=False)
        with col2:
            st.markdown("**Average length by sentiment**")
            st.dataframe(df.groupby('sentiment')[length_col].mean().reindex(SENTIMENT_ORDER).round(1).rename('avg_length'))

    # Q7 
    with st.expander("7. Which locations show the most positive or negative sentiment?"):
        ct = pd.crosstab(df['location'], df['sentiment'], normalize='index')
        ct = ct.reindex(columns=SENTIMENT_ORDER, fill_value=0) * 100
        ct = ct.sort_values('Positive', ascending=False)
        fig, ax = plt.subplots(figsize=SMALL)
        ct.plot(kind='bar', stacked=True, ax=ax, color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
        ax.set_ylabel('% of reviews', fontsize=8); ax.set_xlabel('')
        ax.tick_params(labelsize=7); ax.legend(fontsize=6)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig, use_container_width=False)

    # Q8
    with st.expander("8. Is there a difference in sentiment across platforms?"):
        
        ct = pd.crosstab(df['platform'], df['sentiment'], normalize='index')
        ct = ct.reindex(columns=SENTIMENT_ORDER, fill_value=0) * 100
        fig, ax = plt.subplots(figsize=SMALL)
        ct.plot(kind='bar', stacked=True, ax=ax, color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
        ax.set_ylabel('% of reviews', fontsize=8); ax.set_xlabel('')
        ax.tick_params(labelsize=7); ax.legend(fontsize=6)
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig, use_container_width=False)

    # Q9
    with st.expander("9. Which app versions are associated with higher/lower sentiment?"):
        ct = pd.crosstab(df['version'], df['sentiment'], normalize='index')
        ct = ct.reindex(columns=SENTIMENT_ORDER, fill_value=0) * 100
        fig, ax = plt.subplots(figsize=SMALL)
        ct.plot(kind='bar', stacked=True, ax=ax, color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
        ax.set_ylabel('% of reviews', fontsize=8); ax.set_xlabel('')
        ax.tick_params(labelsize=7); ax.legend(fontsize=6)
        plt.xticks(rotation=0)
        st.pyplot(fig, use_container_width=False)

    # Q10
    with st.expander("10. What are the most common negative feedback themes?"):
        neg_words = top_words(df.loc[df['sentiment'] == 'Negative', 'review_clean'], n=10)
        fig, ax = plt.subplots(figsize=SMALL)
        sns.barplot(data=neg_words, x='count', y='word', ax=ax, palette='Reds_r')
        ax.set_xlabel('Count', fontsize=8); ax.set_ylabel('')
        ax.tick_params(labelsize=7)
        st.pyplot(fig, use_container_width=False)