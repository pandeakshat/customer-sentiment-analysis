import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk import ngrams
import re

# Initialize NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Load data (assuming it's already cleaned)
df = pd.read_csv('data/tripadvisor_hotel_reviews.csv')

# Function to clean and tokenize text
def clean_and_tokenize(text):
    # Clean the text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    # Tokenize the text and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Function to extract n-grams (bigrams/trigrams) from tokens
def extract_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Apply Sentiment Analysis
df['Sentiment_Score'] = df['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['Cleaned_Review'] = df['Review'].apply(clean_and_tokenize)

# Streamlit UI setup
st.title("Customer Sentiment Analysis with User Review Input")

# User Input Section
st.subheader("Enter a Review and Rating")

# Text input for the review
user_review = st.text_area("Enter the review:", value="")

# Select rating input
user_rating = st.selectbox("Select a rating (1-5):", options=[1, 2, 3, 4, 5])

# Process user input
if user_review:
    st.write(f"**Review:** {user_review}")
    st.write(f"**Rating:** {user_rating}")
    
    # Clean and tokenize the input review
    cleaned_review = clean_and_tokenize(user_review)
    
    # Extract n-grams (bigrams)
    bigrams = extract_ngrams(cleaned_review, 2)
    ngram_phrases = [' '.join(ngram) for ngram in bigrams]
    
    # Count n-gram frequencies (for display purposes, single review doesn't need frequencies)
    ngram_counts = Counter(ngram_phrases)
    
    if user_rating in [1, 2]:
        st.write("**Identified Issues from Review:**")
        st.write(", ".join(ngram_phrases) if ngram_phrases else "No significant issues detected.")
    elif user_rating in [4, 5]:
        st.write("**Identified Strengths from Review:**")
        st.write(", ".join(ngram_phrases) if ngram_phrases else "No significant strengths detected.")
    else:
        st.write("This is a neutral review with no major strengths or issues detected.")
    
    # Optionally display a WordCloud for this specific review (based on n-grams)
    st.subheader("WordCloud for this Review")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_counts)
    fig_wc, ax_wc = plt.subplots(figsize=(10,6))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

# Existing functionality for generating WordClouds based on dataset
st.subheader("WordCloud for Low and High Ratings")

# Create WordCloud based on n-grams
def create_wordcloud_from_ngrams(reviews, n=2):
    # Extract n-grams from the reviews
    ngrams_list = [ngram for review in reviews for ngram in extract_ngrams(review, n)]
    
    # Convert n-grams into readable phrases for WordCloud
    ngram_phrases = [' '.join(ngram) for ngram in ngrams_list]
    
    # Count n-gram frequencies
    ngram_counts = Counter(ngram_phrases)
    
    # Create a WordCloud object from the most common n-grams
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_counts)
    
    # Return the WordCloud object
    return wordcloud

# Low ratings (e.g., 1-2)
st.write("**Low Rating Feedback (Issues)**")
low_rating_reviews = df[df['Rating'].isin([1, 2])]['Cleaned_Review'].tolist()
wordcloud_low = create_wordcloud_from_ngrams(low_rating_reviews, n=2)
fig_low, ax_low = plt.subplots(figsize=(10,6))
ax_low.imshow(wordcloud_low, interpolation='bilinear')
ax_low.axis("off")
st.pyplot(fig_low)

# High ratings (e.g., 4-5)
st.write("**High Rating Feedback (Strengths)**")
high_rating_reviews = df[df['Rating'].isin([4, 5])]['Cleaned_Review'].tolist()
wordcloud_high = create_wordcloud_from_ngrams(high_rating_reviews, n=2)
fig_high, ax_high = plt.subplots(figsize=(10,6))
ax_high.imshow(wordcloud_high, interpolation='bilinear')
ax_high.axis("off")
st.pyplot(fig_high)

# Sentiment Score vs Rating boxplot
st.subheader("Sentiment Score vs Rating")
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.boxplot(x=df['Rating'], y=df['Sentiment_Score'], ax=ax4)
ax4.set_title('Sentiment Score vs Rating')
ax4.set_xlabel('Rating')
ax4.set_ylabel('Sentiment Score')
st.pyplot(fig4)
