# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# ------------------- Hide TensorFlow Warnings -------------------
tf.get_logger().setLevel('ERROR')  # Hide TF warnings like reset_default_graph
import logging
logging.getLogger("absl").setLevel(logging.ERROR)  # Hide absl warnings

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # optional for inference

# ------------------- Helper Functions -------------------
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ------------------- Streamlit UI -------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
)

# Sidebar
st.sidebar.header("ℹ️ About this App")
st.sidebar.write("""
This app predicts whether a movie review is **Positive** or **Negative** using a pre-trained RNN model.
- Enter your movie review in the text area.
- Click 'Classify' to see the sentiment.
- The model predicts based on the IMDB dataset.
""")
st.sidebar.markdown("Developed by **Akshita** | ML & NLP Enthusiast")

# Home page / Title & instructions
st.markdown("""
# 🎬 IMDB Movie Review Sentiment Analysis
Welcome! This app predicts whether a movie review is **Positive** or **Negative** using a pre-trained RNN model.

💡 **Instructions:**
1. Enter your movie review in the box below.
2. Click the **Classify** button.
3. View the predicted sentiment and score.
""")

# User input
user_input = st.text_area('✍️ Enter your Movie Review here', placeholder="Type your review...")

# Prediction logic
if st.button('Classify'):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment_score = prediction[0][0]
        sentiment = 'Positive 😊' if sentiment_score > 0.5 else 'Negative 😞'
        
        # Display result with colored feedback
        if sentiment_score > 0.5:
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")
        st.info(f'Prediction Score: `{sentiment_score:.4f}`')
    else:
        st.warning("⚠️ Please enter a review before clicking classify.")
else:
    st.write("📌 Enter a movie review above and click **Classify** to see the sentiment.")
