import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1️ Load Tokenizer & Models
# -------------------------------
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

lstm_model = tf.keras.models.load_model("sentiment_model.h5")
#gru_model = tf.keras.models.load_model("gru_model.h5")
maxlen = 100  # same as training

# -------------------------------
# 2️ Page config & custom styling
# -------------------------------
st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background-color: #f0f2f6; color: #333; font-family: 'Segoe UI', sans-serif;}
.stButton>button {background-color: #4CAF50; color: white; font-size: 18px; border-radius: 8px; padding: 10px 20px;}
.stTextArea>textarea {font-size: 16px;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 3️ Sidebar
# -------------------------------
st.sidebar.title("Options")
st.sidebar.info("""
- Choose the model: LSTM or GRU  
- Enter multiple reviews separated by line breaks  
- Click Predict to see results
""")

model_choice = st.sidebar.radio("Select Model:", ["LSTM", "GRU"])

# -------------------------------
# 4️ Main UI
# -------------------------------
st.title("🛒 Amazon Reviews Sentiment Analysis")
st.subheader("Detect Positive or Negative reviews using Deep Learning")

user_input = st.text_area("Enter Amazon reviews (one per line):", height=200)

# -------------------------------
# 5️ Prediction
# -------------------------------
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter at least one review!")
    else:
        # Split reviews by lines
        reviews = [r.strip() for r in user_input.strip().split("\n") if r.strip()]
        sequences = tokenizer.texts_to_sequences(reviews)
        padded = pad_sequences(sequences, maxlen=maxlen, padding='pre')

        # Choose model
        model = lstm_model if model_choice == "LSTM" else gru_model

        preds = model.predict(padded)
        sentiments = ["Positive" if p.argmax() == 1 else "Negative" for p in preds]
        confidences = [p.max()*100 for p in preds]

        # Display results in a dataframe
        df = pd.DataFrame({
            "Review": reviews,
            "Sentiment": sentiments,
            "Confidence (%)": confidences
        })
        st.table(df)

        # Optional: plot bar chart of Positive vs Negative
        counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind='bar', color=['#4CAF50','#FF5252'], ax=ax)
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Number of Reviews")
        st.pyplot(fig)



