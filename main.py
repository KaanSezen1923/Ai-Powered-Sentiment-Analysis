from transformers import pipeline
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("Sentiment Analysis App")
st.write("This app uses a pre-trained model to analyze the sentiment of text.")


sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

query=st.text_area("Enter text for sentiment analysis:")

analyze=st.button("Analyze")

if analyze:
    if query.strip():
        #Sentiment Analysis
        st.subheader("Sentiment Analysis Result")
        results= sentiment_analyzer(query)[0]
        label=results['label']
        score=results['score']
        st.write(f"Sentiment: {label}")
        st.write(f"Confident Score: {score:.4f}")

        # Pie Chart
        st.subheader("Sentiment Distribution")
        labels = ["Positive", "Negative"]
        scores = [score if label == "POSITIVE" else 0, score if label == "NEGATIVE" else 1 - score]
        fig, ax = plt.subplots()
        ax.pie(scores, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.write("Please enter some text to analyze.")

