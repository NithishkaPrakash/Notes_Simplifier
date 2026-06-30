import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------
# Load API Key
# ---------------------------------
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ---------------------------------
# Streamlit Page
# ---------------------------------
st.set_page_config(
    page_title="AI Notes Simplifier",
    page_icon="📘",
    layout="wide"
)

st.title("📘 AI Notes Simplifier")
st.write("Paste your notes below and click **Simplify Notes**.")

notes = st.text_area(
    "Enter your Notes",
    height=300,
    placeholder="Paste your notes here..."
)

# ---------------------------------
# Keyword Extraction
# ---------------------------------
def extract_keywords(text, top_n=10):

    vectorizer = TfidfVectorizer(stop_words="english")

    matrix = vectorizer.fit_transform([text])

    feature_names = vectorizer.get_feature_names_out()

    scores = matrix.toarray()[0]

    word_scores = list(zip(feature_names, scores))

    word_scores.sort(key=lambda x: x[1], reverse=True)

    return [word for word, score in word_scores[:top_n]]

# ---------------------------------
# Button
# ---------------------------------
if st.button("🚀 Simplify Notes"):

    if notes.strip() == "":
        st.warning("Please enter some notes.")
    else:

        with st.spinner("Gemini is simplifying your notes..."):

            prompt = f"""
You are an AI Notes Simplifier.

Read the notes carefully.

Generate:

1. Short Summary

2. Important Bullet Points

3. Easy Explanation

Use simple English.

Notes:

{notes}
"""

            try:

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                summary = response.text

                keywords = extract_keywords(notes)

                st.success("Notes Simplified Successfully!")

                st.subheader("📄 Simplified Notes")

                st.write(summary)

                st.subheader("🔑 Important Keywords")

                for word in keywords:
                    st.write("•", word)

            except Exception as e:

                st.error(f"Error: {e}")
