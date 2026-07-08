import streamlit as st
import joblib

# ----------------------------
# Load Model & TF-IDF Vectorizer
# ----------------------------
model = joblib.load("fake_news_model (1).pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ----------------------------
# Title
# ----------------------------
st.title("📰 Fake News Detection System")
st.markdown("### Predict whether a news article is **Fake** or **Real** using Machine Learning.")

st.divider()

# ----------------------------
# User Input
# ----------------------------
news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste the news article here..."
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict", use_container_width=True):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        # TF-IDF Transformation
        news_vector = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(news_vector)[0]

        st.divider()

        if prediction == 0:
            st.error("🚨 Prediction: FAKE NEWS")
        else:
            st.success("✅ Prediction: REAL NEWS")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Project Information")

st.sidebar.write("**Model:** Decision Tree")

st.sidebar.write("**Vectorizer:** TF-IDF")

st.sidebar.write("**Algorithm:** Machine Learning")

st.sidebar.write("**Project:** Fake News Detection")

st.sidebar.markdown("---")

st.sidebar.info(
    "Developed using Streamlit, Scikit-learn, and NLP."
)