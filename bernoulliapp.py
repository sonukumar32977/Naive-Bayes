import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Twitter Sentiment",
    page_icon="📩",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open('bernoullimodel.pkl', 'rb'))
vectorizer = pickle.load(open('bernoullivectorizer.pkl', 'rb'))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    p, div {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("📩 Twitter Sentiment Analysis")
st.markdown("Detect whether a message is **Positive Popularity or Negative Popularity** using Machine Learning.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About App")
st.sidebar.info("""
This app uses **Bernoulli Naive Bayes** model.

### How to use:
1. Enter your Tweet  
2. Click Analyze  
3. View prediction  

### Tech Stack:
- Python  
- Scikit-learn  
- Streamlit  
""")

# ---------------- INPUT ----------------
Tweet_input = st.text_area(
    "✉️ Enter your Tweet:",
    height=150,
    placeholder="Type your Tweet here..."
)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze Message"):

    if Tweet_input.strip() == "":
        st.warning("⚠️ Please enter a Tweet")
    else:
        # Transform input
        transformed_Tweet = vectorizer.transform([Tweet_input])

        # Prediction
        prediction = model.predict(transformed_Tweet)[0]
        prob = model.predict_proba(transformed_Tweet)[0]

        st.markdown("---")

        # ---------------- RESULT ----------------
        if prediction == 0:
            st.markdown("### 🚨 Negative Tweet")
            st.error("This message looks like Negative!")
        else:
            st.markdown("### ✅ Safe Message")
            st.success("This message is Postive.")

        # ---------------- PROBABILITY GRAPH ----------------
        df = pd.DataFrame({
            "Category": ["Negative", "Postive"],
            "Probability": prob
        })

        st.subheader("📊 Prediction Confidence")
        st.bar_chart(df.set_index("Category"))

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 Built with Streamlit | Machine Learning Project")