import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SMS Spam Detector",
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
st.title("📩 SMS Spam Detection System")
st.markdown("Detect whether a message is **Spam or Not Spam** using Machine Learning.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About App")
st.sidebar.info("""
This app uses **Bernoulli Naive Bayes** model.

### How to use:
1. Enter your SMS  
2. Click Analyze  
3. View prediction  

### Tech Stack:
- Python  
- Scikit-learn  
- Streamlit  
""")

# ---------------- INPUT ----------------
sms_input = st.text_area(
    "✉️ Enter your message:",
    height=150,
    placeholder="Type your SMS here..."
)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze Message"):

    if sms_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Transform input
        transformed_sms = vectorizer.transform([sms_input])

        # Prediction
        prediction = model.predict(transformed_sms)[0]
        prob = model.predict_proba(transformed_sms)[0]

        st.markdown("---")

        # ---------------- RESULT ----------------
        if prediction == 1:
            st.markdown("### 🚨 Spam Detected")
            st.error("This message looks like SPAM!")
        else:
            st.markdown("### ✅ Safe Message")
            st.success("This message is NOT spam.")

        # ---------------- PROBABILITY GRAPH ----------------
        df = pd.DataFrame({
            "Category": ["Not Spam", "Spam"],
            "Probability": prob
        })

        st.subheader("📊 Prediction Confidence")
        st.bar_chart(df.set_index("Category"))

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 Built with Streamlit | Machine Learning Project")