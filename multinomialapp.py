import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Email Spam",
    page_icon="📩",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load(open('multinomialmodel.pkl', 'rb'))
vectorizer = joblib.load(open('multinomial_vectorizer.pkl', 'rb'))

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
st.title("📩 Email Spam Analysis")
st.markdown("Detect whether a message is **Spam or Not Spam** using Machine Learning.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About App")
st.sidebar.info("""
This app uses **Multinomial Naive Bayes** model.

### How to use:
1. Enter your email-messege 
2. Click Analyze  
3. View prediction  

### Tech Stack:
- Python  
- Scikit-learn  
- Streamlit  
""")

# ---------------- INPUT ----------------
spam_input = st.text_area(
    "✉️ Enter your email-message:",
    height=150,
    placeholder="Type your email-message here..."
)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze Message"):

    if spam_input.strip() == "":
        st.warning("⚠️ Please enter a email-message")
    else:
        # Transform input
        transformed_spam = vectorizer.transform([spam_input])

        # Prediction
        prediction = model.predict(transformed_spam)[0]
        prob = model.predict_proba(transformed_spam)[0]

        st.markdown("---")

        # ---------------- RESULT ----------------
        if prediction == 1:
            st.markdown("### 🚨 Spam")
            st.error("This message looks like Spam!")
        else:
            st.markdown("### ✅ Safe Message")
            st.success("This message is Not Spam.")

        # ---------------- PROBABILITY GRAPH ----------------
        df = pd.DataFrame({
            "Category": ["Spam", "Not Spam"],
            "Probability": prob
        })

        st.subheader("📊 Prediction Confidence")
        st.bar_chart(df.set_index("Category"))

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 Built with Streamlit | Machine Learning Project")