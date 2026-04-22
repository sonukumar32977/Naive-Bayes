import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load('gaussian_model.pkl')
except Exception as e:
    st.error(f"Model not loaded: {e}")

# ---------------- TITLE ----------------
st.title("🩺 Diabetes Prediction System")
st.markdown("Predict whether a person has diabetes using **Gaussian Naive Bayes**")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 About")
st.sidebar.info("""
This app predicts diabetes based on medical inputs.

Model used:
- Gaussian Naive Bayes

Dataset:
- PIMA Diabetes Dataset
""")

# ---------------- INPUT FIELDS ----------------
st.subheader("Enter Patient Details")

Pregnancies = st.number_input("Pregnancies", min_value=0, value=6)
Glucose = st.number_input("Glucose Level", value=9)
BloodPressure = st.number_input("Blood Pressure", value=50)
SkinThickness = st.number_input("Skin Thickness", value=35)
Insulin = st.number_input("Insulin Level", value=0)
BMI = st.number_input("BMI", value=33.6)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", value=0.627)
Age = st.number_input("Age", value=50)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    input_data = np.array([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.markdown("---")

    # RESULT
    if prediction == 1:
        st.error("⚠️ High chance of Diabetes")
    else:
        st.success("✅ Low chance of Diabetes")

    # PROBABILITY
    st.subheader("Prediction Confidence")
    st.write(f"Not Diabetic: {prob[0]:.2f}")
    st.write(f"Diabetic: {prob[1]:.2f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💡 Built with Streamlit")