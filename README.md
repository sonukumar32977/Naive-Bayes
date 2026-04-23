# Naive Bayes Algorithms

This repository demonstrates three different implementations of the Naive Bayes algorithm:

* Gaussian Naive Bayes
* Bernoulli Naive Bayes
* Multinomial Naive Bayes

These models are widely used for classification problems in machine learning.

---

## 📌 Overview

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' Theorem. It assumes that features are independent of each other, which makes it simple yet powerful for many real-world applications such as spam detection, sentiment analysis, and classification tasks.

---

## 📂 Project Structure

```
Naive-Bayes/
│── Gaussian.ipynb
│── Bernoulli.ipynb
│── Multinomial.ipynb
│── README.md
```

---

## ⚙️ Algorithms Explained

### 1. Gaussian Naive Bayes

* Used when features are continuous
* Assumes data follows a normal (Gaussian) distribution
* Common in datasets like medical data or numerical features
* Diabetes risk Prediction
* Project link 👇
* https://skdiabetes.streamlit.app/

### 2. Bernoulli Naive Bayes

* Used for binary/boolean features (0 or 1)
* Works well for text classification with binary word occurrence
* Twitter Sentiment Analysis
* Project link 👇
* https://sktwittersentiment.streamlit.app/

### 3. Multinomial Naive Bayes

* Used for discrete count data
* Commonly applied in NLP tasks like spam detection and document classification
* Email Spam Detection
* Project link 👇
* https://skspam.streamlit.app/

---

## 📊 Project Outputs

Each notebook includes:

### ✔️ Model Training

* Data preprocessing and splitting (train/test)
* Training the Naive Bayes model

### ✔️ Predictions

* Model predictions on test data
* Probability outputs using `predict_proba()`

### ✔️ Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report

### ✔️ ROC Curve & AUC Score

* ROC Curve visualization
* AUC (Area Under Curve) value to measure performance

Example:

* AUC Score ≈ **0.84** indicates a good model performance

---

## 📈 Output Interpretation

* **Accuracy** → Overall correctness of the model
* **Precision** → How many predicted positives are actually correct
* **Recall** → How many actual positives are captured
* **F1-score** → Balance between precision and recall
* **ROC Curve** → Shows trade-off between True Positive Rate and False Positive Rate
* **AUC Score** → Higher value (closer to 1) means better model

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## 📌 Conclusion

This project provides a clear comparison of three Naive Bayes algorithms and demonstrates how their performance varies depending on the type of data used.

---

## ⭐ Contributing

Feel free to fork this repository, improve it, and submit a pull request.

---

