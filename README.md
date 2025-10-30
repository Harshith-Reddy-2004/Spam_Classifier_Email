# 📧 SMS/Email Spam Classifier

An end-to-end **Machine Learning NLP project** that classifies messages as **Spam** or **Ham (Not Spam)** using text preprocessing, TF-IDF vectorization, and multiple ML models.  
Deployed using **Streamlit** for an interactive web experience.

---

## 🚀 Project Overview

- **Goal:** Build a model that detects spam in SMS or email messages.  
- **Dataset:** [SMS Spam Collection Dataset – UCI Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- **Tech Stack:** Python, Pandas, Scikit-learn, NLTK, Streamlit  
- **ML Focus:** Natural Language Processing (NLP) + Supervised Learning  

---

## 🧹 1. Data Cleaning

- Removed unnecessary and unnamed columns.  
- Renamed columns: `v1 → target`, `v2 → text`.  
- Checked for null values (none found).  
- Removed duplicates to ensure data consistency.

---

## 📊 2. Exploratory Data Analysis (EDA)

- Class distribution: **Spam ≈ 12.63%**, **Ham ≈ 87.37%**.  
- Visualized:
  - Message **length**, **word count**, and **sentence count**.
  - **Correlation matrix** among numeric features and target.
- Observed spam messages generally have **longer lengths** and **specific keywords** (like *win*, *free*, *call*).

---

## 🧠 3. Text Preprocessing

- Converted all text to lowercase.  
- Tokenized using `nltk.word_tokenize()`.  
- Removed special characters, punctuation, and stopwords.  
- Applied **stemming** using `PorterStemmer`.  
- Visualized frequent words with **WordCloud**.  
- Vectorized messages using **TF-IDF** (max_features = 3000).

---

## 🤖 4. Model Building

Split: **80% train / 20% test**

Models tested:
- Naïve Bayes: `GaussianNB`, `MultinomialNB`, `BernoulliNB`
- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree, Random Forest  
- AdaBoost, Bagging, Gradient Boosting  
- XGBoost

---

## 📈 5. Model Evaluation

Metrics used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

| Model | Accuracy | Precision |
|:------|:----------|:-----------|
| Naïve Bayes | 97.2% | 1.0 |
| Logistic Regression | 98.0% | 0.99 |
| SVM | 97.8% | 0.99 |
| Random Forest | 97.5% | 0.98 |

✅ **Best Model:** `Multinomial Naïve Bayes`  
Achieved **97.2% accuracy** and **100% precision** — meaning **no legitimate message was flagged as spam**.

---

## 🧩 6. Model Improvement

- Added `num_characters` feature.  
- Tuned TF-IDF parameter `max_features = 3000`.  
- Applied **feature scaling**.  
- Ensemble methods:
  - **Voting Classifier (SVM + NB + ExtraTrees)** → Accuracy: 0.9816  
  - **Stacking Classifier (SVM + NB + ExtraTrees)** → Accuracy: 0.9787  

### ⚖️ Precision as Key Metric

False Positives (legitimate messages marked as spam) are **more severe** than False Negatives.  
Hence, **Precision** is prioritized over Accuracy.

\[
Precision = \frac{TP}{TP + FP}
\]

---

## 🌐 7. Streamlit Web App

A lightweight **Streamlit** interface for real-time spam prediction.  

🔗 **Live Demo:**  
[Spam Classifier Web App](https://spamclassifieremail-m8luarknjk5odsbkirg8nw.streamlit.app/)

## 🧑‍💻 Author

**Harshith Reddy**  
Machine Learning & Data Science Enthusiast
