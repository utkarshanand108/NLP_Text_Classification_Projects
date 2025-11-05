# DataScienceCourse7_Project
**NLP Projects — IMDb Sentiment Analysis & News Article Classification**  
*Internshala Data Science PGC — Course 7 Project by Utkarsh Anand*

---

## 🧩 Overview
This repository combines **two major NLP projects**:

### 🎬 Part A — IMDb Sentiment Analysis  
Builds a sentiment classifier on 50,000 IMDb movie reviews to predict **positive** or **negative** sentiment using TF-IDF and classical ML.

### 📰 Part B — News Article Classification  
Classifies 50,000 news articles into **10 categories** (Politics, Sports, Tech, etc.) using TF-IDF and multi-class linear models.

---

## 🧠 Part A — IMDb Sentiment Analysis
- **Dataset:** IMDb reviews (`review`, `sentiment`)
- **Goal:** Predict review sentiment (Positive / Negative)

### Steps
1. **Data Cleaning:** lowercasing, punctuation removal, stopword removal, lemmatization  
2. **Feature Extraction:** TF-IDF with 5000 features  
3. **Models Compared:** Logistic Regression, Linear SVM, Multinomial Naive Bayes  
4. **Evaluation Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix  
5. **Visualization:** Word clouds for positive vs. negative reviews  

### 📈 Key Results
| Model | Accuracy | F1 Score |
|--------|-----------|----------|
| Logistic Regression | 0.885 | 0.887 |
| Linear SVM | 0.877 | 0.879 |
| Multinomial NB | 0.851 | 0.853 |

✅ **Best Model:** Logistic Regression (TF-IDF features)  
🧩 **Interpretation:** Linear models work best on sparse high-dimensional text.  

---

## 🗞️ Part B — News Article Classification
- **Dataset:** 50,000 labeled articles (category, headline, short description)
- **Goal:** Predict article category among 10 classes

### Steps
1. **Data Cleaning:** remove punctuation, lowercase, remove stopwords  
2. **Feature Extraction:** TF-IDF (max 5000 features)  
3. **Model Comparison:** Logistic Regression, Linear SVM, Multinomial Naive Bayes  
4. **Validation:** 5-Fold Stratified CV (macro-F1)  
5. **Evaluation:** Confusion matrix + per-class precision/recall  

### 📊 Key Results
| Model | Accuracy | Macro-F1 |
|--------|-----------|----------|
| Logistic Regression | 0.643 | 0.641 |
| Linear SVM | 0.638 | 0.633 |
| Multinomial NB | 0.629 | 0.626 |

✅ **Best Model:** Logistic Regression  
🔍 **Insights:** Confusion mainly between semantically close categories like *Politics* ↔ *World News*.

---

## 🧰 Tools & Libraries
| Category | Tools Used |
|-----------|-------------|
| Language | Python 3 |
| IDE | Jupyter / Google Colab |
| Libraries | `pandas`, `nltk`, `sklearn`, `matplotlib`, `seaborn`, `wordcloud` |
| Algorithms | Logistic Regression, Linear SVM, Multinomial NB |
| Feature Engineering | TF-IDF Vectorization |

---

## 📂 Files Included
```
NLP_PDF.pdf                                   # Combined project intro/report
IMDbSentimentAnalysisPartAPDF.pdf             # IMDb sentiment analysis report
IMDbSentimentAnalysisPartAPythonNotebook.ipynb # IMDb Jupyter notebook
Topic Modeling using LDA (NLP Project) Python Script.py # IMDb project script
NewsArticleClassificationPDF.pdf              # News article classification report
NewsArticleClassificationPythonNotebook.ipynb # News classification notebook
NewsArticleClassificationPythonScript.py      # News classification script
```

---

## 📎 How to Run
1. Open `.ipynb` in Jupyter or Google Colab.  
2. Ensure dependencies (`nltk`, `sklearn`, etc.) are installed.  
3. Run cells sequentially.  
4. Check accuracy reports and visualizations in outputs.

---

## 📚 Learning Outcomes
- Text preprocessing & feature engineering  
- TF-IDF transformation  
- Model evaluation for NLP classification  
- Visualization with word clouds & confusion matrices  

---

## 👤 Author
**Utkarsh Anand**  
Data Science PGC Course 7 Project  
Internshala Placement Guarantee Program
