# Sri-Lanka-Weather-Based-Clothing-Recommendation

This project focuses on training a **traditional Machine Learning model** to recommend clothing based on Sri Lanka's weather patterns.  
It uses a **Random Forest Classifier** trained on weather data, with support for synthetic data generation and feature engineering.

---

## 🎯 Purpose

- To build a **Machine Learning model** that predicts clothing recommendations for different districts in Sri Lanka using weather data.
- Demonstrates traditional ML techniques including:
  - Data preprocessing and feature engineering.
  - Synthetic data generation.
  - Model training and evaluation.
  - Saving models and encoders for deployment.

---

## ⚙ Technologies Used

- Python 3.x
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- streamlit
- joblib
- dotenv

---

## 🧪 Features

- Load and preprocess real or synthetic weather data.
- Perform feature engineering (e.g., heat index, temp range, monsoon season, etc.).
- Generate nuanced clothing labels using weather conditions.
- Train and evaluate a Random Forest model.
- Save the trained model and label encoder.
- Streamlit integration for interactive UI.

---

## 🏗 Model Pipeline Overview

1. **Data Loading & Preprocessing**
    - Load dataset or generate synthetic data if insufficient.
    - Extract features and create derived columns.

2. **Label Generation**
    - Generate clothing labels based on temperature, rainfall, humidity, etc.
    - Introduce realistic noise and variation.

3. **Model Training**
    - Encode labels.
    - Train-test split with stratification.
    - Train Random Forest Classifier.
    - Evaluate using classification report, confusion matrix, cross-validation.

4. **Model Saving**
    - Save trained model (`.pkl` file).
    - Save label encoder.

---


## 📜 License

This project is for academic and research use only.

---
