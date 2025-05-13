# Sri Lanka Weather-Based Clothing Recommendation System

This project is a **Traditional Machine Learning-based Clothing Recommendation System** that suggests appropriate clothing based on **Sri Lankan weather conditions**. It uses a **Random Forest Classifier** trained on weather data to predict clothing recommendations for different climatic scenarios across Sri Lanka.

---

## Project Overview

This system recommends clothing based on current or forecasted weather conditions in Sri Lanka using **Traditional Machine Learning** techniques. The model predicts one of several predefined clothing categories (e.g., Light Cotton Outfit, Rain Protection) considering weather attributes like temperature, precipitation, humidity, windspeed, and seasonality.

The system is built using:
- **Python (pandas, scikit-learn, numpy, seaborn, matplotlib)**
- **Streamlit for the user interface**
- **Joblib for model persistence**
- **OpenWeather API for real-time data (optional integration)**

---

## Technologies Used

- Python 
- pandas
- numpy
- scikit-learn (Random Forest Classifier, LabelEncoder, cross-validation)
- matplotlib & seaborn (visualizations)
- streamlit (web application)
- joblib (model persistence)
- dotenv (environment variable handling)
- OpenWeather API (live weather integration)

---

## Features

- Weather-based clothing recommendation
- Sri Lanka-specific weather data
- Synthetic data generation if dataset is missing or small
- Model training with evaluation (Accuracy, Confusion Matrix, Classification Report)
- Streamlit web UI to interact with the model and get clothing recommendations

---

## Dataset

The system uses a weather dataset (`SriLanka_Weather_Dataset.csv`) containing historical weather data for Sri Lanka. The dataset includes:
- Temperature (Mean, Min, Max)
- Apparent temperature
- Precipitation
- Windspeed
- Humidity
- Month (derived from time column)

If the dataset is missing or too small, the system generates **realistic synthetic data** simulating Sri Lanka's climate.

Link: https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset

---

## Model

- **Algorithm:** Random Forest Classifier (Traditional ML)
- **Features:** Engineered from weather data (e.g., temperature range, heat index, is rainy, is humid)
- **Target:** Clothing category (5 predefined categories)
- **Evaluation:** Accuracy, Classification Report, Confusion Matrix, Cross-Validation Score

---

## 📜 License

This project is for academic and research use only.

---
