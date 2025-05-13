import os
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
MODEL_PATH = "srilanka_clothing_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
DATASET_PATH = "SriLanka_Weather_Dataset.csv"

# --- Constants ---
CLOTHING_CATEGORIES = [
    "Light Cotton Outfit (Shorts+T-shirt)",
    "Breathable Work Attire (Polo+Light Pants)",
    "Rain Protection (Quick-dry clothes+Umbrella)",
    "Evening Wear (Long Sleeve+Light Pants)",
    "Highland Outfit (Light Jacket+Pants)"
]

DISTRICTS = [
    "Athurugiriya",
    "Badulla",
    "Batticaloa",
    "Bentota",
    "Beruwala",
    "Chilaw",
    "Colombo",
    "Dambulla",
    "Dehiwala-Mount Lavinia",
    "Embilipitiya",
    "Galle",
    "Gampaha",
    "Hambantota",
    "Hatton",
    "Horana",
    "Jaffna",
    "Kalmunai",
    "Kalutara",
    "Kandy",
    "Kegalle",
    "Kesbewa",
    "Kilinochchi",
    "Kolonnawa",
    "Kurunegala",
    "Mabole",
    "Maharagama",
    "Mannar",
    "Matale",
    "Matara",
    "Monaragala",
    "Moratuwa",
    "Nawalapitiya",
    "Negombo",
    "Nuwara Eliya",
    "Oruwala",
    "Panadura",
    "Polonnaruwa",
    "Pothuhera",
    "Puttalam",
    "Ratnapura",
    "Sri Jayewardenepura Kotte",
    "Talawakelle",
    "Tangalle",
    "Trincomalee",
    "Valvettithurai",
    "Vavuniya",
    "Weligama",
    "Welimada"
]

# --- Dataset Processing (Updated) ---
def load_and_preprocess_data():
    """Load and preprocess the weather dataset with clothing labels"""
    try:
        df = pd.read_csv(DATASET_PATH)

        # Basic validation
        if len(df) < 1000:
            st.warning("Dataset is too small. Generating synthetic data...")
            df = generate_realistic_dataset()

        # Convert time column to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['month'] = df['time'].dt.month
        else:
            df['month'] = np.random.randint(1, 13, size=len(df))

        # Feature engineering
        df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
        df['heat_index'] = df['apparent_temperature_mean'] - df['temperature_2m_mean']
        df['is_monsoon'] = df['month'].isin([5, 6, 7, 9, 10, 11]).astype(int)
        df['is_rainy'] = (df['precipitation_sum'] > 0.1).astype(int)
        df['is_heavy_rain'] = (df['precipitation_sum'] > 10).astype(int)

        # Calculate humidity if not available
        if 'relativehumidity_2m_mean' in df.columns:
            df['is_humid'] = (df['relativehumidity_2m_mean'] > 75).astype(int)
        else:
            df['is_humid'] = ((df['heat_index'] > 3) |
                              (df['temperature_2m_mean'] > 30)).astype(int)

        # Generate clothing labels
        df['clothing'] = df.apply(generate_tropical_clothing_label, axis=1)

        # Select relevant features
        features = [
            'temperature_2m_mean',
            'apparent_temperature_mean',
            'precipitation_sum',
            'windspeed_10m_max',
            'temp_range',
            'heat_index',
            'is_monsoon',
            'is_rainy',
            'is_heavy_rain',
            'is_humid'
        ]

        return df[features + ['clothing']]

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}. Generating synthetic data...")
        return generate_realistic_dataset()


def generate_realistic_dataset(size=10000):
    """Generate realistic synthetic weather data for Sri Lanka"""
    np.random.seed(42)

    # Base temperatures by month (Colombo-like climate)
    month_temps = {
        1: 27, 2: 28, 3: 29, 4: 30, 5: 29, 6: 28,
        7: 28, 8: 28, 9: 28, 10: 28, 11: 27, 12: 27
    }

    data = {
        'time': pd.date_range(start='2020-01-01', periods=size, freq='D'),
        'temperature_2m_mean': np.array(
            [month_temps[d.month] for d in pd.date_range(start='2020-01-01', periods=size, freq='D')])
                               + np.random.normal(0, 2, size),
        'apparent_temperature_mean': np.array(
            [month_temps[d.month] for d in pd.date_range(start='2020-01-01', periods=size, freq='D')])
                                     + np.random.normal(1, 2, size),
        'precipitation_sum': np.random.exponential(0.5, size),
        'windspeed_10m_max': np.random.weibull(1.5, size) * 10,
        'relativehumidity_2m_mean': np.random.normal(75, 10, size).clip(40, 100)
    }

    # Add highland variations (10% of data)
    highland_mask = np.random.random(size) < 0.1
    data['temperature_2m_mean'][highland_mask] -= 8
    data['apparent_temperature_mean'][highland_mask] -= 10
    data['precipitation_sum'][highland_mask] *= 1.5

    df = pd.DataFrame(data)

    # Add derived features
    df['temp_range'] = np.random.uniform(3, 8, size)
    df['heat_index'] = df['apparent_temperature_mean'] - df['temperature_2m_mean']
    df['month'] = df['time'].dt.month
    df['is_monsoon'] = df['month'].isin([5, 6, 7, 9, 10, 11]).astype(int)
    df['is_rainy'] = (df['precipitation_sum'] > 0.1).astype(int)
    df['is_heavy_rain'] = (df['precipitation_sum'] > 10).astype(int)
    df['is_humid'] = (df['relativehumidity_2m_mean'] > 75).astype(int)

    # Generate clothing labels
    df['clothing'] = df.apply(generate_tropical_clothing_label, axis=1)

    return df


def generate_noisy_label(row):
    """Generate labels with realistic noise and variations"""
    base_label = generate_tropical_clothing_label(row)

    # Add 10% chance of getting a different label (simulating real-world variation)
    if np.random.random() < 0.1:
        return np.random.choice([
            l for l in CLOTHING_CATEGORIES if l != base_label
        ])
    return base_label


def generate_tropical_clothing_label(row):
    """Generate clothing labels with more nuanced rules"""
    temp = row['temperature_2m_mean']
    rain = row['precipitation_sum']
    humid = row.get('is_humid', (row.get('relativehumidity_2m_mean', 70) > 75))

    # Rain takes priority but with some variation
    if row['is_heavy_rain']:
        return "Rain Protection (Quick-dry clothes+Umbrella)"
    elif row['is_rainy']:
        if temp > 28:
            return "Rain Protection (Quick-dry clothes+Umbrella)"
        else:
            return np.random.choice([
                "Evening Wear (Long Sleeve+Light Pants)",
                "Rain Protection (Quick-dry clothes+Umbrella)"
            ], p=[0.7, 0.3])

    # Temperature and humidity with overlap zones
    if temp >= 32:
        if humid:
            return "Light Cotton Outfit (Shorts+T-shirt)"
        return np.random.choice([
            "Light Cotton Outfit (Shorts+T-shirt)",
            "Breathable Work Attire (Polo+Light Pants)"
        ], p=[0.8, 0.2])
    elif temp >= 28:
        if humid:
            return np.random.choice([
                "Breathable Work Attire (Polo+Light Pants)",
                "Light Cotton Outfit (Shorts+T-shirt)"
            ], p=[0.7, 0.3])
        return "Breathable Work Attire (Polo+Light Pants)"
    elif temp >= 24:
        return "Evening Wear (Long Sleeve+Light Pants)"
    elif temp >= 18:
        return np.random.choice([
            "Evening Wear (Long Sleeve+Light Pants)",
            "Highland Outfit (Light Jacket+Pants)"
        ], p=[0.6, 0.4])
    else:
        return "Highland Outfit (Light Jacket+Pants)"


# --- Model Training ---
def train_model(df):
    """Train and save the clothing recommendation model with proper evaluation"""
    # Encode clothing labels
    le = LabelEncoder()
    le.fit(CLOTHING_CATEGORIES)
    y = le.transform(df['clothing'])
    X = df.drop('clothing', axis=1)

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Store test data in session state
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

    # Optimized Random Forest with regularization
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        oob_score=True
    )

    # Add cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    model.fit(X_train, y_train)

    # Evaluate model more thoroughly
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        log_loss
    )

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    loss = log_loss(y_test, y_proba)

    # Print comprehensive evaluation
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
    print(f"OOB Score: {model.oob_score_:.2f}")
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Log Loss: {loss:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    return model, le, accuracy, pd.DataFrame(report).transpose(), cv_scores


# --- Visualization ---
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(plt)


def fetch_current_weather(city_name):
    """Fetch REAL-TIME weather for Sri Lankan cities using exact coordinates"""
    city_coords = {
        "Athurugiriya": {"lat": 6.8765, "lon": 79.9550},
        "Badulla": {"lat": 6.9934, "lon": 81.0550},
        "Batticaloa": {"lat": 7.7167, "lon": 81.7000},
        "Bentota": {"lat": 6.4189, "lon": 79.9967},
        "Beruwala": {"lat": 6.4667, "lon": 79.9833},
        "Chilaw": {"lat": 7.5758, "lon": 79.7953},
        "Colombo": {"lat": 6.9271, "lon": 79.8612},
        "Dambulla": {"lat": 7.8600, "lon": 80.6517},
        "Dehiwala-Mount Lavinia": {"lat": 6.8409, "lon": 79.8716},
        "Embilipitiya": {"lat": 6.3500, "lon": 80.8500},
        "Galle": {"lat": 6.0535, "lon": 80.2210},
        "Gampaha": {"lat": 7.0917, "lon": 79.9997},
        "Hambantota": {"lat": 6.1245, "lon": 81.1186},
        "Hatton": {"lat": 6.8917, "lon": 80.5958},
        "Horana": {"lat": 6.7167, "lon": 80.0667},
        "Jaffna": {"lat": 9.6615, "lon": 80.0255},
        "Kalmunai": {"lat": 7.4167, "lon": 81.8167},
        "Kalutara": {"lat": 6.5833, "lon": 79.9667},
        "Kandy": {"lat": 7.2906, "lon": 80.6337},
        "Kegalle": {"lat": 7.2500, "lon": 80.3333},
        "Kesbewa": {"lat": 6.7958, "lon": 79.9386},
        "Kilinochchi": {"lat": 9.4000, "lon": 80.4000},
        "Kolonnawa": {"lat": 6.9167, "lon": 79.8833},
        "Kurunegala": {"lat": 7.4833, "lon": 80.3667},
        "Mabole": {"lat": 7.2500, "lon": 79.9500},
        "Maharagama": {"lat": 6.8500, "lon": 79.9167},
        "Mannar": {"lat": 8.9667, "lon": 79.9167},
        "Matale": {"lat": 7.4667, "lon": 80.6167},
        "Matara": {"lat": 5.9483, "lon": 80.5353},
        "Monaragala": {"lat": 6.8667, "lon": 81.3500},
        "Moratuwa": {"lat": 6.7806, "lon": 79.8817},
        "Nawalapitiya": {"lat": 7.0500, "lon": 80.5333},
        "Negombo": {"lat": 7.2111, "lon": 79.8386},
        "Nuwara Eliya": {"lat": 6.9708, "lon": 80.7829},
        "Oruwala": {"lat": 6.8333, "lon": 79.9667},
        "Panadura": {"lat": 6.7133, "lon": 79.9042},
        "Polonnaruwa": {"lat": 7.9333, "lon": 81.0000},
        "Pothuhera": {"lat": 7.4167, "lon": 80.3333},
        "Puttalam": {"lat": 8.0333, "lon": 79.8167},
        "Ratnapura": {"lat": 6.6806, "lon": 80.4022},
        "Sri Jayewardenepura Kotte": {"lat": 6.9108, "lon": 79.8878},
        "Talawakelle": {"lat": 6.9333, "lon": 80.6667},
        "Tangalle": {"lat": 6.0167, "lon": 80.7833},
        "Trincomalee": {"lat": 8.5833, "lon": 81.2333},
        "Valvettithurai": {"lat": 9.8167, "lon": 80.1667},
        "Vavuniya": {"lat": 8.7500, "lon": 80.5000},
        "Weligama": {"lat": 5.9667, "lon": 80.4167},
        "Welimada": {"lat": 6.9000, "lon": 80.9167}
    }

    # Get coordinates directly
    if city_name not in city_coords:
        st.error(f"Weather data not available for {city_name}")
        return None

    lat, lon = city_coords[city_name]["lat"], city_coords[city_name]["lon"]

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Debug: Print raw API response
        print(f"Weather API response for {city_name}: {data}")

        # Extract key features with proper error handling
        weather_data = {
            'temperature_2m_mean': data['main']['temp'],
            'apparent_temperature_mean': data['main']['feels_like'],
            'precipitation_sum': data.get('rain', {}).get('1h', 0) or 0,
            'windspeed_10m_max': data['wind']['speed'],
            'relativehumidity_2m_mean': data['main']['humidity'],
            'temp_range': data['main']['temp_max'] - data['main']['temp_min'],
            'heat_index': calculate_heat_index(
                data['main']['temp'],
                data['main']['humidity']
            ),
            'month': pd.Timestamp.now().month,
            'is_monsoon': 1 if pd.Timestamp.now().month in [5, 6, 7, 9, 10, 11] else 0,
            'is_rainy': 1 if 'rain' in data else 0,
            'is_heavy_rain': 1 if data.get('rain', {}).get('1h', 0) > 10 else 0,
            'is_humid': 1 if data['main']['humidity'] > 75 else 0,
        }

        # Convert wind speed to km/h only for display
        weather_data['windspeed_10m_max_kmh'] = weather_data['windspeed_10m_max'] * 3.6

        return weather_data

    except Exception as e:
        st.error(f"Weather API error: {str(e)}")
        return None

def calculate_heat_index(temp, humidity):
    """Calculate heat index (feels-like temperature) more accurately"""
    # Simplified heat index calculation
    if temp < 27 or humidity < 40:
        return temp
    else:
        return temp + 0.5 * (humidity / 100) * temp


def prepare_input_for_model(weather_data):
    """Prepare the weather data in the exact format the model expects"""
    features = [
        'temperature_2m_mean',
        'apparent_temperature_mean',
        'precipitation_sum',
        'windspeed_10m_max',
        'temp_range',
        'heat_index',
        'is_monsoon',
        'is_rainy',
        'is_heavy_rain',
        'is_humid'
    ]

    # Create DataFrame with correct feature names and order
    input_data = {
        'temperature_2m_mean': weather_data['temperature_2m_mean'],
        'apparent_temperature_mean': weather_data['apparent_temperature_mean'],
        'precipitation_sum': weather_data['precipitation_sum'],
        'windspeed_10m_max': weather_data['windspeed_10m_max'],  # In m/s
        'temp_range': weather_data['temp_range'],
        'heat_index': weather_data['heat_index'],
        'is_monsoon': weather_data['is_monsoon'],
        'is_rainy': weather_data['is_rainy'],
        'is_heavy_rain': weather_data['is_heavy_rain'],
        'is_humid': weather_data['is_humid']
    }

    return pd.DataFrame([input_data])[features]


def prepare_input_for_model(weather_data):
    """Prepare the weather data in the exact format the model expects"""
    features = [
        'temperature_2m_mean',
        'apparent_temperature_mean',
        'precipitation_sum',
        'windspeed_10m_max',
        'temp_range',
        'heat_index',
        'is_monsoon',
        'is_rainy',
        'is_heavy_rain',
        'is_humid'
    ]

    # Create DataFrame with correct feature names and order
    input_data = {
        'temperature_2m_mean': weather_data['temperature_2m_mean'],
        'apparent_temperature_mean': weather_data['apparent_temperature_mean'],
        'precipitation_sum': weather_data['precipitation_sum'],
        'windspeed_10m_max': weather_data['windspeed_10m_max'],
        'temp_range': weather_data['temp_range'],
        'heat_index': weather_data['heat_index'],
        'is_monsoon': weather_data['is_monsoon'],
        'is_rainy': weather_data['is_rainy'],
        'is_heavy_rain': weather_data['is_heavy_rain'],
        'is_humid': weather_data['is_humid']
    }

    return pd.DataFrame([input_data])[features]


# --- Streamlit App ---
def main():
    st.title("🇱🇰 Sri Lanka Smart Clothing Recommender")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.le = None
        st.session_state.accuracy = None
        st.session_state.report = None
        st.session_state.cv_scores = None
        st.session_state.X_test = None
        st.session_state.y_test = None

    # Sidebar for model management
    with st.sidebar:
        st.header("Model Management")
        if st.button("Train/Retrain Model"):
            with st.spinner("Training model with realistic variations..."):
                df = load_and_preprocess_data()
                model, le, accuracy, report, cv_scores = train_model(df)
                st.session_state.model = model
                st.session_state.le = le
                st.session_state.accuracy = accuracy
                st.session_state.report = report
                st.session_state.cv_scores = cv_scores
                st.success(f"Model trained with cross-val accuracy: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

        if st.session_state.model:
            st.subheader("Model Performance")
            st.write(f"Test Accuracy: {st.session_state.accuracy:.2%}")
            st.write(
                f"Cross-Validation Accuracy: {np.mean(st.session_state.cv_scores):.2%} ± {np.std(st.session_state.cv_scores):.2%}")

            st.write("Classification Report:")
            st.dataframe(st.session_state.report)

            if st.button("Show Confusion Matrix"):
                y_pred = st.session_state.model.predict(st.session_state.X_test)
                plot_confusion_matrix(st.session_state.y_test, y_pred, st.session_state.le.classes_)

    # Main interface
    st.subheader("Get Clothing Recommendation")
    city = st.selectbox("Select City",
                        DISTRICTS,
                        help="Select a city in Sri Lanka")

    if st.button("Get Recommendation"):
        if not st.session_state.model:
            st.warning("Please train the model first using the sidebar")
            return

        try:
            # Get current weather
            current_weather = fetch_current_weather(city)

            # Prepare input data in correct format
            input_df = prepare_input_for_model(current_weather)

            # Make prediction
            prediction = st.session_state.model.predict(input_df)
            recommendation = st.session_state.le.inverse_transform(prediction)[0]

            # Display results
            st.subheader(f"Current Weather in {city}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Temperature", f"{current_weather['temperature_2m_mean']:.1f}°C")
                st.metric("Feels Like", f"{current_weather['apparent_temperature_mean']:.1f}°C")
            with col2:
                st.metric("Precipitation", f"{current_weather['precipitation_sum']:.1f} mm")
                st.metric("Wind Speed", f"{current_weather['windspeed_10m_max']:.1f} km/h")

            st.subheader("Recommended Outfit")
            st.success(f"**{recommendation}**")

            # Show prediction probabilities
            probas = st.session_state.model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame({
                'Outfit': st.session_state.le.classes_,
                'Probability': probas
            }).sort_values('Probability', ascending=False)

            st.write("Prediction Confidence:")
            st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))

        except Exception as e:
            st.error(f"Error getting weather data: {str(e)}")
            st.error("Please make sure you have a valid OpenWeatherMap API key in your .env file")


if __name__ == "__main__":
    main()