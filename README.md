# Disaster Prediction System

A web application that predicts flood/drought risk for Indian states based on rainfall data, using live weather from the Open-Meteo API and a Machine Learning model trained on 50 years of IMD rainfall data (1951-2000).

## Features

- **Live Weather Data** - Fetches real-time rainfall, temperature, humidity, and wind speed from [Open-Meteo API](https://open-meteo.com/) (free, no API key needed)
- **7-Day Forecast** - Shows upcoming rainfall predictions with probability
- **Historical Comparison** - Compares live data against 50-year historical averages
- **ML Prediction** - Logistic Regression model classifies risk into 4 levels (Safe, Moderate, Heavy, Extreme)

## Risk Levels

| Level | Meaning |
|-------|---------|
| Safe | Normal rainfall, no risk |
| Moderate | Light rain, stay prepared |
| Heavy | Flood chances increasing |
| Extreme | Stay indoors, follow emergency guidelines |

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Project Structure

```
Disaster-prediction/
├── app.py                    # Main Streamlit web app
├── risk_model.py             # ML model (Logistic Regression)
├── training_data.csv         # Labeled training dataset
├── historical_rainfall.csv   # 50-year rainfall data from IMD
├── requirements.txt          # Python dependencies
└── venv/                     # Virtual environment
```
link to app - https://disasterrisk-ai.streamlit.app/
## How It Works

1. **Training data** (`training_data.csv`) was created by labeling historical rainfall with risk levels based on thresholds (0-204mm = safe, 204-254mm = moderate, 254-354mm = heavy, 354+mm = extreme)
2. **Logistic Regression** model learns the relationship between state + rainfall and risk level
3. **Live weather** is fetched from Open-Meteo API based on selected state coordinates
4. The model predicts risk using the current rainfall data

## Data Sources

- [Indian Govt Open Data Portal](https://data.gov.in/) - Historical rainfall data (1951-2000)
- [Open-Meteo API](https://open-meteo.com/) - Live weather data
