import streamlit as st
import numpy as np
import requests
from datetime import datetime
import pandas as pd
import risk_model as fp

# --- Page Config ---
st.set_page_config(
    page_title="Disaster Prediction System",
    page_icon="🌧️",
    layout="centered",
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .safe-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        color: #155724;
    }
    .moderate-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        color: #856404;
    }
    .heavy-box {
        background-color: #ffe0b2;
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        color: #e65100;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.3rem;
        color: #721c24;
    }
    .info-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- State Model ID Mapping ---
states = {
    'ANDAMAN & NICOBAR ISLANDS': 0, 'ARUNACHAL PRADESH': 1, 'ASSAM': 2,
    'MEGHALAYA': 2, 'BIHAR': 3, 'CHHATTISGARH': 4, 'ANDHRA PRADESH': 5,
    'KARNATAKA': 6, 'MADHYA PRADESH': 7, 'RAJASTHAN': 8, 'UTTAR PRADESH': 9,
    'WEST BENGAL': 10, 'GUJARAT': 11, 'HARYANA': 12, 'DELHI': 12,
    'HIMACHAL PRADESH': 13, 'JAMMU & KASHMIR': 14, 'JHARKHAND': 15,
    'KERALA': 16, 'GOA': 17, 'LAKSHADWEEP': 18, 'MADHYA MAHARASHTRA': 19,
    'MATATHWADA': 20, 'NAGALAND': 21, 'MANIPUR': 21, 'MIZORAM': 21,
    'TRIPURA': 21, 'ODISSA': 23, 'PUNJAB': 24, 'RAYALSEEMA': 25,
    'SAURASHTRA & KUTCH': 26, 'SOUTH INTERIOR KARNATAKA': 27,
    'SUB HIMALAYAN WEST BENGAL & SIKKIM': 28, 'TAMIL NADU': 29,
    'TELANGANA': 30, 'UTTARAKHAND': 31, 'VIDARBHA': 32,
    'WEST MADHYA PRADESH': 33, 'WEST RAJASTHAN': 34, 'WEST UTTAR PRADESH': 35
}

# --- State Coordinates (lat, lon) for Open-Meteo API ---
state_coords = {
    'ANDAMAN & NICOBAR ISLANDS': (11.74, 92.65),
    'ARUNACHAL PRADESH': (27.08, 93.60),
    'ASSAM': (26.20, 92.94),
    'MEGHALAYA': (25.57, 91.88),
    'BIHAR': (25.60, 85.10),
    'CHHATTISGARH': (21.27, 81.86),
    'ANDHRA PRADESH': (15.91, 79.74),
    'KARNATAKA': (15.32, 75.71),
    'MADHYA PRADESH': (22.97, 78.65),
    'RAJASTHAN': (27.02, 74.22),
    'UTTAR PRADESH': (26.85, 80.91),
    'WEST BENGAL': (22.98, 87.85),
    'GUJARAT': (22.26, 71.19),
    'HARYANA': (29.06, 76.08),
    'DELHI': (28.70, 77.10),
    'HIMACHAL PRADESH': (31.10, 77.17),
    'JAMMU & KASHMIR': (34.08, 74.80),
    'JHARKHAND': (23.61, 85.28),
    'KERALA': (10.85, 76.27),
    'GOA': (15.30, 74.12),
    'LAKSHADWEEP': (10.56, 72.63),
    'MADHYA MAHARASHTRA': (18.52, 73.85),
    'MATATHWADA': (19.57, 75.47),
    'NAGALAND': (26.15, 94.56),
    'MANIPUR': (24.82, 93.95),
    'MIZORAM': (23.36, 92.79),
    'TRIPURA': (23.94, 91.63),
    'ODISSA': (20.95, 85.09),
    'PUNJAB': (31.14, 75.34),
    'RAYALSEEMA': (14.68, 77.60),
    'SAURASHTRA & KUTCH': (22.31, 70.78),
    'SOUTH INTERIOR KARNATAKA': (12.30, 76.65),
    'SUB HIMALAYAN WEST BENGAL & SIKKIM': (27.33, 88.61),
    'TAMIL NADU': (11.13, 78.66),
    'TELANGANA': (17.12, 79.21),
    'UTTARAKHAND': (30.07, 79.02),
    'VIDARBHA': (21.09, 79.09),
    'WEST MADHYA PRADESH': (23.26, 77.41),
    'WEST RAJASTHAN': (26.45, 71.64),
    'WEST UTTAR PRADESH': (28.64, 77.22)
}


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_live_weather(lat, lon):
    """
    Fetch current weather from Open-Meteo API (free, no API key needed).
    Returns dict with rainfall, temperature, humidity, weather code, wind speed.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "daily": "precipitation_sum,precipitation_hours,precipitation_probability_max",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600)
def get_historical_rainfall(state_name):
    """
    Get historical average monthly rainfall from historical_rainfall.csv for the selected state.
    """
    try:
        df = pd.read_csv("historical_rainfall.csv")
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

        # Try to match the state name in the SUBDIVISION or STATE column
        state_upper = state_name.upper()
        state_data = None

        if 'STATE' in df.columns:
            state_data = df[df['STATE'].str.upper().str.contains(state_upper, na=False)]
        if (state_data is None or len(state_data) == 0) and 'SUBDIVISION' in df.columns:
            state_data = df[df['SUBDIVISION'].str.upper().str.contains(state_upper, na=False)]

        if state_data is not None and len(state_data) > 0:
            current_month = datetime.now().month
            month_col = months[current_month - 1]
            if month_col in df.columns:
                avg_rain = state_data[month_col].mean()
                if pd.notna(avg_rain):
                    return round(avg_rain, 1)
        return None
    except Exception:
        return None


# --- Header ---
st.markdown('<p class="main-header">🌧️ Disaster Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Flood & Drought Risk Analysis with Live Weather Data</p>', unsafe_allow_html=True)

st.markdown("---")

# --- Sidebar ---
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app predicts flood/drought risk using **live weather data** "
    "from the Open-Meteo API combined with an ML model trained on "
    "50 years of IMD rainfall data (1951-2000)."
)

st.sidebar.title("🚦 Risk Levels")
st.sidebar.markdown("""
- 🟢 **Safe** - No risk
- 🟡 **Moderate** - Light rain, stay prepared
- 🟠 **Heavy** - Flood chances increasing
- 🔴 **Extreme** - Stay indoors
""")

st.sidebar.title("📡 Data Sources")
st.sidebar.markdown("""
- **Live Weather:** [Open-Meteo API](https://open-meteo.com/) (free, no key needed)
- **Historical Data:** Indian Govt Open Data Portal (data.gov.in)
- **ML Model:** Logistic Regression (scikit-learn)
""")

# --- Main Content ---
selected_state = st.selectbox(
    "📍 Select your State / Region",
    sorted(states.keys()),
    index=sorted(states.keys()).index('DELHI')
)

# Fetch live weather data
lat, lon = state_coords.get(selected_state, (28.70, 77.10))

st.markdown("---")

# --- Live Weather Section ---
st.subheader("📡 Live Weather Data")

with st.spinner(f"Fetching live weather for {selected_state}..."):
    weather = fetch_live_weather(lat, lon)

rainfall_mm = 0.0
live_data_available = False

if "error" not in weather and "current" in weather:
    live_data_available = True
    current = weather["current"]
    daily = weather.get("daily", {})

    live_rain = current.get("precipitation", 0.0)
    temp = current.get("temperature_2m", "N/A")
    humidity = current.get("relative_humidity_2m", "N/A")
    wind = current.get("wind_speed_10m", "N/A")

    # Show current weather in columns
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🌧️ Current Rain", f"{live_rain} mm")
    with c2:
        st.metric("🌡️ Temperature", f"{temp}°C")
    with c3:
        st.metric("💧 Humidity", f"{humidity}%")
    with c4:
        st.metric("💨 Wind Speed", f"{wind} km/h")

    # 7-day forecast
    if "time" in daily and "precipitation_sum" in daily:
        st.markdown("**📅 7-Day Rainfall Forecast:**")
        forecast_df = pd.DataFrame({
            "Date": daily["time"],
            "Rainfall (mm)": daily["precipitation_sum"],
            "Rain Hours": daily.get("precipitation_hours", ["N/A"] * len(daily["time"])),
            "Rain Prob %": daily.get("precipitation_probability_max", ["N/A"] * len(daily["time"])),
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    # Use today's total precipitation as the rainfall input
    if "time" in daily and "precipitation_sum" in daily and len(daily["precipitation_sum"]) > 0:
        today_rain = daily["precipitation_sum"][0]
        rainfall_mm = float(today_rain) if today_rain is not None else 0.0
    else:
        rainfall_mm = float(live_rain) if live_rain else 0.0

    st.success(f"✅ Live data fetched at {datetime.now().strftime('%I:%M %p')} | Today's rainfall: **{rainfall_mm} mm**")
else:
    st.warning(
        f"⚠️ Could not fetch live weather ({weather.get('error', 'Unknown error')}). "
        "Falling back to historical data."
    )

# Historical fallback
historical_rain = get_historical_rainfall(selected_state)
if historical_rain is not None:
    st.info(f"📊 Historical average rainfall for {selected_state} this month: **{historical_rain} mm**")

# If live data failed, use historical or let user enter manually
if not live_data_available:
    st.markdown("---")
    st.subheader("✋ Manual Entry (fallback)")
    default_val = historical_rain if historical_rain is not None else 150.0
    rainfall_mm = st.number_input(
        "🌧️ Enter Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=default_val,
        step=10.0,
        help="Enter current rainfall in mm since live data is unavailable"
    )
else:
    # Allow manual override even when live data is available
    with st.expander("✏️ Override with manual rainfall value"):
        rainfall_mm = st.number_input(
            "🌧️ Manual Rainfall (mm)",
            min_value=0.0,
            max_value=1000.0,
            value=rainfall_mm,
            step=10.0,
            help="Override the live rainfall data with your own value"
        )

st.markdown("---")

# --- Predict Button ---
if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
    with st.spinner("Running ML prediction model..."):
        state_id = states[selected_state]
        prediction = fp.prediction1([[state_id, rainfall_mm]])
        result = int(prediction[0]) if hasattr(prediction, '__len__') else int(prediction)

    st.markdown(f"**Region:** {selected_state}  |  **Rainfall:** {rainfall_mm} mm  |  **State ID:** {state_id}")
    st.markdown("")

    if result == 0:
        st.markdown(
            '<div class="safe-box">🟢 <b>You are completely safe!</b><br>'
            'Current rainfall levels are normal. No risk of flooding.</div>',
            unsafe_allow_html=True
        )
    elif result == 1:
        st.markdown(
            '<div class="moderate-box">🟡 <b>Moderate Rainfall</b><br>'
            'Keep your umbrella with you, but you\'re safe. Stay updated with weather reports.</div>',
            unsafe_allow_html=True
        )
    elif result == 2:
        st.markdown(
            '<div class="heavy-box">🟠 <b>Heavy Rainfall Alert!</b><br>'
            'Chances of floods are increasing. Please take necessary precautions. '
            'Avoid low-lying areas.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="danger-box">🔴 <b>Extreme Flood Risk!</b><br>'
            'Flood chances are at peak. Stay in your house. '
            'Follow emergency guidelines from local authorities.</div>',
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown("---")
st.caption(
    "⚠️ This is a demonstration project. Do not rely solely on this app for "
    "emergency decisions. Always follow official weather advisories from the "
    "Indian Meteorological Department (IMD)."
)
st.caption(
    "📡 Live weather: Open-Meteo API | "
    "📊 Historical data: data.gov.in | "
    "🤖 Model: Logistic Regression (scikit-learn)"
)
