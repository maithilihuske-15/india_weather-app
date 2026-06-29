# -*- coding: utf-8 -*-
"""
weather_app.py
Hybrid ML Weather Forecasting — GRP17
Run with: streamlit run weather_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import requests
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import time
from datetime import datetime
from timezonefinder import TimezoneFinder
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh



random.seed(42)
np.random.seed(42)
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

MODEL_DIR       = "saved_models"
INDIA_MODEL_DIR = "saved_models_india"

# Set of city-display names that should use the India daily model
INDIA_CITY_NAMES = {
    "Mumbai, India", "Delhi, India", "Pune, India",
    "Bangalore, India", "Baramati, India",
    "Hyderabad, India", "Chennai, India",
    "Kolkata, India", "Ahmedabad, India", "Nagpur, India"
}

CITY_OPTIONS = {
    "Mumbai, India":      ("Mumbai", "IN"),
    "Delhi, India":       ("Delhi", "IN"),
    "Pune, India":        ("Pune", "IN"),
    "Bangalore, India":   ("Bangalore", "IN"),
    "Baramati, India":    ("Baramati", "IN"),
    "Hyderabad, India":   ("Hyderabad", "IN"),
    "Chennai, India":     ("Chennai", "IN"),
    "Kolkata, India":     ("Kolkata", "IN"),
    "Ahmedabad, India":   ("Ahmedabad", "IN"),
    "Nagpur, India":      ("Nagpur", "IN"),
    "Enter Custom City":  ("", "IN"),
}

WEATHER_ICONS = {
    "Rainy":  "🌧️",
    "Cloudy": "☁️",
    "Sunny":  "☀️",
    "Cold":   "🥶",
    "Normal": "🌤️",
}

# ══════════════════════════════════════════════════════════
# ── HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def get_city_time(city_name, country_code):
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?q={city_name},{country_code}&appid={API_KEY}&units=metric")
        resp    = requests.get(url, timeout=5).json()
        lat     = resp["coord"]["lat"]
        lon     = resp["coord"]["lon"]
        tf      = TimezoneFinder()
        tz_name = tf.timezone_at(lat=lat, lng=lon)
        tz      = pytz.timezone(tz_name)
        return datetime.now(tz), tz_name
    except Exception:
        return datetime.now(pytz.UTC), "UTC"


def generate_smart_forecast(X_input_original, models, feature_cols,
                             start_hour, current_temp, city_name, country_code):
    feat = list(feature_cols)
    def idx(name): return feat.index(name) if name in feat else None
    T_idx=idx("T"); rh_idx=idx("rh"); rain_idx=idx("rain"); wv_idx=idx("wv"); hour_idx=idx("hour")
    base_T = current_temp
    X_df = pd.DataFrame(X_input_original, columns=feature_cols)
    base_preds = []
    for model in [models["xgboost"], models["lgbm"], models["randomforest"]]:
        try: base_preds.append(model.predict(X_df))
        except: base_preds.append(np.array([[current_temp, 60.0, 0.0, 3.0]]))
    meta_input = np.hstack(base_preds)
    base_pred  = models["meta_learner"].predict(meta_input)[0]
    base_rh=float(base_pred[1]); base_rain=max(0.0,float(base_pred[2])); base_wv=float(base_pred[3])
    temps,humidities,rains,winds,time_labels,hours_ahead=[],[],[],[],[],[]
    current_input = X_input_original.copy()
    city_tz_name = get_city_time(city_name, country_code)[1]
    city_tz = pytz.timezone(city_tz_name)
    city_now = datetime.now(city_tz).replace(minute=0, second=0, microsecond=0)
    for i in range(25):
        hour = (start_hour + i) % 24
        time_labels.append(city_now.strftime("%H:%M"))
        hours_ahead.append(i)
        diurnal_amplitude=5.0; diurnal_offset=diurnal_amplitude*np.sin((hour-8)*np.pi/12)
        if i == 0:
            T,rh,rain,wv = current_temp,base_rh,base_rain,base_wv
        else:
            step_preds=[]
            curr_df=pd.DataFrame(current_input,columns=feature_cols)
            for model in [models["xgboost"],models["lgbm"],models["randomforest"]]:
                try: step_preds.append(model.predict(curr_df))
                except: step_preds.append(np.array([[base_T,base_rh,base_rain,base_wv]]))
            meta_in=np.hstack(step_preds); step_pred=models["meta_learner"].predict(meta_in)[0]
            ml_T=float(step_pred[0]); ml_rh=float(step_pred[1])
            ml_rain=max(0.0,float(step_pred[2])); ml_wv=float(step_pred[3])
            prev_diurnal=diurnal_amplitude*np.sin(((hour-1)%24-8)*np.pi/12)
            diurnal_change=diurnal_offset-prev_diurnal; ml_delta=(ml_T-base_T)*0.1
            T=temps[-1]+diurnal_change+ml_delta+np.random.normal(0,0.1)
            rh=base_rh-(T-base_T)*2+np.random.normal(0,0.5); rh=min(100,max(20,rh))
            rain=max(0.0,ml_rain+np.random.normal(0,0.02))
            wv=base_wv+0.5*np.sin((hour/12)*np.pi)+np.random.normal(0,0.1); wv=max(0,wv)
        temps.append(round(T,2)); humidities.append(round(rh,1))
        rains.append(round(rain,3)); winds.append(round(max(0,wv),2))
        next_input=current_input.copy()
        if T_idx is not None: next_input[0][T_idx]=T
        if rh_idx is not None: next_input[0][rh_idx]=rh
        if rain_idx is not None: next_input[0][rain_idx]=rain
        if wv_idx is not None: next_input[0][wv_idx]=wv
        if hour_idx is not None: next_input[0][hour_idx]=(hour+1)%24
        next_input+=np.random.normal(0,0.005,next_input.shape); current_input=next_input
        city_now+=pd.Timedelta(hours=1)
    return pd.DataFrame({"Time":time_labels,"Hours Ahead":hours_ahead,
        "Temperature (°C)":temps,"Humidity (%)":humidities,"Rain (mm)":rains,"Wind (m/s)":winds})


def generate_lgbm_forecast(X_input_original, models, feature_cols,
                            start_hour, current_temp, city_name, country_code):
    feat=list(feature_cols)
    def idx(name): return feat.index(name) if name in feat else None
    T_idx=idx("T"); rh_idx=idx("rh"); rain_idx=idx("rain"); wv_idx=idx("wv"); hour_idx=idx("hour")
    base_T=current_temp
    X_df=pd.DataFrame(X_input_original,columns=feature_cols)
    try:
        base_pred=models["lgbm"].predict(X_df)[0]
        base_rh=float(base_pred[1]); base_rain=max(0.0,float(base_pred[2])); base_wv=float(base_pred[3])
    except: base_rh,base_rain,base_wv=60.0,0.0,3.0
    temps,humidities,rains,winds,time_labels,hours_ahead=[],[],[],[],[],[]
    current_input=X_input_original.copy()
    city_tz_name=get_city_time(city_name,country_code)[1]
    city_tz=pytz.timezone(city_tz_name)
    city_now=datetime.now(city_tz).replace(minute=0,second=0,microsecond=0)
    for i in range(25):
        hour=(start_hour+i)%24; time_labels.append(city_now.strftime("%H:%M")); hours_ahead.append(i)
        diurnal_amplitude=4.5; diurnal_offset=diurnal_amplitude*np.sin((hour-5)*np.pi/12)
        if i==0: T,rh,rain,wv=current_temp,base_rh,base_rain,base_wv
        else:
            try:
                curr_df=pd.DataFrame(current_input,columns=feature_cols)
                pred=models["lgbm"].predict(curr_df)[0]
                ml_T=float(pred[0]); ml_rh=float(pred[1]); ml_rain=max(0.0,float(pred[2])); ml_wv=float(pred[3])
            except: ml_T,ml_rh,ml_rain,ml_wv=base_T,base_rh,0.0,3.0
            prev_diurnal=diurnal_amplitude*np.sin(((hour-1)%24-8)*np.pi/12)
            diurnal_change=diurnal_offset-prev_diurnal; ml_delta=(ml_T-base_T)*0.1
            T=temps[-1]+diurnal_change+ml_delta+np.random.normal(0,0.1)
            rh=base_rh-(T-base_T)*2+np.random.normal(0,0.5); rh=min(100,max(20,rh))
            rain=max(0.0,ml_rain+np.random.normal(0,0.02))
            wv=base_wv+0.4*np.sin((hour/12)*np.pi)+np.random.normal(0,0.1); wv=max(0,wv)
        temps.append(round(T,2)); humidities.append(round(rh,1))
        rains.append(round(rain,3)); winds.append(round(max(0,wv),2))
        next_input=current_input.copy()
        if T_idx is not None: next_input[0][T_idx]=T
        if rh_idx is not None: next_input[0][rh_idx]=rh
        if rain_idx is not None: next_input[0][rain_idx]=rain
        if wv_idx is not None: next_input[0][wv_idx]=wv
        if hour_idx is not None: next_input[0][hour_idx]=(hour+1)%24
        next_input+=np.random.normal(0,0.005,next_input.shape); current_input=next_input
        city_now+=pd.Timedelta(hours=1)
    return pd.DataFrame({"Time":time_labels,"Hours Ahead":hours_ahead,
        "Temperature (°C)":temps,"Humidity (%)":humidities,"Rain (mm)":rains,"Wind (m/s)":winds})


def generate_xgboost_forecast(X_input_original, models, feature_cols,
                               start_hour, current_temp, city_name, country_code):
    feat=list(feature_cols)
    def idx(name): return feat.index(name) if name in feat else None
    T_idx=idx("T"); rh_idx=idx("rh"); rain_idx=idx("rain"); wv_idx=idx("wv"); hour_idx=idx("hour")
    base_T=current_temp
    X_df=pd.DataFrame(X_input_original,columns=feature_cols)
    try:
        base_pred=models["xgboost"].predict(X_df)[0]
        base_rh=float(base_pred[1]); base_rain=max(0.0,float(base_pred[2])); base_wv=float(base_pred[3])
    except: base_rh,base_rain,base_wv=60.0,0.0,3.0
    temps,humidities,rains,winds,time_labels,hours_ahead=[],[],[],[],[],[]
    current_input=X_input_original.copy()
    city_tz_name=get_city_time(city_name,country_code)[1]
    city_tz=pytz.timezone(city_tz_name)
    city_now=datetime.now(city_tz).replace(minute=0,second=0,microsecond=0)
    for i in range(25):
        hour=(start_hour+i)%24; time_labels.append(city_now.strftime("%H:%M")); hours_ahead.append(i)
        diurnal_amplitude=5.0; diurnal_offset=diurnal_amplitude*np.sin((hour-8)*np.pi/12)
        if i==0: T,rh,rain,wv=current_temp,base_rh,base_rain,base_wv
        else:
            try:
                curr_df=pd.DataFrame(current_input,columns=feature_cols)
                pred=models["xgboost"].predict(curr_df)[0]
                ml_T=float(pred[0]); ml_rh=float(pred[1]); ml_rain=max(0.0,float(pred[2])); ml_wv=float(pred[3])
            except: ml_T,ml_rh,ml_rain,ml_wv=base_T,base_rh,0.0,3.0
            prev_diurnal=diurnal_amplitude*np.sin(((hour-1)%24-8)*np.pi/12)
            diurnal_change=diurnal_offset-prev_diurnal; ml_delta=(ml_T-base_T)*0.1
            T=temps[-1]+diurnal_change+ml_delta+np.random.normal(0,0.1)
            rh=base_rh-(T-base_T)*2+np.random.normal(0,0.5); rh=min(100,max(20,rh))
            rain=max(0.0,ml_rain+np.random.normal(0,0.02))
            wv=base_wv+0.5*np.sin((hour/12)*np.pi)+np.random.normal(0,0.1); wv=max(0,wv)
        temps.append(round(T,2)); humidities.append(round(rh,1))
        rains.append(round(rain,3)); winds.append(round(max(0,wv),2))
        next_input=current_input.copy()
        if T_idx is not None: next_input[0][T_idx]=T
        if rh_idx is not None: next_input[0][rh_idx]=rh
        if rain_idx is not None: next_input[0][rain_idx]=rain
        if wv_idx is not None: next_input[0][wv_idx]=wv
        if hour_idx is not None: next_input[0][hour_idx]=(hour+1)%24
        next_input+=np.random.normal(0,0.005,next_input.shape); current_input=next_input
        city_now+=pd.Timedelta(hours=1)
    return pd.DataFrame({"Time":time_labels,"Hours Ahead":hours_ahead,
        "Temperature (°C)":temps,"Humidity (%)":humidities,"Rain (mm)":rains,"Wind (m/s)":winds})


def generate_rf_forecast(X_input_original, models, feature_cols,
                          start_hour, current_temp, city_name, country_code):
    feat=list(feature_cols)
    def idx(name): return feat.index(name) if name in feat else None
    T_idx=idx("T"); rh_idx=idx("rh"); rain_idx=idx("rain"); wv_idx=idx("wv"); hour_idx=idx("hour")
    base_T=current_temp
    X_df=pd.DataFrame(X_input_original,columns=feature_cols)
    try:
        base_pred=models["randomforest"].predict(X_df)[0]
        base_rh=float(base_pred[1]); base_rain=max(0.0,float(base_pred[2])); base_wv=float(base_pred[3])
    except: base_rh,base_rain,base_wv=60.0,0.0,3.0
    temps,humidities,rains,winds,time_labels,hours_ahead=[],[],[],[],[],[]
    current_input=X_input_original.copy()
    city_tz_name=get_city_time(city_name,country_code)[1]
    city_tz=pytz.timezone(city_tz_name)
    city_now=datetime.now(city_tz).replace(minute=0,second=0,microsecond=0)
    for i in range(25):
        hour=(start_hour+i)%24; time_labels.append(city_now.strftime("%H:%M")); hours_ahead.append(i)
        diurnal_amplitude=5.0; diurnal_offset=diurnal_amplitude*np.sin((hour-8)*np.pi/12)
        if i==0: T,rh,rain,wv=current_temp,base_rh,base_rain,base_wv
        else:
            try:
                curr_df=pd.DataFrame(current_input,columns=feature_cols)
                pred=models["randomforest"].predict(curr_df)[0]
                ml_T=float(pred[0]); ml_rh=float(pred[1]); ml_rain=max(0.0,float(pred[2])); ml_wv=float(pred[3])
            except: ml_T,ml_rh,ml_rain,ml_wv=base_T,base_rh,0.0,3.0
            prev_diurnal=diurnal_amplitude*np.sin(((hour-1)%24-8)*np.pi/12)
            diurnal_change=diurnal_offset-prev_diurnal; ml_delta=(ml_T-base_T)*0.1
            T=temps[-1]+diurnal_change+ml_delta+np.random.normal(0,0.1)
            rh=base_rh-(T-base_T)*2+np.random.normal(0,0.5); rh=min(100,max(20,rh))
            rain=max(0.0,ml_rain+np.random.normal(0,0.02))
            wv=base_wv+0.5*np.sin((hour/12)*np.pi)+np.random.normal(0,0.1); wv=max(0,wv)
        temps.append(round(T,2)); humidities.append(round(rh,1))
        rains.append(round(rain,3)); winds.append(round(max(0,wv),2))
        next_input=current_input.copy()
        if T_idx is not None: next_input[0][T_idx]=T
        if rh_idx is not None: next_input[0][rh_idx]=rh
        if rain_idx is not None: next_input[0][rain_idx]=rain
        if wv_idx is not None: next_input[0][wv_idx]=wv
        if hour_idx is not None: next_input[0][hour_idx]=(hour+1)%24
        next_input+=np.random.normal(0,0.005,next_input.shape); current_input=next_input
        city_now+=pd.Timedelta(hours=1)
    return pd.DataFrame({"Time":time_labels,"Hours Ahead":hours_ahead,
        "Temperature (°C)":temps,"Humidity (%)":humidities,"Rain (mm)":rains,"Wind (m/s)":winds})


@st.cache_data(ttl=1800)
def cached_forecast(X_input_bytes, start_hour, current_temp,
                    model_dir, city_name, country_code, selected_model):
    X_input = np.frombuffer(X_input_bytes).reshape(1, -1)
    models_cached, _ = load_models()
    feature_cols = models_cached["features"]
    forecast = generate_smart_forecast(X_input, models_cached, feature_cols,
                                       start_hour, current_temp, city_name, country_code)
    lgbm = generate_lgbm_forecast(X_input, models_cached, feature_cols,
                                   start_hour, current_temp, city_name, country_code)
    xgb_forecast = generate_xgboost_forecast(X_input, models_cached, feature_cols,
                                              start_hour, current_temp, city_name, country_code)
    rf_forecast = generate_rf_forecast(X_input, models_cached, feature_cols,
                                        start_hour, current_temp, city_name, country_code)
    return forecast, lgbm, xgb_forecast, rf_forecast


# ══════════════════════════════════════════════════════════
# ── OTHER FUNCTIONS
# ══════════════════════════════════════════════════════════

def generate_alerts(temp, humidity, rain, wind):
    alerts = []
    if rain > 3:   alerts.append("🌧️ Heavy Rain — Carry Umbrella")
    if wind > 10:  alerts.append("⛈️ Storm Risk — Stay Indoors")
    if temp > 38:  alerts.append("🌡️ Heatwave Warning")
    if temp < 8:   alerts.append("🥶 Cold Wave Alert")
    return alerts


def generate_shap_plot(explainer, X_input, feature_cols):
    try:
        shap_vals = explainer(X_input)
        fig, ax   = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
        plt.title("Why did the model predict this temperature?", fontsize=11, pad=10)
        plt.tight_layout()
        return fig
    except Exception:
        fig, ax      = plt.subplots(figsize=(9, 5))
        top_features = feature_cols[:12]
        values       = X_input[0][:12]
        colors       = ["#ef4444" if v > 0 else "#3b82f6" for v in values]
        ax.barh(top_features, values, color=colors)
        ax.set_title("Top Feature Values (SHAP unavailable)", fontsize=11)
        ax.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        return fig


@st.cache_resource
def load_models():
    try:
        models = {
            "xgboost":      joblib.load(f"{MODEL_DIR}/xgboost_model.pkl"),
            "lgbm":         joblib.load(f"{MODEL_DIR}/lgbm_model.pkl"),
            "catboost":     joblib.load(f"{MODEL_DIR}/catboost_model.pkl"),
            "randomforest": joblib.load(f"{MODEL_DIR}/randomforest_model.pkl"),
            "linear":       joblib.load(f"{MODEL_DIR}/linear_model.pkl"),
            "meta_learner": joblib.load(f"{MODEL_DIR}/meta_learner.pkl"),
            "scaler":       joblib.load(f"{MODEL_DIR}/scaler.pkl"),
            "features":     joblib.load(f"{MODEL_DIR}/feature_columns.pkl"),
        }
        if os.path.exists(f"{MODEL_DIR}/shap_explainer.pkl"):
            models["shap_explainer"] = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
        return models, None
    except FileNotFoundError as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════
# ── INDIA MODEL HELPERS  (daily next-day prediction)
# ══════════════════════════════════════════════════════════

@st.cache_resource
def load_india_models():
    """Load saved_models_india/ — cached for session."""
    if not os.path.exists(INDIA_MODEL_DIR):
        return None, f"Folder '{INDIA_MODEL_DIR}' not found. Run India notebook first."
    try:
        m = {
            "xgboost":      joblib.load(f"{INDIA_MODEL_DIR}/xgboost_model.pkl"),
            "lgbm":         joblib.load(f"{INDIA_MODEL_DIR}/lgbm_model.pkl"),
            "catboost":     joblib.load(f"{INDIA_MODEL_DIR}/catboost_model.pkl"),
            "randomforest": joblib.load(f"{INDIA_MODEL_DIR}/randomforest_model.pkl"),
            "linear":       joblib.load(f"{INDIA_MODEL_DIR}/linear_model.pkl"),
            "meta_learner": joblib.load(f"{INDIA_MODEL_DIR}/meta_learner.pkl"),
            "scaler":       joblib.load(f"{INDIA_MODEL_DIR}/scaler.pkl"),
            "feature_cols": joblib.load(f"{INDIA_MODEL_DIR}/feature_columns.pkl"),
        }
        return m, None
    except Exception as e:
        return None, str(e)


def build_india_row(api_data):
    """Map OpenWeatherMap JSON to India model feature dict."""
    dt          = datetime.utcfromtimestamp(api_data["dt"])
    month_num   = dt.month
    day_of_year = dt.timetuple().tm_yday
    return {
        "avg_temp":            api_data["main"]["temp"],
        "min_temp":            api_data["main"]["temp_min"],
        "max_temp":            api_data["main"]["temp_max"],
        "wind_speed":          api_data["wind"]["speed"],
        "air_pressure":        api_data["main"]["pressure"],
        "elevation":           200,
        "latitude":            api_data["coord"]["lat"],
        "longitude":           api_data["coord"]["lon"],
        "rainfall":            api_data.get("rain", {}).get("1h", 0.0),
        "year":                dt.year,
        "month_num":           month_num,
        "day":                 dt.day,
        "day_of_week":         dt.weekday(),
        "day_of_year":         day_of_year,
        "month_sin":           np.sin(2 * np.pi * month_num / 12),
        "month_cos":           np.cos(2 * np.pi * month_num / 12),
        "doy_sin":             np.sin(2 * np.pi * day_of_year / 365),
        "doy_cos":             np.cos(2 * np.pi * day_of_year / 365),
        "season_Monsoon":      1 if 6 <= month_num <= 9 else 0,
        "season_Post-Monsoon": 1 if month_num in [10, 11] else 0,
        "season_Summer":       1 if month_num in [3, 4, 5] else 0,
        "season_Winter":       1 if month_num in [12, 1, 2] else 0,
    }


def predict_india(city_name, country_code, india_m):
    """Fetch live weather + run India hybrid stacking. Returns (current_dict, pred_dict, error)."""
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?q={city_name},{country_code}&appid={API_KEY}&units=metric")
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if resp.status_code != 200:
            return None, None, data.get("message", "API error")
    except Exception as e:
        return None, None, str(e)

    row       = build_india_row(data)
    feat_cols = india_m["feature_cols"]

    st.write("Feature columns loaded from feature_columns.pkl:")
    st.write(india_m["feature_cols"])

    st.write("Number of features:")
    st.write(len(india_m["feature_cols"]))

    rt_df     = pd.DataFrame([row])
    for col in feat_cols:
        if col not in rt_df.columns:
            rt_df[col] = 0.0
    X_rt = rt_df[feat_cols].values

    st.write("Feature values before scaling:")
    for name, value in zip(feat_cols, X_rt[0]):
        st.write(f"{name}: {value}")
        
    X_sc = india_m["scaler"].transform(X_rt)

    st.write("Feature values after scaling:")
    for name, value in zip(feat_cols, X_sc[0]):
        st.write(f"{name}: {value}")
# ===== DEBUG =====
    import os

    st.write("Current folder:", os.getcwd())

    st.write("Feature row:")
    st.write(rt_df.T)

    st.write("Feature order:")
    st.write(feat_cols)

    st.write("Scaled values:")
    st.write(X_sc)

    p_xgb = india_m["xgboost"].predict(X_sc)
    p_lgbm = india_m["lgbm"].predict(X_sc)
    p_rf = india_m["randomforest"].predict(X_sc)

    st.write("XGBoost Prediction:", p_xgb)
    st.write("LightGBM Prediction:", p_lgbm)
    st.write("RandomForest Prediction:", p_rf)

    meta = np.hstack([p_xgb, p_lgbm, p_rf])

    st.write("Meta Input:")
    st.write(meta)

    pred = india_m["meta_learner"].predict(meta)[0]

    st.write("Final Prediction:")  
    st.write(pred)
# ==================

    current = {
        "temp":        data["main"]["temp"],
        "humidity":    data["main"]["humidity"],
        "pressure":    data["main"]["pressure"],
        "wind_speed":  data["wind"]["speed"],
        "rain_1h":     data.get("rain", {}).get("1h", 0.0),
        "description": data["weather"][0].get("description", "").title(),
        "city":        city_name,
        "country":     country_code,
    }
    prediction = {
        "avg_temp":     round(float(pred[0]), 1),
        "wind_speed":   round(float(pred[1]), 2),
        "rainfall":     round(max(0.0, float(pred[2])), 2),
        "air_pressure": round(float(pred[3]), 1),
    }
    return current, prediction, None

def india_condition(avg_temp, rainfall, wind_speed):
    if rainfall > 5:                     return "Rainy",  "🌧️"
    if rainfall > 0.5:                   return "Drizzle","🌦️"
    if avg_temp > 35:                    return "Hot",    "🌡️"
    if avg_temp > 25 and wind_speed < 3: return "Sunny",  "☀️"
    if avg_temp < 10:                    return "Cold",   "🥶"
    return "Normal", "🌤️"


def fetch_weather(city, country):
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?q={city},{country}&appid={API_KEY}&units=metric")
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if resp.status_code != 200:
            return None, data.get("message", "API error")
        temp       = data["main"].get("temp", 20.0)
        humidity   = data["main"].get("humidity", 60.0)
        pressure   = data["main"].get("pressure", 1013.0)
        wind_speed = data["wind"].get("speed", 3.0)
        wind_deg   = data["wind"].get("deg", 180.0)
        rain_1h    = data.get("rain", {}).get("1h", 0.0)
        description = data["weather"][0].get("description", "").title()
        Tdew  = temp - ((100 - humidity) / 5)
        VPmax = 6.11 * np.exp((17.62 * temp) / (243.12 + temp))
        VPact = (humidity / 100.0) * VPmax
        VPdef = VPmax - VPact
        sh    = 0.622 * VPact / (pressure - VPact) if pressure else 0
        H2OC  = VPact * 100
        rho   = pressure / (287.05 * (temp + 273.15))
        weather_data = {
            "p": pressure, "T": temp, "Tpot": temp + 1.5,
            "Tdew": Tdew, "rh": humidity, "VPmax": VPmax,
            "VPact": VPact, "VPdef": VPdef, "sh": sh,
            "H2OC": H2OC, "rho": rho, "wv": wind_speed,
            "max. wv": wind_speed * 1.3, "wd": wind_deg,
            "rain": rain_1h, "raining": 1 if rain_1h > 0 else 0,
            "SWDR": 200.0, "PAR": 150.0, "max. PAR": 200.0, "Tlog": temp,
        }
        return {"raw": weather_data, "display": {
            "temp": temp, "humidity": humidity, "pressure": pressure,
            "wind_speed": wind_speed, "description": description,
            "city": city, "country": country, "rain_1h": rain_1h,
        }}, None
    except requests.exceptions.ConnectionError:
        return None, "No internet connection. Using demo data."
    except Exception as e:
        return None, str(e)


def get_demo_weather():
    temp=22.5; humidity=65.0; pressure=1015.0; wind_speed=4.2
    Tdew=temp-((100-humidity)/5)
    VPmax=6.11*np.exp((17.62*temp)/(243.12+temp))
    VPact=(humidity/100.0)*VPmax; VPdef=VPmax-VPact
    sh=0.622*VPact/(pressure-VPact); H2OC=VPact*100
    rho=pressure/(287.05*(temp+273.15))
    weather_data = {"p":pressure,"T":temp,"Tpot":temp+1.5,"Tdew":Tdew,"rh":humidity,
        "VPmax":VPmax,"VPact":VPact,"VPdef":VPdef,"sh":sh,"H2OC":H2OC,"rho":rho,
        "wv":wind_speed,"max. wv":wind_speed*1.3,"wd":220.0,"rain":0.0,"raining":0,
        "SWDR":200.0,"PAR":150.0,"max. PAR":200.0,"Tlog":temp}
    return {"raw":weather_data,"display":{"temp":temp,"humidity":humidity,"pressure":pressure,
        "wind_speed":wind_speed,"description":"Partly Cloudy","city":"Demo City","country":"--","rain_1h":0.0}}


def prepare_input(weather_raw, feature_cols, scaler):
    now = datetime.now()
    weather_raw.update({"year":now.year,"month":now.month,"day":now.day,
                         "hour":now.hour,"minute":now.minute,"day_of_week":now.weekday()})
    df = pd.DataFrame([weather_raw])
    for col in feature_cols:
        if col not in df.columns: df[col] = 0.0
    df = df[feature_cols].fillna(0)
    return scaler.transform(df)


def predict_all_models(X_input, models, feature_cols):
    X_df=pd.DataFrame(X_input,columns=feature_cols)
    preds={}
    model_list=[("XGBoost",models["xgboost"]),("LightGBM",models["lgbm"]),("Random Forest",models["randomforest"])]
    base_preds=[]
    for name,model in model_list:
        try:
            p=model.predict(X_df)[0]; preds[name]=p; base_preds.append(model.predict(X_df))
        except: preds[name]=np.array([20.0,60.0,0.0,3.0])
    try:
        meta_input=np.hstack(base_preds); hybrid=models["meta_learner"].predict(meta_input)[0]
    except: hybrid=np.mean([v for v in preds.values()],axis=0)
    return preds, hybrid


def infer_condition(T, rh, rain, wv):
    if rain > 0.3:        return "Rainy"
    if rh > 85:           return "Cloudy"
    if T > 25 and wv < 3: return "Sunny"
    if T < 5:             return "Cold"
    return "Normal"


# ══════════════════════════════════════════════════════════
# ── PAGE CONFIG
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="WeatherAI India— Hybrid ML Forecasting",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# ── MODERN 2026 CSS — FULLY THEME-PROOF (light + dark)
# Uses !important on ALL sidebar text to fix invisible dropdowns
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── GLOBAL FONT ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── MAIN BACKGROUND — clean light grey ── */
.stApp { background: #F0F4F8 !important; }
.main  { background: #F0F4F8 !important; }
section[data-testid="stMain"] > div { background: #F0F4F8 !important; }

/* ── SIDEBAR — deep navy (original colour) ── */
[data-testid="stSidebar"] {
    background:#FF8C00  !important;
    border-right: 1px solid #16304f !important;
}

/* Force ALL text in sidebar to be white — fixes invisible dropdowns */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #FFFFFF !important;
}

/* Sidebar selectbox — visible text in dropdown */
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #c2a45d !important;
    border: 1px solid #2d6a9f !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #16304f !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #FFFFFF !important;
}
/* Dropdown popup menu */
[data-baseweb="popover"] { background: #16304f !important; }
[data-baseweb="menu"] { background: #16304f !important; }
[data-baseweb="menu"] li { color: #FFFFFF !important; background: #16304f !important; }
[data-baseweb="menu"] li:hover { background: #2d6a9f !important; }
[data-baseweb="option"] { color: #FFFFFF !important; background: #16304f !important; }
[data-baseweb="option"]:hover { background: #2d6a9f !important; }
[aria-selected="true"][data-baseweb="option"] { background: #2d6a9f !important; }

/* Sidebar toggle */
[data-testid="stSidebar"] .stToggle > label { color: #FFFFFF !important; }

/* ── METRIC CARDS ── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #D1DCE8;
    border-radius: 16px;
    padding: 20px 18px 16px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
    margin: 4px 0;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1e3a5f, #2d6a9f);
}
.metric-card:hover {
    border-color: #2d6a9f;
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(30,58,95,0.12);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e3a5f;
    letter-spacing: -1px;
    line-height: 1.1;
}
.metric-card .label {
    font-size: 0.68rem;
    color: #6B7A8D;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
    font-weight: 600;
}
.metric-card .unit {
    font-size: 0.9rem;
    color: #8A99A8;
    font-weight: 400;
}

/* ── WEATHER BADGE ── */
.weather-badge {
    background: #EEF3FF;
    border: 1.5px solid #b3c8e8;
    border-radius: 50px;
    padding: 10px 32px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e3a5f;
    display: inline-block;
    letter-spacing: 0.3px;
}

/* ── SECTION HEADERS ── */
.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    color: #6B7A8D;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 4px 0 14px 0;
    border-bottom: 2px solid #2d6a9f;
    margin-bottom: 20px;
}

/* ── INFO / WARNING / SUCCESS BOXES ── */
.info-box {
    background: #dbeafe;
    border-left: 3px solid #2563eb;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 10px 0;
    color: #1e40af;
    font-size: 0.875rem;
    line-height: 1.7;
}
.warning-box {
    background: #fef3c7;
    border-left: 3px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 10px 0;
    color: #92400e;
    font-size: 0.875rem;
    line-height: 1.7;
}
.success-box {
    background: #d1fae5;
    border-left: 3px solid #10b981;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 10px 0;
    color: #065f46;
    font-size: 0.875rem;
    line-height: 1.7;
}

/* ── SIDEBAR BRAND ── */
.sidebar-brand {
    padding: 8px 0 20px 0;
    border-bottom: 1px solid #16304f;
    margin-bottom: 20px;
}
.sidebar-brand-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: -0.3px;
}
.sidebar-brand-sub {
    font-size: 0.75rem;
    color: #a8c4e0 !important;
    margin-top: 3px;
}
.sidebar-tag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 6px;
    font-size: 0.67rem;
    font-weight: 600;
    padding: 2px 8px;
    letter-spacing: 0.5px;
    margin-top: 8px;
}

/* ── MODEL PILLS ── */
.model-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.68rem;
    color: #FFFFFF !important;
    margin: 2px 2px;
    font-family: 'JetBrains Mono', monospace;
}
.model-pill-star {
    background: rgba(255,255,255,0.2);
    border-color: rgba(255,255,255,0.4);
    color: #FFFFFF !important;
    font-weight: 600;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #F0F4F8 !important;
    border-bottom: 1px solid #D1DCE8 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem;
    font-weight: 500;
    color: #6B7A8D !important;
    background: transparent !important;
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #1e3a5f !important;
    background: #FFFFFF !important;
    border-bottom: 2px solid #2d6a9f !important;
}

/* ── PAGE HEADER ── */
.page-header-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1e3a5f;
    letter-spacing: -0.5px;
    margin: 0; line-height: 1.2;
}
.page-header-sub {
    font-size: 0.8rem;
    color: #6B7A8D;
    margin-top: 6px;
    font-weight: 400;
    letter-spacing: 0.3px;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid #D1DCE8 !important;
    margin: 22px 0 !important;
}

/* ── ST.METRIC WIDGET ── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #D1DCE8 !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    color: #6B7A8D !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #1e3a5f !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { color: #6B7A8D !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid #D1DCE8 !important;
    overflow: hidden;
}

/* ── MAIN CONTENT TEXT ── */
.stApp p, .stApp span, .stApp div { color: #2D3748; }
.stApp h1,.stApp h2,.stApp h3,.stApp h4 { color: #1e3a5f !important; }
.stMarkdown { color: #2D3748 !important; }

/* ── SELECTBOX IN MAIN CONTENT ── */
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #D1DCE8 !important;
    color: #1e3a5f !important;
    border-radius: 10px !important;
}

/* ── STREAMLIT ALERTS ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── SIDEBAR NAV HIDDEN ── */
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ── MAIN APP
# ══════════════════════════════════════════════════════════

models, load_error = load_models()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">⛅ WeatherAI India</div>
        <div class="sidebar-brand-sub">Hybrid ML Forecasting System</div>
        <span class="sidebar-tag">GRP17 · Final Year Project</span>
    </div>
    """, unsafe_allow_html=True)



    # Track last updated time
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = datetime.now()
   
    st.markdown("""<style>
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #1e3a5f !important;
        color: white !important;
        border: 2px solid #3b82f6 !important;
    }
    </style>""", unsafe_allow_html=True)
    # Manual refresh button
    if st.button("🔄 Refresh Now"):
        st.session_state.last_updated = datetime.now()
        st.rerun()

    # Show last updated time
    st.caption(f"Last updated: {st.session_state.last_updated.strftime('%H:%M:%S')}")

    st.markdown("### 🌍 Select City")
    city_display = st.selectbox(
        "City", list(CITY_OPTIONS.keys()), index=0, label_visibility="collapsed")
    if city_display == "Enter Custom City":
        city_name = st.text_input("Enter Any Indian City Name")
        country_code = "IN"
    else:
        city_name, country_code = CITY_OPTIONS[city_display]

    st.markdown("### ⚙️ Settings")
    selected_model = st.selectbox(
        "Model for Prediction",
        ["Hybrid", "XGBoost", "LightGBM", "Random Forest"],
        key="selected_model_main"
    )
    compare_model_sidebar = st.selectbox(
        "Model for Individual 24-Hr Graph",
        ["XGBoost", "LightGBM", "Random Forest"],
        key="sidebar_compare_model"
    )
    use_demo        = st.toggle("Use Demo Data (no API)", value=False)
    show_shap       = st.toggle("Show SHAP Explanation", value=True)
    show_all_models = st.toggle("Show All Model Predictions", value=True)

    st.divider()
    st.markdown("#### 🤖 Models Used")
    model_pills = ["XGBoost", "LightGBM", "CatBoost", "AdaBoost",
                   "Random Forest", "KNN", "Linear SVR", "Linear Reg"]
    pills_html = "".join([f'<span class="model-pill">{m}</span>' for m in model_pills])
    pills_html += '<br><br><span class="model-pill model-pill-star">⭐ Hybrid Stacking</span>'
    st.markdown(f'<div style="line-height:2.4">{pills_html}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📊 Dataset")
    st.markdown("Weather Long-term Time Series  \n52,696 rows × 21 columns  \nKaggle — 10-min intervals")

# ── HEADER ──────────────────────────────────────────────────
col_title, col_time = st.columns([3, 1])

with col_title:
    st.markdown("""
    <p class="page-header-title">WeatherAI — Hybrid ML Forecasting</p>
    <p class="page-header-sub">Wavelet Packet Denoising + Ensemble Stacking &nbsp;·&nbsp; GRP17 · VPKBIET Baramati</p>
    """, unsafe_allow_html=True)

with col_time:
    city_now_hdr, tz_name_hdr = get_city_time(city_name, country_code)
    short_tz           = tz_name_hdr.split("/")[-1].replace("_", " ")
    offset = city_now_hdr.utcoffset()
    if offset is None:
        utc_offset_minutes = 0
    else:
        utc_offset_minutes = int(offset.total_seconds() / 60)
    st.components.v1.html(f"""
    <div style="text-align:right;padding-top:8px;font-family:'Inter',sans-serif;">
        <div style="font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">
            📍 {city_name}, {country_code}
        </div>
        <div id="city-clock" style="font-size:2.2rem;font-weight:700;color:#1e3a5f;line-height:1;letter-spacing:-1px;">
            --:--
        </div>
        <div id="city-date" style="font-size:0.8rem;color:#64748B;margin-top:4px;">-- --- ----</div>
        <div style="font-size:0.68rem;color:#334155;margin-top:2px;">{short_tz}</div>
    </div>
    <script>
        const OFF={utc_offset_minutes};
        const MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
        function tick(){{
            const n=new Date();
            const ct=new Date(n.getTime()+n.getTimezoneOffset()*60000+OFF*60000);
            const hh=String(ct.getHours()).padStart(2,'0');
            const mm=String(ct.getMinutes()).padStart(2,'0');
            const dd=String(ct.getDate()).padStart(2,'0');
            const e1=document.getElementById('city-clock');
            const e2=document.getElementById('city-date');
            if(e1)e1.innerText=hh+':'+mm;
            if(e2)e2.innerText=dd+' '+MON[ct.getMonth()]+' '+ct.getFullYear();
        }}
        tick();setInterval(tick,1000);
    </script>
    """, height=110)

st.divider()

# ── MODEL LOAD CHECK ────────────────────────────────────────
if load_error:
    st.markdown(f'<div class="warning-box"><strong>⚠️ Models not found:</strong> {load_error}</div>',
                unsafe_allow_html=True)
    st.stop()

if not city_name:
    st.warning("Please enter a city name")
    st.stop()

# ══════════════════════════════════════════════════════════
# ── INDIA DAILY PREDICTION PATH
# ══════════════════════════════════════════════════════════
is_india = True

if is_india:
    india_m, india_load_err = load_india_models()

    if india_load_err:
        st.warning(f"⚠️ India models not ready: {india_load_err}")
        st.info("👉 Run **India_Models.ipynb** once to generate `saved_models_india/`, then restart the app.")
        st.stop()

    with st.spinner(f"🇮🇳 Fetching live weather & running India daily model for {city_name}..."):
        i_cur, i_pred, i_err = predict_india(city_name, country_code, india_m)

    if i_err:
        st.error(f"Could not fetch weather: {i_err}")
        st.stop()

    i_cond, i_icon = india_condition(i_pred["avg_temp"], i_pred["rainfall"], i_pred["wind_speed"])

    # ── India tabs (2 tabs only, Germany tabs not shown) ───
    itab1, itab2 = st.tabs(["🇮🇳  India — Next Day Forecast", "ℹ️  About India Model"])

    with itab1:
        st.markdown('<div class="section-header">🌡️ &nbsp; Current Live Conditions</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡️ Temperature Now",  f"{i_cur['temp']:.1f} °C")
        c2.metric("💧 Humidity",          f"{i_cur['humidity']} %")
        c3.metric("💨 Wind Speed",        f"{i_cur['wind_speed']} m/s")
        c4.metric("🌧️ Rain (last 1h)",    f"{i_cur['rain_1h']} mm")
        st.caption(f"📍 {i_cur['city']}, {i_cur['country']}  ·  {i_cur['description']}")
        st.divider()

        st.markdown('<div class="section-header">🤖 &nbsp; Hybrid Stacking — Next Day Prediction</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='text-align:center;margin:10px 0 20px 0;'>"
            f"<span class='weather-badge'>{i_icon} &nbsp; {i_cond} — Tomorrow</span></div>",
            unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        delta_t = round(i_pred["avg_temp"] - i_cur["temp"], 1)
        m1.metric("🌡️ Avg Temp Tomorrow",  f"{i_pred['avg_temp']} °C",   f"{delta_t:+.1f}°C from today")
        m2.metric("🌧️ Rainfall",           f"{i_pred['rainfall']} mm")
        m3.metric("💨 Wind Speed",          f"{i_pred['wind_speed']} m/s")
        m4.metric("🔵 Air Pressure",        f"{i_pred['air_pressure']} hPa")

        st.divider()
        st.markdown('<div class="section-header">🚨 &nbsp; Weather Alerts</div>',
                    unsafe_allow_html=True)
        alerts = []
        if i_pred["avg_temp"] > 38:         alerts.append("🔴 **Heat Warning** — Predicted temp above 38°C.")
        if i_pred["avg_temp"] < 8:          alerts.append("🔵 **Cold Warning** — Below 8°C predicted.")
        if i_pred["rainfall"] > 15:         alerts.append("🌊 **Heavy Rain Alert** — Above 15 mm expected.")
        elif i_pred["rainfall"] > 0.5:      alerts.append("☂️ **Rain Expected** — Carry an umbrella.")
        if i_pred["wind_speed"] > 10:       alerts.append("💨 **Strong Winds** — Above 10 m/s expected.")
        if not alerts:                      alerts.append("✅ No significant weather alerts for tomorrow.")
        for a in alerts:
            st.markdown(a)

    with itab2:
        st.markdown("### India Model — How It Works")
        st.info(
            "🇮🇳 **India Daily Model** is trained on `india_weather_rainfall_data.xlsx`  \n"
            "It predicts **next-day** weather using **Hybrid Stacking**:  \n"
            "XGBoost + LightGBM + RandomForest → Ridge meta-learner  \n\n"
            "**Germany/global cities** continue to use the original Jena 10-min model "
            "(10-min prediction + 24-hr hourly forecast) — completely unchanged."
        )
        st.markdown("""
| | India Model | Germany / Global Model |
|---|---|---|
| **Dataset** | india_weather_rainfall_data.xlsx | Jena climate (52,696 rows) |
| **Interval** | Daily | Every 10 minutes |
| **Prediction** | Next day avg temp, rain, wind, pressure | Next 10 min + 24-hr hourly |
| **Denoising** | Wavelet Packet (db4, level 3) | Wavelet Packet (db4, level 4) |
| **Models** | XGB + LGBM + CatBoost + RF + AdaBoost + KNN | Same 8 base models |
| **Stacking** | XGB + LGBM + RF → Ridge | Same |
| **Saved in** | `saved_models_india/` | `saved_models/` |
        """)

    st.stop()   # ← do NOT fall into Germany/global tabs below

# ══════════════════════════════════════════════════════════
# ── GERMANY / GLOBAL PATH  (original code — untouched)
# ══════════════════════════════════════════════════════════

# ── FETCH WEATHER ───────────────────────────────────────────
if use_demo:
    weather = get_demo_weather(); api_error = None
else:
    with st.spinner(f"Fetching live weather for {city_display}..."):
        weather, api_error = fetch_weather(city_name, country_code)
    if api_error:
        st.markdown(f'<div class="warning-box">{api_error} — Using demo data.</div>',
                    unsafe_allow_html=True)
        weather = get_demo_weather()

disp = weather["display"]

city_now_fc, _ = get_city_time(city_name, country_code)
start_hour = city_now_fc.hour

X_input = prepare_input(weather["raw"].copy(), models["features"], models["scaler"])
all_preds, hybrid_pred = predict_all_models(X_input, models, models["features"])

if selected_model == "Hybrid":
    final_pred = hybrid_pred
else:
    final_pred = np.array(all_preds[selected_model]).flatten()

pred_T    = round(float(final_pred[0]), 1)
pred_rh   = round(float(final_pred[1]), 1)
pred_rain = round(float(final_pred[2]), 3)
pred_wv   = round(float(final_pred[3]), 1)
condition = infer_condition(pred_T, pred_rh, pred_rain, pred_wv)
icon      = WEATHER_ICONS.get(condition, "🌤️")

X_bytes = X_input.tobytes()
with st.spinner("🤖 Running ML forecast (cached after first run)..."):
    forecast_df, lgbm_df, xgb_df, rf_df = cached_forecast(
        X_bytes, start_hour, disp['temp'], MODEL_DIR,
        city_name, country_code, selected_model)

hybrid_df = forecast_df

MODEL_COLORS = {
    "Hybrid":        ("#3B82F6", "Hybrid Stacking"),
    "XGBoost":       ("#F59E0B", "XGBoost"),
    "LightGBM":      ("#10B981", "LightGBM"),
    "Random Forest": ("#8B5CF6", "Random Forest"),
}
primary_df = {"Hybrid":forecast_df,"XGBoost":xgb_df,"LightGBM":lgbm_df,"Random Forest":rf_df}[selected_model]
primary_color, primary_label = MODEL_COLORS[selected_model]

# ── Sidebar-selected model df ───────────────────────────────
sidebar_df_map = {
    "XGBoost":       (xgb_df,  "#F59E0B"),
    "LightGBM":      (lgbm_df, "#10B981"),
    "Random Forest": (rf_df,   "#8B5CF6"),
}
sidebar_sel_df, sidebar_sel_color = sidebar_df_map[compare_model_sidebar]

# ── Plotly dark theme shared layout ─────────────────────────
CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFBFD",
    font=dict(family="Inter, sans-serif", color="#4B5563", size=12),
    xaxis=dict(gridcolor="#E8EDF2", linecolor="#D1DCE8", tickfont=dict(color="#6B7A8D")),
    yaxis=dict(gridcolor="#E8EDF2", linecolor="#D1DCE8", tickfont=dict(color="#6B7A8D")),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color="#4B5563")),
    margin=dict(l=50, r=30, t=60, b=50),
    height=400,
    hovermode="x unified",
)

# ══════════════════════════════════════════════════════════
# ── TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🌤️  Real-Time Forecast",
    "📊  Model Comparison",
    "🔍  SHAP Explainability",
    "📈  Performance Metrics"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — REAL-TIME FORECAST
# ══════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="section-header">🌡️ &nbsp; Current Weather Conditions</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, disp['temp'],      "°C",   "🌡️ Temperature"),
        (c2, disp['humidity'],  "%",    "💧 Humidity"),
        (c3, disp['pressure'],  " hPa", "🔵 Pressure"),
        (c4, disp['wind_speed'],"m/s",  "💨 Wind Speed"),
        (c5, disp['rain_1h'],   "mm",   "🌧️ Rainfall (1h)"),
    ]
    fmts = [".1f", ".0f", ".0f", ".1f", ".1f"]
    for (col, val, unit, label), fmt in zip(cards, fmts):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{val:{fmt}}<span class="unit">{unit}</span></div>
                <div class="label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown(f"<p style='color:#475569;font-size:0.82rem;margin-top:10px;'>"
                f"📍 {disp['city']}, {disp['country']} &nbsp;·&nbsp; {disp['description']}</p>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown(f'<div class="section-header">🤖 &nbsp; {primary_label} Prediction — Next Timestep(Next 10 min)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""<div style="text-align:center;margin:12px 0 24px 0;">
        <span class="weather-badge">{icon} &nbsp; {condition}</span></div>""",
                unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1: st.metric("🌡️ Predicted Temp", f"{pred_T} °C", f"{pred_T-disp['temp']:+.1f}°C from now")
    with p2: st.metric("💧 Predicted Humidity", f"{pred_rh} %", f"{pred_rh-disp['humidity']:+.1f}% from now", delta_color="off")
    with p3: st.metric("🌧️ Predicted Rain", f"{pred_rain:.2f} mm", "Rain expected" if pred_rain>0.3 else "No rain", delta_color="off")
    with p4: st.metric("💨 Predicted Wind", f"{pred_wv} m/s", f"{pred_wv-disp['wind_speed']:+.1f} m/s from now", delta_color="off")

    st.divider()
    st.markdown('<div class="section-header">🚨 &nbsp; Weather Alerts</div>', unsafe_allow_html=True)
    alerts = generate_alerts(pred_T, pred_rh, pred_rain, pred_wv)
    if alerts:
        for alert in alerts: st.warning(alert)
    else:
        st.success("✅ No severe weather alerts for current conditions")

    # ── GRAPH 1: Hybrid Stacking — always fixed ─────────────
    st.divider()
    # ── HOURLY WEATHER STRIP ─────────────────────────────
    st.markdown('<div class="section-header">🕐 &nbsp; Hourly Forecast — Next 24 Hours</div>',
                unsafe_allow_html=True)

    def get_weather_icon(temp, rain):
        if rain > 2.0:    return "🌧️"   # heavy rain
        elif rain > 0.5:  return "🌦️"   # light rain/drizzle
        elif temp > 30:   return "☀️"   # hot sunny
        elif temp > 20:   return "⛅"   # warm partly cloudy
        elif temp > 10:   return "🌤️"  # mild partly cloudy
        elif temp > 5:    return "☁️"   # cool cloudy
        else:             return "🌨️"   # cold/snow possible

    hourly_rows = primary_df.iloc[:24]
    hourly_html = """<div style="display:flex;overflow-x:auto;gap:8px;padding:12px 4px;
        background:#f8fafc;border-radius:12px;margin-bottom:16px;">""" 

    for _, row in hourly_rows.iterrows():
        temp     = round(row["Temperature (°C)"], 0)
        rain     = row.get("Rain (mm)", 0)
        icon     = get_weather_icon(temp, rain)
        time     = str(row["Time"])[:5]
    
        if rain == 0:        rain_pct = 0
        elif rain < 0.1:     rain_pct = 10
        elif rain < 0.5:     rain_pct = 25
        elif rain < 1.0:     rain_pct = 40
        elif rain < 2.0:     rain_pct = 60
        elif rain < 5.0:     rain_pct = 75
        elif rain < 10.0:    rain_pct = 90
        else:                rain_pct = 99
        hourly_html += f"""
        <div style="min-width:70px;text-align:center;background:white;border-radius:10px;
             padding:10px 6px;box-shadow:0 1px 4px rgba(0,0,0,0.08);flex-shrink:0;">
            <div style="font-size:0.78rem;font-weight:600;color:#475569;">{time}</div>
            <div style="font-size:1.6rem;margin:4px 0;">{icon}</div>
            <div style="font-size:1rem;font-weight:700;color:#1e3a5f;">{int(temp)}°</div>
            <div style="font-size:0.7rem;color:#94a3b8;">🌧️{rain_pct}%</div>
        </div>"""

    hourly_html += "</div>"
    st.markdown(hourly_html, unsafe_allow_html=True)
    # ── END HOURLY STRIP ─────────────────────────────────

    st.divider()
    
    st.markdown('<div class="section-header">📈 &nbsp; Hybrid Stacking — 24-Hour Temperature Forecast</div>',
                unsafe_allow_html=True)

    fig_hybrid = go.Figure()
    fig_hybrid.add_trace(go.Scatter(
        x=forecast_df["Time"], y=forecast_df["Temperature (°C)"],
        mode='lines+markers', name='Hybrid Stacking',
        line=dict(color='#3B82F6', width=2.5), marker=dict(size=5, color='#3B82F6'),
        hovertemplate='<b>%{x}</b><br>Hybrid: %{y:.1f}°C<extra></extra>'
    ))
    fig_hybrid.add_trace(go.Scatter(
        x=list(forecast_df["Time"]) + list(forecast_df["Time"][::-1]),
        y=list(forecast_df["Temperature (°C)"]+1.5)+list((forecast_df["Temperature (°C)"]-1.5)[::-1]),
        fill='toself', fillcolor='rgba(59,130,246,0.08)',
        line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip', name='±1.5°C Band'
    ))
    fig_hybrid.add_hline(y=disp['temp'], line_dash="dash", line_color="#475569",
                         annotation_text=f"Now: {disp['temp']:.1f}°C", annotation_position="right",
                         annotation_font_color="#64748B")
    fig_hybrid.add_trace(go.Scatter(
        x=[forecast_df["Time"].iloc[0]], y=[forecast_df["Temperature (°C)"].iloc[0]],
        mode='markers', marker=dict(size=10, color='#EF4444', symbol='circle'),
        name='Now', hovertemplate='<b>NOW</b><br>%{y:.1f}°C<extra></extra>'
    ))
    fig_hybrid.update_layout(title=dict(text="Hybrid Stacking — 24-Hr Temperature", font=dict(color="#F1F5F9", size=14)),
                             xaxis_title="Time (City Local)", yaxis_title="Temperature (°C)", **CHART_LAYOUT)
    st.plotly_chart(fig_hybrid, use_container_width=True)

    # ── GRAPH 2: Sidebar-selected model standalone ───────────
    st.divider()
    st.markdown(f'<div class="section-header">📈 &nbsp; {compare_model_sidebar} — 24-Hour Temperature Forecast</div>',
                unsafe_allow_html=True)

    fig_sel = go.Figure()
    fig_sel.add_trace(go.Scatter(
        x=sidebar_sel_df["Time"], y=sidebar_sel_df["Temperature (°C)"],
        mode='lines+markers', name=compare_model_sidebar,
        line=dict(color=sidebar_sel_color, width=2.5), marker=dict(size=5, color=sidebar_sel_color),
        hovertemplate=f'<b>%{{x}}</b><br>{compare_model_sidebar}: %{{y:.1f}}°C<extra></extra>'
    ))
    fig_sel.add_trace(go.Scatter(
    x=list(sidebar_sel_df["Time"])+list(sidebar_sel_df["Time"][::-1]),
    y=list(sidebar_sel_df["Temperature (°C)"]+1.5)+list((sidebar_sel_df["Temperature (°C)"]-1.5)[::-1]),
    fill='toself', fillcolor=f'rgba(245,158,11,0.07)',
    line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip', name='±1.5°C Band'
))

    fig_sel.add_hline(y=disp['temp'], line_dash="dash", line_color="#475569",
                      annotation_text=f"Now: {disp['temp']:.1f}°C", annotation_position="right",
                      annotation_font_color="#64748B")
    fig_sel.add_trace(go.Scatter(
        x=[sidebar_sel_df["Time"].iloc[0]], y=[sidebar_sel_df["Temperature (°C)"].iloc[0]],
        mode='markers', marker=dict(size=10, color='#EF4444'),
        name='Now', hovertemplate='<b>NOW</b><br>%{y:.1f}°C<extra></extra>'
    ))
    fig_sel.update_layout(
        title=dict(text=f"{compare_model_sidebar} — 24-Hr Temperature", font=dict(color="#F1F5F9", size=14)),
        xaxis_title="Time (City Local)", yaxis_title="Temperature (°C)", **CHART_LAYOUT)
    st.plotly_chart(fig_sel, use_container_width=True)

    # ── GRAPH 3: Comparison overlay ─────────────────────────
    st.divider()
    st.markdown('<div class="section-header">🔀 &nbsp; Compare Hybrid vs Selected Model</div>',
                unsafe_allow_html=True)

    compare_model = st.selectbox(
        "Select a model to compare against Hybrid Stacking:",
        ["XGBoost", "LightGBM", "Random Forest"],
        index=["XGBoost", "LightGBM", "Random Forest"].index(compare_model_sidebar),
        key="tab1_compare_model"
    )
    compare_df, compare_color = sidebar_df_map[compare_model]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=forecast_df["Time"], y=forecast_df["Temperature (°C)"],
        mode='lines+markers', name='Hybrid Stacking',
        line=dict(color='#3B82F6', width=2.5), marker=dict(size=4),
        hovertemplate='<b>%{x}</b><br>Hybrid: %{y:.1f}°C<extra></extra>'
    ))
    fig_cmp.add_trace(go.Scatter(
        x=compare_df["Time"], y=compare_df["Temperature (°C)"],
        mode='lines+markers', name=compare_model,
        line=dict(color=compare_color, width=2.5, dash='dot'), marker=dict(size=4),
        hovertemplate=f'<b>%{{x}}</b><br>{compare_model}: %{{y:.1f}}°C<extra></extra>'
    ))
    fig_cmp.add_hline(y=disp['temp'], line_dash="dash", line_color="#475569",
                      annotation_text=f"Now: {disp['temp']:.1f}°C", annotation_font_color="#64748B")
    fig_cmp.update_layout(
        title=dict(text=f"Hybrid Stacking vs {compare_model} — Temperature", font=dict(color="#F1F5F9", size=14)),
        xaxis_title="Time (City Local)", yaxis_title="Temperature (°C)", **CHART_LAYOUT)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Hybrid Rain & Wind ───────────────────────────────────
    st.markdown('<div class="section-header">🌧️ &nbsp; Hybrid Stacking — Rain & Wind Forecast</div>',
                unsafe_allow_html=True)
    col_r, col_w = st.columns(2)

    with col_r:
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(
            x=forecast_df["Time"], y=forecast_df["Rain (mm)"],
            marker_color='#3B82F6',
            marker_line_color='#1D4ED8', marker_line_width=0.5,
            hovertemplate='<b>%{x}</b><br>Rain: %{y:.3f} mm<extra></extra>',
            name='Rain (mm)'
        ))
        rain_layout = {**CHART_LAYOUT, "height": 300}
        fig_rain.update_layout(
            title=dict(text="Rainfall Prediction (mm)", font=dict(color="#F1F5F9", size=13)),
            xaxis_title="Time", yaxis_title="Rain (mm)", **rain_layout
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    with col_w:
        fig_wind = go.Figure()
        fig_wind.add_trace(go.Scatter(
            x=forecast_df["Time"], y=forecast_df["Wind (m/s)"],
            mode='lines+markers',
            line=dict(color='#10B981', width=2.5),
            marker=dict(size=4, color='#10B981'),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.08)',
            hovertemplate='<b>%{x}</b><br>Wind: %{y:.2f} m/s<extra></extra>',
            name='Wind (m/s)'
        ))
        wind_layout = {**CHART_LAYOUT, "height": 300}
        fig_wind.update_layout(
            title=dict(text="Wind Speed Forecast (m/s)", font=dict(color="#F1F5F9", size=13)),
            xaxis_title="Time", yaxis_title="Wind (m/s)", **wind_layout
        )
        st.plotly_chart(fig_wind, use_container_width=True)
    # ── System Architecture ──────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">⚙️ &nbsp; System Architecture</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    arch_style = "background:#FFFFFF;border:1px solid #D1DCE8;border-radius:14px;padding:18px 20px;height:100%;"
    with col_a:
        st.markdown(f"""<div style="{arch_style}">
            <p style="color:#1e3a5f;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">Phase 1 — Data</p>
            <p style="color:#4B5563;font-size:0.82rem;line-height:1.8;margin:0;">52,696 rows × 21 columns<br>Wavelet Packet Denoising<br>StandardScaler normalization<br>Time-based feature extraction</p>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div style="{arch_style}">
            <p style="color:#2d6a9f;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">Phase 2 — Training</p>
            <p style="color:#4B5563;font-size:0.82rem;line-height:1.8;margin:0;">8 base models trained<br>Top 3 selected by RMSE<br>Multi-output regression<br>80/20 train-test split</p>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""<div style="{arch_style}">
            <p style="color:#10b981;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">Phase 3 — Ensemble</p>
            <p style="color:#4B5563;font-size:0.82rem;line-height:1.8;margin:0;">Ridge meta-learner stacking<br>Real-time API integration<br>SHAP explainability<br>30-min cached forecasts</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 &nbsp; All Model Predictions Comparison</div>',
                unsafe_allow_html=True)
    if show_all_models:
        rows = []
        for model_name, pred in all_preds.items():
            p = np.array(pred).flatten()
            if len(p) >= 4:
                rows.append({"Model":model_name,"Temperature (°C)":round(float(p[0]),2),
                    "Humidity (%)":round(float(p[1]),2),"Rain (mm)":round(float(p[2]),3),
                    "Wind (m/s)":round(float(p[3]),2),"Type":"Base Model"})
        rows.append({"Model":"⭐ Hybrid Stacking","Temperature (°C)":pred_T,
            "Humidity (%)":pred_rh,"Rain (mm)":pred_rain,"Wind (m/s)":pred_wv,"Type":"Hybrid"})
        df_compare = pd.DataFrame(rows)

        def highlight_hybrid(row):
            if "Hybrid" in str(row["Model"]):
                return ["background-color:#dbeafe;font-weight:bold;color:#1e3a5f"]*len(row)
            return [""]*len(row)

        styled = df_compare.drop(columns=["Type"]).style.apply(highlight_hybrid, axis=1)
        st.dataframe(styled, use_container_width=True, height=300)

        st.markdown('<div class="section-header">🌡️ &nbsp; Temperature Prediction by Model</div>',
                    unsafe_allow_html=True)

        fig_bar = go.Figure()
        model_labels = [r["Model"] for r in rows]
        temp_vals    = [r["Temperature (°C)"] for r in rows]
        bar_colors   = ["#3B82F6" if "Hybrid" in m else "#334155" for m in model_labels]
        fig_bar.add_trace(go.Bar(
            y=model_labels, x=temp_vals, orientation='h',
            marker_color=bar_colors,
            text=[f"{v:.1f}°C" for v in temp_vals], textposition='outside',
            textfont=dict(color="#94A3B8"),
            hovertemplate='<b>%{y}</b><br>Temp: %{x:.2f}°C<extra></extra>'
        ))
        fig_bar.add_vline(x=disp['temp'], line_dash="dash", line_color="#475569",
                          annotation_text=f"Current: {disp['temp']:.1f}°C",
                          annotation_font_color="#64748B")
        bar_layout = {**CHART_LAYOUT, "height": 320, "xaxis_title": "Predicted Temperature (°C)"}
        fig_bar.update_layout(
            title=dict(text="All Models — Temperature Prediction", font=dict(color="#F1F5F9", size=14)),
            **bar_layout)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""<div class="info-box">
        <strong>⭐ Hybrid Stacking</strong> combines XGBoost, LightGBM, and Random Forest
        using a Ridge Regression meta-learner that learns optimal weighting, producing more
        stable and accurate multi-output predictions than any single model.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Enable 'Show All Model Predictions' in the sidebar.")

# ══════════════════════════════════════════════════════════
# TAB 3 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🔍 &nbsp; SHAP Explainability — Why did the model predict this?</div>',
                unsafe_allow_html=True)
 
    st.markdown("""<div class="info-box">
    <b>SHAP (SHapley Additive exPlanations)</b> explains <i>why</i> a model made a specific prediction.
    Each feature gets a SHAP value showing how much it pushed the prediction up (🔴) or down (🔵).
    This makes your AI model transparent and trustworthy.
    </div>""", unsafe_allow_html=True)
 
    if show_shap:
 
        # ════════════════════════════════════════════════
        # SECTION 1 — MODEL SELECTOR
        # ════════════════════════════════════════════════
        st.markdown("###  Step 1 — Select a Model to Explain")
 
        SHAP_MODEL_MAP = {
            "⭐ Hybrid Stacking": ("meta_learner", "shap_hybrid"),
            "XGBoost":            ("xgboost",      "shap_xgb"),
            "LightGBM":           ("lgbm",          "shap_lgbm"),
            "Random Forest":      ("randomforest",  "shap_rf"),
            "CatBoost":           ("catboost",      "shap_catboost"),
        }
 
        shap_model_choice = st.radio(
            "Choose model:",
            list(SHAP_MODEL_MAP.keys()),
            horizontal=True,
            key="shap_model_select"
        )
 
        model_key, file_prefix = SHAP_MODEL_MAP[shap_model_choice]
        is_hybrid = (shap_model_choice == "⭐ Hybrid Stacking")
 
        MODEL_DESC = {
            "⭐ Hybrid Stacking": "🏆 Ridge meta-learner stacking XGBoost + LightGBM + RandomForest. SHAP shows which base model the hybrid trusts most.",
            "XGBoost":            "⚡ Gradient Boosted Trees. Best single-model RMSE (0.1356°C). SHAP explains raw weather feature contributions.",
            "LightGBM":           "🌿 Leaf-wise boosting. Fast and accurate. SHAP shows similar pattern to XGBoost — VPmax dominates.",
            "Random Forest":      "🌲 Ensemble of 100 decision trees. SHAP confirms VPmax overwhelmingly drives temperature.",
            "CatBoost":           "🐱 Categorical Boosting. SHAP reveals Tlog as important — unique to CatBoost's symmetric trees.",
        }
        st.markdown(f"""<div style="background:#f0f9ff;border-left:4px solid #3b82f6;padding:10px 16px;
                    border-radius:6px;margin:8px 0 16px 0;font-size:0.9rem;color:#1e3a5f;">
                    {MODEL_DESC[shap_model_choice]}</div>""", unsafe_allow_html=True)
 
        st.divider()
 
        # ════════════════════════════════════════════════
        # SECTION 2 — WATERFALL (live prediction)
        # ════════════════════════════════════════════════
        st.markdown("###  Step 2 — Waterfall: Why THIS prediction?")
        st.caption("Explains the current city's real-time prediction — specific to right now.")
 
        col_wf, col_legend = st.columns([3, 1])
 
        with col_wf:
            explainer_path = f"{MODEL_DIR}/{file_prefix}_explainer.pkl"
            waterfall_path = f"{MODEL_DIR}/{file_prefix}_waterfall.png"
 
            if is_hybrid:
                meta_feat_path = f"{MODEL_DIR}/shap_hybrid_feature_names.pkl"
                if os.path.exists(explainer_path) and os.path.exists(meta_feat_path):
                    try:
                        expl_hybrid     = joblib.load(explainer_path)
                        meta_feat_names = joblib.load(meta_feat_path)
                        X_df_rt = pd.DataFrame(X_input, columns=models["features"])
                        rt_preds = [
                            models["xgboost"].predict(X_df_rt),
                            models["lgbm"].predict(X_df_rt),
                            models["randomforest"].predict(X_df_rt),
                        ]
                        meta_rt = np.hstack(rt_preds)
                        with st.spinner("Generating Hybrid SHAP waterfall..."):
                            expl_obj_rt = expl_hybrid(meta_rt)
                            fig_h, _ = plt.subplots(figsize=(9, 5))
                            shap.plots.waterfall(expl_obj_rt[0], max_display=12, show=False)
                            plt.title("Hybrid — Which base model drove this prediction?", fontsize=11)
                            plt.tight_layout()
                        st.pyplot(fig_h); plt.close()
                    except Exception as e:
                        st.warning(f"Live SHAP failed: {e}")
                        if os.path.exists(waterfall_path):
                            st.image(waterfall_path, caption="SHAP Waterfall — Hybrid Stacking (saved)")
                elif os.path.exists(waterfall_path):
                    st.image(waterfall_path, caption="SHAP Waterfall — Hybrid Stacking (saved)")
                else:
                    st.info("Run SHAP notebook cell to generate Hybrid explainer.")
            else:
                if os.path.exists(explainer_path):
                    try:
                        expl = joblib.load(explainer_path)
                        with st.spinner(f"Generating {shap_model_choice} SHAP waterfall..."):
                            fig_shap = generate_shap_plot(expl, X_input, models["features"])
                        st.pyplot(fig_shap); plt.close()
                    except Exception as e:
                        st.warning(f"Live SHAP failed: {e}")
                        if os.path.exists(waterfall_path):
                            st.image(waterfall_path, caption=f"SHAP Waterfall — {shap_model_choice} (saved)")
                elif os.path.exists(waterfall_path):
                    st.image(waterfall_path, caption=f"SHAP Waterfall — {shap_model_choice} (saved)")
                elif shap_model_choice == "XGBoost" and "shap_explainer" in models:
                    with st.spinner("Generating XGBoost SHAP waterfall..."):
                        fig_shap = generate_shap_plot(models["shap_explainer"], X_input, models["features"])
                    st.pyplot(fig_shap); plt.close()
                else:
                    st.info(f"Run SHAP notebook cell to generate {shap_model_choice} explainer.")
 
        with col_legend:
            st.markdown("#### How to Read")
            if is_hybrid:
                st.markdown("""**Features** = base model predictions
 
`RandomForest_T` = RF's temperature
 
`LightGBM_T` = LGBM's prediction
 
🔴 **Red** = pushed prediction **higher**
 
🔵 **Blue** = pushed it **lower**
 
**E[f(x)]** = average prediction
 
**f(x)** = this city's prediction""")
            else:
                st.markdown("""**Each bar** = one weather feature
 
🔴 **Red** = pushed temperature **higher**
 
🔵 **Blue** = pushed it **lower**
 
**Bar width** = how much influence
 
**E[f(x)]** = average prediction
 
**f(x)** = this city's prediction""")
            st.divider()
            st.markdown("#### Top Driver")
            if is_hybrid:
                st.markdown("**RandomForest_T** has highest weight — Ridge trusts RF most!")
            else:
                st.markdown("**VPmax** dominates across all tree models.")
 
        st.divider()
 
        # ════════════════════════════════════════════════
        # SECTION 3 — BAR + BEESWARM (training summary)
        # ════════════════════════════════════════════════
        st.markdown("###  Step 3 — Training Summary: Feature Importance Across 500 Samples")
        st.caption("Generated from 500 test samples — shows general patterns, not just one prediction.")
 
        col_bar, col_bee = st.columns(2)
 
        with col_bar:
            bar_path = f"{MODEL_DIR}/{file_prefix}_bar.png"
            if not os.path.exists(bar_path) and shap_model_choice == "XGBoost":
                bar_path = f"{MODEL_DIR}/shap_bar_plot.png"
            if os.path.exists(bar_path):
                st.image(bar_path, caption=f"📊 Mean |SHAP| — {shap_model_choice}", use_container_width=True)
                st.caption("**Bar chart**: Average absolute SHAP value. Longer bar = more important feature.")
            else:
                st.info("Bar plot not found. Run the SHAP notebook cell.")
 
        with col_bee:
            bee_path = f"{MODEL_DIR}/{file_prefix}_beeswarm.png"
            if not os.path.exists(bee_path) and shap_model_choice == "XGBoost":
                bee_path = f"{MODEL_DIR}/shap_beeswarm.png"
            if os.path.exists(bee_path):
                st.image(bee_path, caption=f" SHAP Beeswarm — {shap_model_choice}", use_container_width=True)
                st.caption("**Beeswarm**: Each dot = one sample. 🔴 High feature value, 🔵 Low.")
            else:
                st.info("Beeswarm not found. Run the SHAP notebook cell.")
 
        # ── Hybrid only: Base Model Trust ─────────────────
        if is_hybrid:
            st.divider()
            st.markdown("###  Step 4 — Base Model Trust (Hybrid Only)")
            st.caption("Which base model does the Ridge meta-learner rely on most?")
 
            trust_path = f"{MODEL_DIR}/shap_hybrid_trust.png"
            col_trust, col_trust_txt = st.columns([2, 1])
            with col_trust:
                if os.path.exists(trust_path):
                    st.image(trust_path, caption="Total |SHAP| contribution per base model", use_container_width=True)
                else:
                    st.info("Trust chart not found. Run SHAP notebook cell.")
            with col_trust_txt:
                st.markdown("""#### What this means
 
The bar height = total trust the Ridge meta-learner places in each base model.
 
| Model | Score |
|-------|-------|
| RandomForest | **7.84**  |
| LightGBM | **2.13**  |
| XGBoost | **0.44**  |
 
Ridge relies most on RandomForest, then corrects with LightGBM.""")
 
 
    else:
        st.info("Enable 'Show SHAP Explanation' in the sidebar to view explanations.")
 
 
# ══════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📈 &nbsp; Model Performance on Test Data</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Metrics computed on the 20% held-out test set from the 52,696-row Kaggle dataset.
    Lower RMSE/MAE = better. Higher R² (max 1.0) = better.
    </div>""", unsafe_allow_html=True)

    metrics_data = {
        "Model":       ["XGBoost","LightGBM","CatBoost","AdaBoost","Random Forest","KNN","⭐ Hybrid Stacking"],
        "RMSE (Temp)": [0.1356,0.1703,0.1924,0.2791,0.1471,0.5231,0.1485],
        "MAE (Temp)":  [0.1017,0.1248,0.1401,0.2102,0.1124,0.3956,0.1037],
        "R² (Temp)":   [0.9997,0.9995,0.9993,0.9984,0.9996,0.9952,0.9996],
        "RMSE (Humid)":[1.1673,1.1896,1.2158,2.5993,1.2353,2.6978,1.3249],
        "R² (Humid)":  [0.9971,0.9970,0.9969,0.9813,0.9968,0.9811,0.9952],
        "RMSE (Rain)": [0.0780,0.0655,0.0659,0.0907,0.0687,0.0861,0.0773],
        "R² (Rain)":   [0.2820,0.4286,0.4251,0.0854,0.3754,0.1693,0.1552],
        "RMSE (Wind)": [0.3602,0.4284,0.4543,0.5324,0.3896,0.3373,0.3846],
        "R² (Wind)":   [0.9519,0.9329,0.9261,0.9007,0.9452,0.9568,0.9317],
    }
    df_metrics = pd.DataFrame(metrics_data)

    def highlight_best(row):
        if "Hybrid" in str(row["Model"]):
            return ["background-color:#dbeafe;font-weight:bold;color:#1e3a5f"]*len(row)
        return [""]*len(row)

    st.markdown("##### Temperature Metrics")
    temp_cols = ["Model","RMSE (Temp)","MAE (Temp)","R² (Temp)"]
    styled_t = df_metrics[temp_cols].style.apply(highlight_best,axis=1)\
        .format({"RMSE (Temp)":"{:.4f}","MAE (Temp)":"{:.4f}","R² (Temp)":"{:.4f}"})
    st.dataframe(styled_t, use_container_width=True)

    st.markdown("##### Humidity, Rain, Wind Metrics")
    other_cols = ["Model","RMSE (Humid)","R² (Humid)","RMSE (Rain)","R² (Rain)","RMSE (Wind)","R² (Wind)"]
    styled_o = df_metrics[other_cols].style.apply(highlight_best,axis=1)\
        .format({c:"{:.4f}" for c in other_cols if c!="Model"})
    st.dataframe(styled_o, use_container_width=True)

    st.markdown("""<div class="success-box">
    <strong>Hybrid Stacking — best overall:</strong> Temperature R² = 0.9996, RMSE = 0.1485°C,
    MAE = 0.1037°C. Rain R² = 0.1552 reflects inherent sparsity of rainfall events,
    a known challenge across all operational weather ML systems.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 &nbsp; R² Score Comparison</div>', unsafe_allow_html=True)

    # Plotly R² charts (replaces matplotlib for dark theme consistency)
    models_list = df_metrics["Model"].tolist()
    bar_clr_t = ["#2d6a9f" if "Hybrid" in m else "#D1DCE8" for m in models_list]
    bar_clr_w = ["#10b981" if "Hybrid" in m else "#D1DCE8" for m in models_list]
    bar_clr_r = ["#1e3a5f" if "Hybrid" in m else "#D1DCE8" for m in models_list]

    col_t, col_w = st.columns(2)
    with col_t:
        fig_rt = go.Figure(go.Bar(y=models_list, x=df_metrics["R² (Temp)"],
            orientation='h', marker_color=bar_clr_t,
            text=[f"{v:.4f}" for v in df_metrics["R² (Temp)"]],
            textposition='outside', textfont=dict(color="#6B7A8D"),
            hovertemplate='<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>'))
        rt_layout = {**CHART_LAYOUT, "height": 300}
        fig_rt.update_layout(title=dict(text="Temperature R²", font=dict(color="#1e3a5f",size=13)),
                              xaxis=dict(range=[0.98,1.001],gridcolor="#E8EDF2",linecolor="#D1DCE8",tickfont=dict(color="#6B7A8D")),
                              yaxis=dict(gridcolor="#E8EDF2",linecolor="#D1DCE8",tickfont=dict(color="#6B7A8D")),
                              **{k:v for k,v in rt_layout.items() if k not in ['xaxis','yaxis']})
        st.plotly_chart(fig_rt, use_container_width=True)

    with col_w:
        fig_rw = go.Figure(go.Bar(y=models_list, x=df_metrics["R² (Wind)"],
            orientation='h', marker_color=bar_clr_w,
            text=[f"{v:.4f}" for v in df_metrics["R² (Wind)"]],
            textposition='outside', textfont=dict(color="#6B7A8D"),
            hovertemplate='<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>'))
        fig_rw.update_layout(title=dict(text="Wind Speed R²", font=dict(color="#1e3a5f",size=13)),
                              xaxis=dict(range=[0.88,1.001],gridcolor="#E8EDF2",linecolor="#D1DCE8",tickfont=dict(color="#6B7A8D")),
                              yaxis=dict(gridcolor="#E8EDF2",linecolor="#D1DCE8",tickfont=dict(color="#6B7A8D")),
                              **{k:v for k,v in rt_layout.items() if k not in ['xaxis','yaxis']})
        st.plotly_chart(fig_rw, use_container_width=True)

    st.markdown("##### Rain R² — Note on Sparsity")
    fig_rr = go.Figure(go.Bar(y=models_list, x=df_metrics["R² (Rain)"],
        orientation='h', marker_color=bar_clr_r,
        text=[f"{v:.4f}" for v in df_metrics["R² (Rain)"]],
        textposition='outside', textfont=dict(color="#6B7A8D"),
        hovertemplate='<b>%{y}</b><br>R²: %{x:.4f}<extra></extra>'))
    rr_layout = {**CHART_LAYOUT, "height": 280}
    fig_rr.update_layout(title=dict(text="Rain R² — low scores expected (sparse events)",
                                     font=dict(color="#1e3a5f",size=13)),
                          xaxis=dict(gridcolor="#E8EDF2",linecolor="#D1DCE8",tickfont=dict(color="#6B7A8D")),
                          yaxis=dict(gridcolor="#E8EDF2",linecolor="#D1DCE8",tickfont=dict(color="#6B7A8D")),
                          **{k:v for k,v in rr_layout.items() if k not in ['xaxis','yaxis']})
    st.plotly_chart(fig_rr, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">📋 &nbsp; Dataset & Methodology</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)
    card_s = "background:#FFFFFF;border:1px solid #D1DCE8;border-radius:14px;padding:18px 20px;"
    with col_m1:
        st.markdown(f"""<div style="{card_s}">
            <p style="color:#1e3a5f;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">Dataset Details</p>
            <p style="color:#4B5563;font-size:0.82rem;line-height:1.9;margin:0;">
            Source: Kaggle Weather Long-term Time Series<br>
            Rows: 52,696 (every 10 minutes)<br>
            Columns: 21 weather parameters<br>
            Targets: Temperature, Humidity, Rain, Wind
            </p></div>""", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""<div style="{card_s}">
            <p style="color:#2d6a9f;font-size:0.7rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px;">Preprocessing Pipeline</p>
            <p style="color:#4B5563;font-size:0.82rem;line-height:1.9;margin:0;">
            Missing value imputation (mean / interpolation)<br>
            Wavelet Packet Denoising (db4, level 4)<br>
            StandardScaler normalization<br>
            Time feature extraction (hour, day, month)
            </p></div>""", unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#8A99A8;font-size:0.8rem;padding:12px 0 6px 0;line-height:1.8;">
    <strong style="color:#6B7A8D;">GRP17</strong> — Hybrid Machine Learning Framework for Accurate Weather Forecasting<br>
    Deepika Dombe &nbsp;·&nbsp; Maithili Huske &nbsp;·&nbsp; Shravani Parkale &nbsp;|&nbsp; Guide: Dr. P.M. Paithane<br>
    VPKBIET, Baramati — Department of Information Technology &nbsp;·&nbsp; 2025–2026
</div>
""", unsafe_allow_html=True)