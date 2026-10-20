
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "03_Machine_Learning"

st.set_page_config(
    page_title="Predictive Battery Management System",
    page_icon="🔋",
    layout="wide",
)

@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)

@st.cache_data
def load_features(filename):
    with open(MODEL_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)

st.title("🔋 Predictive Battery Management System")
st.caption("Machine-learning demonstration using trained battery models")

st.info(
    "Demo mode: values entered here are example inputs, not live ESP32 sensor data. "
    "Capacity and SoH models use completed-discharge features and are not early-warning models."
)

try:
    soc_model = load_model("pbms_soc_model.pkl")
    capacity_model = load_model("pbms_capacity_model.pkl")
    soh_model = load_model("pbms_soh_model.pkl")

    soc_features = load_features("soc_features.json")
    capacity_features = load_features("capacity_features.json")
    soh_features = load_features("soh_features.json")

except Exception as error:
    st.error(f"Could not load the models or feature files: {error}")
    st.stop()

st.subheader("1. State of Charge (SoC)")
st.write("Enter voltage, current, temperature and C-rate.")

with st.form("soc_form"):
    col1, col2 = st.columns(2)

    with col1:
        voltage = st.number_input(
            "Voltage (V)", min_value=0.0, value=4.06584, step=0.01
        )
        current = st.number_input(
            "Current (A)", value=5.44958, step=0.1
        )

    with col2:
        temperature = st.number_input(
            "Bottom temperature (°C)", value=23.19666, step=0.5
        )
        c_rate = st.number_input(
            "C-rate", min_value=0.0, value=1.0, step=0.1
        )

    soc_submit = st.form_submit_button("Estimate SoC")

if soc_submit:
    soc_values = {
        "Voltage [V]": voltage,
        "Current [A]": current,
        "TempBottom [C]": temperature,
        "CRate [-]": c_rate,
    }

    try:
        soc_input = pd.DataFrame(
            [[soc_values[name] for name in soc_features]],
            columns=soc_features,
        )
        prediction = float(soc_model.predict(soc_input)[0])
        st.metric("Estimated SoC", f"{prediction:.2f}%")

        if not 0 <= prediction <= 100:
            st.warning("Prediction is outside 0–100%; check model output and inputs.")
        st.caption("Estimate only; validate against real battery measurements.")
    except Exception as error:
        st.error(f"SoC prediction failed: {error}")

st.divider()
st.subheader("2. Discharge Capacity and Relative SoH")
st.write(
    "These inputs represent a completed discharge profile. "
    "Use realistic values from a compatible dataset; defaults are only for testing."
)

with st.form("health_form"):
    col1, col2 = st.columns(2)

    with col1:
        cycle = st.number_input(
            "Cycle number", min_value=1, value=100, step=1
        )
        mean_voltage = st.number_input(
            "Mean voltage (V)", min_value=0.0, value=3.8, step=0.01
        )
        min_voltage = st.number_input(
            "Minimum voltage (V)", min_value=0.0, value=3.5, step=0.01
        )
        mean_current = st.number_input(
            "Mean current (A)", value=1.5, step=0.1
        )

    with col2:
        mean_temp = st.number_input(
            "Mean temperature (°C)", value=25.0, step=0.5
        )
        max_temp = st.number_input(
            "Maximum temperature (°C)", value=30.0, step=0.5
        )
        duration = st.number_input(
            "Discharge duration (seconds)",
            min_value=0.0,
            value=3000.0,
            step=100.0,
        )

    health_submit = st.form_submit_button("Estimate Capacity and SoH")

if health_submit:
    health_values = {
        "Cycle_Number": cycle,
        "Mean_Voltage_V": mean_voltage,
        "Min_Voltage_V": min_voltage,
        "Mean_Current_A": mean_current,
        "Mean_Temperature_C": mean_temp,
        "Max_Temperature_C": max_temp,
        "Discharge_Duration_s": duration,
    }

    try:
        capacity_input = pd.DataFrame(
            [[health_values[name] for name in capacity_features]],
            columns=capacity_features,
        )
        soh_input = pd.DataFrame(
            [[health_values[name] for name in soh_features]],
            columns=soh_features,
        )

        capacity_prediction = float(
            capacity_model.predict(capacity_input)[0]
        )
        soh_prediction = float(soh_model.predict(soh_input)[0])

        col1, col2 = st.columns(2)
        col1.metric(
            "Estimated discharge capacity",
            f"{capacity_prediction:.3f} Ah",
        )
        col2.metric(
            "Estimated relative SoH",
            f"{soh_prediction:.2f}%",
        )

        st.caption(
            "These are model estimates, not verified battery health measurements. "
            "Accuracy can vary significantly between batteries."
        )
    except Exception as error:
        st.error(f"Capacity/SoH prediction failed: {error}")

st.divider()
st.subheader("Project status")
st.write("- SoC estimation model: loaded")
st.write("- Discharge capacity estimation model: loaded")
st.write("- Relative SoH estimation model: loaded")
st.write("- ESP32 live sensor integration: not connected yet")
st.write("- Early fault prediction: not implemented or validated yet")
