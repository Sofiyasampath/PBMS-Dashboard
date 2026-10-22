import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from early_warning import get_warning

# ---------------------------------------------------------
# PATHS AND PAGE CONFIGURATION
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "03_Machine_Learning"

st.set_page_config(
    page_title="PBMS | Battery Analytics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional navy + teal theme. Streamlit's theme can also
# be configured in .streamlit/config.toml if desired.
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #F4F7FB 0%, #EDF2F7 100%);
        color: #182536;
    }
    [data-testid="stHeader"] { background: rgba(244,247,251,0.92); }
    [data-testid="stSidebar"] {
        background: #101827;
    }
    [data-testid="stSidebar"] * { color: #F4F7FB; }
    .hero {
        padding: 1.35rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #101827 0%, #1C3448 65%, #176B70 100%);
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #D5E4ED; margin: .45rem 0 0 0; }
    .eyebrow {
        color: #63E0CC;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .75rem;
        font-weight: 700;
        margin-bottom: .45rem;
    }
    .section-card {
        background: white;
        border: 1px solid #DCE5EE;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin: .4rem 0 1rem 0;
        box-shadow: 0 5px 18px rgba(16, 24, 39, .045);
    }
    .status-pill {
        display: inline-block;
        padding: .35rem .7rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: .78rem;
        letter-spacing: .04em;
        background: #D9F5EF;
        color: #09695E;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE5EE;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 14px rgba(16,24,39,.04);
    }
    div.stButton > button[kind="primary"] {
        background: #087F8C;
        border: 1px solid #087F8C;
        color: white;
        border-radius: 10px;
        font-weight: 650;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #066873;
        border-color: #066873;
    }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #DCE5EE;
        border-radius: 14px;
        padding: 1rem;
    }
    .small-muted { color: #627386; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


@st.cache_data
def load_features(filename):
    with open(MODEL_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# HEADER + SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔋 PBMS")
    st.caption("Battery analytics workspace")
    st.divider()
    st.markdown("**Dashboard sections**")
    st.markdown("- State of Charge")
    st.markdown("- Capacity & State of Health")
    st.markdown("- Early Degradation Warning")
    st.divider()
    st.markdown("**System connection**")
    st.warning("Prototype mode")
    st.caption("ESP32 live sensor integration is not connected.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">EV Battery Intelligence · Prototype</div>
      <h1>Predictive Battery Management System</h1>
      <p>Estimate battery state, review capacity health, and explore an early capacity-degradation warning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top1, top2, top3 = st.columns([1.15, 1.15, 1.1])
with top1:
    st.metric("Model availability", "3 models expected", help="The dashboard attempts to load the SoC, capacity, and SoH model files.")
with top2:
    st.metric("Live sensor feed", "Not connected")
with top3:
    st.metric("Warning mode", "Manual prototype")

st.info(
    "Demo mode: values entered here are example inputs, not live ESP32 sensor data. "
    "The capacity/SoH models use completed-discharge features; the early-warning "
    "panel below applies capacity-decline categories to a value you enter."
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

# ---------------------------------------------------------
# SOC ESTIMATION
# ---------------------------------------------------------
st.markdown("## 01 · State of Charge")
st.markdown(
    '<p class="small-muted">Enter voltage, current, temperature and C-rate to run the trained SoC model.</p>',
    unsafe_allow_html=True,
)

with st.form("soc_form"):
    col1, col2 = st.columns(2)
    with col1:
        voltage = st.number_input("Voltage (V)", min_value=0.0, value=4.06584, step=0.01)
        current = st.number_input("Current (A)", value=5.44958, step=0.1)
    with col2:
        temperature = st.number_input("Bottom temperature (°C)", value=23.19666, step=0.5)
        c_rate = st.number_input("C-rate", min_value=0.0, value=1.0, step=0.1)
    soc_submit = st.form_submit_button("Estimate SoC", type="primary", use_container_width=True)

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
        st.metric("Estimated State of Charge", f"{prediction:.2f}%")
        # A visual gauge makes the SoC output easier to read.
        gauge_value = max(0.0, min(100.0, prediction))
        fig_soc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={"suffix": "%", "font": {"color": "#182536", "size": 32}},
            title={"text": "SoC estimate", "font": {"color": "#182536", "size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#627386"},
                "bar": {"color": "#087F8C"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#DCE5EE",
                "steps": [
                    {"range": [0, 20], "color": "#FBE3E3"},
                    {"range": [20, 80], "color": "#DDF4EF"},
                    {"range": [80, 100], "color": "#DDEBFA"},
                ],
            },
        ))
        fig_soc.update_layout(height=260, margin=dict(l=20, r=20, t=55, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_soc, use_container_width=True)
        if not 0 <= prediction <= 100:
            st.warning("Prediction is outside 0–100%; check model output and inputs.")
        st.caption("Estimate only; validate against real battery measurements.")
    except Exception as error:
        st.error(f"SoC prediction failed: {error}")

st.divider()

# ---------------------------------------------------------
# CAPACITY + SOH
# ---------------------------------------------------------
st.markdown("## 02 · Discharge Capacity & Relative SoH")
st.markdown(
    '<p class="small-muted">These inputs represent a completed discharge profile. Defaults are only for testing.</p>',
    unsafe_allow_html=True,
)

with st.form("health_form"):
    col1, col2 = st.columns(2)
    with col1:
        cycle = st.number_input("Cycle number", min_value=1, value=100, step=1)
        mean_voltage = st.number_input("Mean voltage (V)", min_value=0.0, value=3.8, step=0.01)
        min_voltage = st.number_input("Minimum voltage (V)", min_value=0.0, value=3.5, step=0.01)
        mean_current = st.number_input("Mean current (A)", value=1.5, step=0.1)
    with col2:
        mean_temp = st.number_input("Mean temperature (°C)", value=25.0, step=0.5)
        max_temp = st.number_input("Maximum temperature (°C)", value=30.0, step=0.5)
        duration = st.number_input("Discharge duration (seconds)", min_value=0.0, value=3000.0, step=100.0)
    health_submit = st.form_submit_button("Estimate Capacity & SoH", type="primary", use_container_width=True)

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
        capacity_prediction = float(capacity_model.predict(capacity_input)[0])
        soh_prediction = float(soh_model.predict(soh_input)[0])

        metric1, metric2 = st.columns(2)
        metric1.metric("Estimated discharge capacity", f"{capacity_prediction:.3f} Ah")
        metric2.metric("Estimated relative SoH", f"{soh_prediction:.2f}%")

        fig_health = go.Figure()
        fig_health.add_trace(go.Bar(
            x=["Estimated capacity", "Relative SoH"],
            y=[capacity_prediction, soh_prediction],
            marker_color=["#087F8C", "#3D78B5"],
            text=[f"{capacity_prediction:.3f} Ah", f"{soh_prediction:.2f}%"],
            textposition="outside",
            hovertemplate="%{x}<br>Value: %{y:.3f}<extra></extra>",
        ))
        fig_health.update_layout(
            title="Model output overview (different units)",
            yaxis_title="Numeric model output (units differ)",
            height=330,
            margin=dict(l=20, r=20, t=55, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#182536"),
            showlegend=False,
        )
        st.plotly_chart(fig_health, use_container_width=True)
        st.caption(
            "Capacity and SoH have different units, so this chart is for a visual overview only. "
            "These are model estimates, not verified battery measurements."
        )
    except Exception as error:
        st.error(f"Capacity/SoH prediction failed: {error}")

st.divider()

# ---------------------------------------------------------
# EARLY CAPACITY-DEGRADATION WARNING
# ---------------------------------------------------------
st.markdown("## 03 · Early Capacity-Degradation Warning")
st.markdown(
    '<p class="small-muted">Compare a reference capacity with a manually entered future-capacity estimate.</p>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("### Capacity outlook")
    st.info(
        "Prototype demonstration only. This is not a validated battery fault, "
        "thermal-runaway, or certified safety detector."
    )

    input_col, summary_col = st.columns([1, 1.15], gap="large")
    with input_col:
        reference_capacity = st.number_input(
            "Reference capacity (Ah)",
            min_value=0.01,
            value=2.0009,
            step=0.01,
            format="%.4f",
            key="warning_reference_capacity",
        )
        predicted_capacity = st.number_input(
            "Predicted capacity 10 discharge cycles ahead (Ah)",
            min_value=0.0,
            value=1.90,
            step=0.01,
            format="%.4f",
            key="warning_predicted_capacity",
        )
        check_warning = st.button(
            "Analyse Capacity Outlook",
            type="primary",
            use_container_width=True,
        )

    decline = (1 - predicted_capacity / reference_capacity) * 100
    warning = get_warning(predicted_capacity, reference_capacity)

    with summary_col:
        status_colours = {
            "Normal": ("#D9F5EF", "#09695E"),
            "Early Warning": ("#FFF1CC", "#805500"),
            "High Warning": ("#FFE2C2", "#8A3B00"),
            "Critical Degradation": ("#FBE0E0", "#A52222"),
        }
        bg, fg = status_colours.get(warning, ("#E5EAF0", "#344256"))
        st.markdown(
            f"""
            <div style="background:{bg};border:1px solid {fg}33;border-radius:16px;padding:1.15rem 1.3rem;">
              <div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;color:{fg};">
                Prototype warning status
              </div>
              <div style="font-size:1.8rem;font-weight:800;color:{fg};margin:.35rem 0;">
                {warning}
              </div>
              <div style="color:{fg};font-size:.95rem;">
                Capacity decline: {decline:.2f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Capacity difference", f"{reference_capacity - predicted_capacity:.3f} Ah")
        st.progress(max(0.0, min(1.0, decline / 100.0)), text="Relative capacity decline")

    # Compare reference and entered predicted capacity.
    fig_capacity = go.Figure()
    fig_capacity.add_trace(go.Bar(
        x=["Reference capacity", "Entered future capacity"],
        y=[reference_capacity, predicted_capacity],
        marker_color=["#3D78B5", "#087F8C"],
        text=[f"{reference_capacity:.3f} Ah", f"{predicted_capacity:.3f} Ah"],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.3f} Ah<extra></extra>",
    ))
    fig_capacity.update_layout(
        title="Reference vs entered future capacity",
        yaxis_title="Capacity (Ah)",
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#182536"),
        showlegend=False,
    )
    fig_capacity.update_yaxes(rangemode="tozero", gridcolor="#DCE5EE")
    st.plotly_chart(fig_capacity, use_container_width=True)

    if check_warning:
        if warning.startswith("Invalid"):
            st.error(warning)
        elif warning == "Critical Degradation":
            st.error("Critical Degradation — review the entered capacity estimate.")
        elif warning == "High Warning":
            st.warning("High Warning — the entered capacity estimate indicates a larger decline.")
        elif warning == "Early Warning":
            st.info("Early Warning — the entered capacity estimate indicates a measurable decline.")
        else:
            st.success("Normal — the entered estimate is below the prototype warning thresholds.")

    st.caption(
        "Prototype categories: Normal <10% decline; Early Warning ≥10%; "
        "High Warning ≥20%; Critical Degradation ≥30%. These are demonstration "
        "thresholds only, not validated battery-safety limits. The predicted capacity "
        "is entered manually and is not automatically generated by the Random Forest model."
    )

st.divider()

# ---------------------------------------------------------
# PROJECT STATUS
# ---------------------------------------------------------
st.markdown("## System Status")
status1, status2, status3 = st.columns(3)
status1.success("SoC model loaded")
status2.success("Capacity model loaded")
status3.success("SoH model loaded")

status4, status5 = st.columns(2)
status4.warning("ESP32 live sensor integration: not connected")
status5.info("Early capacity warning: manual-input prototype")

st.caption(
    "PBMS dashboard prototype · Model estimates require validation against compatible battery data. "
    "Do not use this interface as the sole basis for battery safety decisions."
)
