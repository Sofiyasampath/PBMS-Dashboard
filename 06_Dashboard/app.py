import json
import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==================================================
# PROJECT CONFIGURATION
# ==================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "03_Machine_Learning"

st.set_page_config(
    page_title="PBMS | Battery Monitoring",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# PROFESSIONAL DASHBOARD STYLE
# ==================================================
st.markdown("""
<style>
    .stApp {
        background: #F3F6FA;
        color: #263445;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #17365D !important;
    }

    p, li, label,
    [data-testid="stCaptionContainer"] {
        color: #34445A !important;
    }

    [data-testid="stSidebar"] {
        background: #E8EEF5;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #263445 !important;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D8E1EC;
        border-radius: 12px;
        padding: 18px;
        animation: fadeIn 0.5s ease-out;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #526579 !important;
    }

    [data-testid="stMetricValue"] {
        color: #17365D !important;
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #D8E1EC;
        border-radius: 12px;
        padding: 20px;
        animation: fadeIn 0.5s ease-out;
    }

    input {
        color: #263445 !important;
        background: #FFFFFF !important;
    }

    /* Visible form buttons */
    .stApp div[data-testid="stFormSubmitButton"] button,
    .stApp div.stFormSubmitButton button,
    .stApp div.stButton button {
        background: #17365D !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        width: 100% !important;
        min-height: 48px !important;
        border: 1px solid #17365D !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    .stApp div[data-testid="stFormSubmitButton"] button p,
    .stApp div[data-testid="stFormSubmitButton"] button span,
    .stApp div.stButton button p,
    .stApp div.stButton button span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    .stApp div[data-testid="stFormSubmitButton"] button:hover,
    .stApp div.stButton button:hover {
        background: #24558A !important;
        border-color: #24558A !important;
    }

    [data-testid="stAlert"] {
        background: #E8F2FF !important;
        border: 1px solid #B8D4F5 !important;
        border-radius: 10px !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {
        color: #17365D !important;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #D8E1EC;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        animation: fadeIn 0.6s ease-out;
    }

    .status-row {
        background: #FFFFFF;
        border: 1px solid #D8E1EC;
        border-radius: 9px;
        padding: 13px 16px;
        margin: 8px 0;
        color: #263445;
        animation: fadeIn 0.5s ease-out;
    }

    .status-good {
        color: #147D64;
        font-weight: 600;
    }

    .status-pending {
        color: #946200;
        font-weight: 600;
    }

    .soc-ring {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 12px;
        animation: fadeIn 0.7s ease-out;
    }

    .soc-ring svg {
        width: 270px;
        max-width: 100%;
        height: auto;
        overflow: visible;
    }

    .soc-progress {
        stroke-dasharray: 301.593;
        stroke-dashoffset: 301.593;
        animation: chargeRing 1.8s ease-out forwards;
        stroke-linecap: round;
    }

    @keyframes chargeRing {
        from { stroke-dashoffset: 301.593; }
        to { stroke-dashoffset: var(--ring-offset); }
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(7px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    hr {
        border-color: #D8E1EC;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD TRAINED MODELS
# ==================================================
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model(filename):
    model_path = hf_hub_download(
        repo_id="sofiyasampath/pbms-trained-models",
        filename=filename
    )
    return joblib.load(model_path)

@st.cache_data
def load_features(filename):
    with open(MODEL_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    soc_model = load_model("pbms_soc_model.pkl")
    capacity_model = load_model("pbms_capacity_model.pkl")
    soh_model = load_model("pbms_soh_model.pkl")

    soc_features = load_features("soc_features.json")
    capacity_features = load_features("capacity_features.json")
    soh_features = load_features("soh_features.json")

except Exception as error:
    st.error(f"Could not load a model or feature file: {error}")
    st.stop()


# ==================================================
# REUSABLE CHART FUNCTIONS
# ==================================================
def make_soc_ring(value):
    """Create an animated visual ring for an SoC estimate."""
    safe_value = max(0.0, min(100.0, value))
    circumference = 2 * math.pi * 48
    offset = circumference * (1 - safe_value / 100)

    return f"""
    <div class="soc-ring">
      <svg viewBox="0 0 120 120"
           role="img"
           aria-label="Estimated state of charge {safe_value:.1f} percent">
        <circle
          cx="60" cy="60" r="48"
          fill="none"
          stroke="#E2E8F0"
          stroke-width="10"
        />
        <circle
          cx="60" cy="60" r="48"
          fill="none"
          stroke="#168C78"
          stroke-width="10"
          stroke-dasharray="{circumference:.3}"
          stroke-dashoffset="{circumference:.3}"
          style="--ring-offset: {offset:.3f};"
          class="soc-progress"
          transform="rotate(-90 60 60)"
        />
        <text
          x="60" y="56"
          text-anchor="middle"
          font-size="19"
          font-weight="700"
          fill="#17365D">{safe_value:.1f}%</text>
        <text
          x="60" y="72"
          text-anchor="middle"
          font-size="6"
          letter-spacing="0.6"
          fill="#526579">ESTIMATED SOC</text>
      </svg>
    </div>
    """


def make_bar_chart(title, label, value, unit, max_value):
    """Create a clearly labelled chart for one model estimate."""
    fig = go.Figure(
        data=[
            go.Bar(
                x=[label],
                y=[value],
                marker_color="#168C78",
                marker_line_width=0,
                text=[f"{value:.2f} {unit}"],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    f"{label}<br>Estimate: %{{y:.2f}} {unit}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#17365D")),
        height=320,
        margin=dict(l=25, r=25, t=65, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#34445A", family="Arial"),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(
            title=unit,
            range=[0, max_value],
            gridcolor="#E7EDF4",
            zeroline=False,
        ),
    )

    return fig


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.title("🔋 PBMS")
    st.caption("Predictive Battery Management System")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard Overview",
            "SoC Estimation",
            "Capacity & SoH",
            "Project Information",
        ],
    )

    st.divider()
    st.caption("CONNECTION STATUS")
    st.markdown(
        '<div class="status-row">'
        '● <span class="status-pending">Demo mode</span><br>'
        '<small>ESP32 not connected</small></div>',
        unsafe_allow_html=True,
    )


# ==================================================
# HEADER
# ==================================================
st.title("Battery Management Dashboard")
st.write(
    "Estimate battery charge, discharge capacity and relative battery health."
)

st.warning(
    "Demonstration mode: inputs are entered manually. "
    "This dashboard is not connected to live ESP32 sensors."
)


# ==================================================
# DASHBOARD OVERVIEW
# ==================================================
if page == "Dashboard Overview":
    st.subheader("System Overview")

    st.write(
        "Choose a section from the navigation menu to enter battery data "
        "and view model estimates."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Available model", "SoC")
        st.caption("Estimates state of charge")

    with col2:
        st.metric("Available model", "Capacity")
        st.caption("Estimates discharge capacity")

    with col3:
        st.metric("Available model", "Relative SoH")
        st.caption("Estimates relative capacity retention")

    st.divider()
    st.subheader("How to use the dashboard")

    st.markdown("""
    1. **SoC Estimation:** enter voltage, current, temperature and C-rate.
    2. **Capacity & SoH:** enter features from a completed discharge profile.
    3. Submit the form to display the model estimates and charts.
    4. Visit Project Information to review the current system limitations.
    """)

    st.info(
        "The initial input values are examples for testing the interface. "
        "Use compatible measured data when evaluating the models."
    )


# ==================================================
# SOC ESTIMATION
# ==================================================
elif page == "SoC Estimation":
    st.subheader("State of Charge (SoC)")
    st.write(
        "Enter battery readings to estimate the state of charge."
    )

    with st.form("soc_form"):
        col1, col2 = st.columns(2)

        with col1:
            voltage = st.number_input(
                "Voltage (V)",
                min_value=0.0,
                value=4.06584,
                step=0.01,
                format="%.5f",
            )

            current = st.number_input(
                "Current (A)",
                value=5.44958,
                step=0.1,
                format="%.5f",
            )

        with col2:
            temperature = st.number_input(
                "Bottom temperature (°C)",
                value=23.19666,
                step=0.5,
                format="%.5f",
            )

            c_rate = st.number_input(
                "C-rate",
                min_value=0.0,
                value=1.0,
                step=0.1,
            )

        soc_submit = st.form_submit_button(
            "Estimate SoC",
            use_container_width=True,
        )

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

            st.divider()
            st.subheader("Estimation Result")

            result_col, chart_col = st.columns([1, 1])

            with result_col:
                st.metric("Estimated SoC", f"{prediction:.2f}%")
                st.caption("Model estimate based on the inputs provided.")

                if not 0 <= prediction <= 100:
                    st.warning(
                        "The estimate is outside the expected 0–100% range. "
                        "Check the inputs and model output."
                    )

            with chart_col:
                st.markdown(
                    make_soc_ring(prediction),
                    unsafe_allow_html=True,
                )

            st.caption(
                "The animated ring is a visual display of the model estimate, "
                "not a live battery-charging indicator."
            )

            with st.expander("View entered battery readings"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Measurement": [
                                "Voltage",
                                "Current",
                                "Bottom temperature",
                                "C-rate",
                            ],
                            "Value": [
                                f"{voltage:.5f} V",
                                f"{current:.5f} A",
                                f"{temperature:.5f} °C",
                                f"{c_rate:.2f}",
                            ],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"SoC prediction failed: {error}")


# ==================================================
# CAPACITY AND SOH ESTIMATION
# ==================================================
elif page == "Capacity & SoH":
    st.subheader("Discharge Capacity and Relative SoH")

    st.write(
        "Enter features from a completed discharge profile. "
        "These estimates are not early fault predictions."
    )

    with st.form("health_form"):
        col1, col2 = st.columns(2)

        with col1:
            cycle = st.number_input(
                "Cycle number",
                min_value=1,
                value=100,
                step=1,
            )

            mean_voltage = st.number_input(
                "Mean voltage (V)",
                min_value=0.0,
                value=3.8,
                step=0.01,
            )

            min_voltage = st.number_input(
                "Minimum voltage (V)",
                min_value=0.0,
                value=3.5,
                step=0.01,
            )

            mean_current = st.number_input(
                "Mean current (A)",
                value=1.5,
                step=0.1,
            )

        with col2:
            mean_temp = st.number_input(
                "Mean temperature (°C)",
                value=25.0,
                step=0.5,
            )

            max_temp = st.number_input(
                "Maximum temperature (°C)",
                value=30.0,
                step=0.5,
            )

            duration = st.number_input(
                "Discharge duration (seconds)",
                min_value=0.0,
                value=3000.0,
                step=100.0,
            )

        health_submit = st.form_submit_button(
            "Estimate Capacity and SoH",
            use_container_width=True,
        )

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

            soh_prediction = float(
                soh_model.predict(soh_input)[0]
            )

            st.divider()
            st.subheader("Estimation Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated Discharge Capacity",
                    f"{capacity_prediction:.3f} Ah",
                )

            with col2:
                st.metric(
                    "Estimated Relative SoH",
                    f"{soh_prediction:.2f}%",
                )

            st.divider()
            st.subheader("Estimate Charts")

            chart1, chart2 = st.columns(2)

            with chart1:
                capacity_chart_max = max(
                    0.1, capacity_prediction * 1.25
                )
                capacity_fig = make_bar_chart(
                    "Discharge Capacity Estimate",
                    "Estimated capacity",
                    capacity_prediction,
                    "Ah",
                    capacity_chart_max,
                )
                st.plotly_chart(
                    capacity_fig,
                    use_container_width=True,
                    key="capacity_estimate_chart",
                )

            with chart2:
                soh_fig = make_bar_chart(
                    "Relative SoH Estimate",
                    "Estimated relative SoH",
                    soh_prediction,
                    "%",
                    100,
                )
                st.plotly_chart(
                    soh_fig,
                    use_container_width=True,
                    key="soh_estimate_chart",
                )

            st.caption(
                "The charts display the model estimates for the submitted inputs. "
                "They do not show historical battery measurements."
            )

            st.info(
                "Capacity and relative SoH use completed-discharge features. "
                "Accuracy can vary between batteries, and relative SoH is not "
                "a verified manufacturer-rated health measurement."
            )

            with st.expander("View entered discharge-profile readings"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Measurement": list(health_values.keys()),
                            "Value": list(health_values.values()),
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"Capacity/SoH prediction failed: {error}")


# ==================================================
# PROJECT INFORMATION
# ==================================================
elif page == "Project Information":
    st.subheader("Project Information")

    st.markdown("""
    **Project:** Predictive Battery Management System (PBMS)

    **Purpose:** Estimate battery state of charge, discharge capacity
    and relative state of health using trained machine-learning models.
    """)

    st.divider()
    st.subheader("Current System Status")

    statuses = [
        ("SoC estimation model", "Loaded", True),
        ("Discharge capacity model", "Loaded", True),
        ("Relative SoH model", "Loaded", True),
        ("ESP32 live sensor connection", "Not connected", False),
        ("Early fault prediction", "Not implemented or validated", False),
    ]

    for label, status, available in statuses:
        css_class = "status-good" if available else "status-pending"
        symbol = "●" if available else "○"

        st.markdown(
            f'<div class="status-row">'
            f'<strong>{label}</strong><br>'
            f'<span class="{css_class}">{symbol} {status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Important Limitations")

    st.markdown("""
    - This version uses manually entered values.
    - The ESP32 is not connected to the dashboard.
    - Capacity and relative SoH use completed-discharge features.
    - Relative SoH is a capacity-retention estimate, not a verified
      manufacturer-rated battery health measurement.
    - Early fault prediction has not been implemented or validated.
    - Model estimates should be checked against compatible real battery data.
    """)


# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "PBMS | Engineering Project Demonstration | "
    "Predictions require validation with real battery data."
)