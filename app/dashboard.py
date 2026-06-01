import streamlit as st
import requests, json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AU Hotels — Cancellation Intelligence",
    page_icon="🏨", layout="wide"
)

API_URL = "http://localhost:8000"

st.title("🏨 Australian Hotels — Booking Cancellation Intelligence")
st.markdown("*Powered by XGBoost · Deployed on AWS ap-southeast-2*")

# ── Sidebar inputs ────────────────────────────────────────────────────────
st.sidebar.header("Booking Details")

hotel          = st.sidebar.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
lead_time      = st.sidebar.slider("Lead Time (days)", 0, 365, 45)
adr            = st.sidebar.number_input("Average Daily Rate (AUD)", 50.0, 2000.0, 150.0)
deposit_type   = st.sidebar.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
market_segment = st.sidebar.selectbox("Market Segment",
    ["Online TA","Offline TA/TO","Direct","Corporate","Groups","Complementary","Aviation"])
prev_cancel    = st.sidebar.number_input("Previous Cancellations", 0, 20, 0)
special_req    = st.sidebar.slider("Special Requests", 0, 5, 1)
weekend_nights = st.sidebar.slider("Weekend Nights", 0, 7, 1)
week_nights    = st.sidebar.slider("Week Nights", 0, 14, 2)
arrival_month  = st.sidebar.selectbox("Arrival Month",
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"])

# ── Predict button ────────────────────────────────────────────────────────
if st.sidebar.button("🔮 Predict Cancellation Risk", use_container_width=True):
    payload = {
        "lead_time": lead_time,
        "arrival_date_year": 2025,
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": week_nights,
        "adults": 2, "children": 0, "babies": 0,
        "is_repeated_guest": 0,
        "previous_cancellations": prev_cancel,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "days_in_waiting_list": 0,
        "adr": adr,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": special_req,
        "hotel": hotel,
        "meal": "BB",
        "market_segment": market_segment,
        "distribution_channel": "TA/TO",
        "deposit_type": deposit_type,
        "customer_type": "Transient",
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "arrival_date_month": arrival_month,
    }

    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        result = res.json()

        prob  = result["cancellation_probability"]
        risk  = result["risk_level"]
        color = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#2ECC71"}[risk]

        # ── Main metrics ──
        col1, col2, col3 = st.columns(3)
        col1.metric("Cancellation Probability", f"{prob:.1%}")
        col2.metric("Risk Level", risk)
        col3.metric("Model AUC-ROC", f"{result['auc_roc']:.4f}")

        # ── Gauge chart ──
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Cancellation Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": color},
                "steps": [
                    {"range": [0,  40], "color": "#D5F5E3"},
                    {"range": [40, 70], "color": "#FDEBD0"},
                    {"range": [70, 100],"color": "#FADBD8"},
                ],
                "threshold": {"line": {"color": "black","width": 3},
                              "thickness": 0.8, "value": prob * 100}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # ── Recommendation ──
        st.info(f"**Recommendation:** {result['recommendation']}")

    except Exception as e:
        st.error(f"API error: {e}. Make sure the FastAPI server is running.")

# ── Model info panel ──────────────────────────────────────────────────────
st.markdown("---")
try:
    info = requests.get(f"{API_URL}/model-info", timeout=3).json()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",     info.get("model_type", "XGBoost"))
    c2.metric("AUC-ROC",   f"{info.get('auc_roc', 0):.4f}")
    c3.metric("F1 Score",  f"{info.get('f1', 0):.4f}")
    c4.metric("Features",  info.get("feature_count", "—"))
except:
    st.warning("Start the FastAPI server to see live model info.")
