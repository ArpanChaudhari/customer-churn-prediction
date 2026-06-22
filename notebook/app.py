import streamlit as st
import pandas as pd
import joblib
import altair as alt

st.set_page_config(page_title="Churn Predictor AI", page_icon="🔮", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """
<style>
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; border-radius: 8px; height: 50px; font-weight: bold;}
    .stButton>button:hover { background-color: #4338CA; }
</style>
""",
    unsafe_allow_html=True,
)

try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("expected_columns.pkl")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


def predict_churn(data_dict):
    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)

    for key in ["tenure", "MonthlyCharges", "Total_Extra_Services"]:
        if key in input_df.columns:
            input_df[key] = data_dict[key]

    categorical_mappings = {
        f"Contract_{data_dict['Contract']}": 1,
        f"PaperlessBilling_{data_dict['PaperlessBilling']}": 1,
        f"InternetService_{data_dict['InternetService']}": 1,
        f"PaymentMethod_{data_dict['PaymentMethod']}": 1,
    }
    for col_name, value in categorical_mappings.items():
        if col_name in input_df.columns:
            input_df[col_name] = 1

    scaled_data = scaler.transform(input_df)
    return model.predict_proba(scaled_data)[0][1]


# --- SIDEBAR ---
st.sidebar.title("Customer Profile")

# CHANGED: Tenure is now a number input just like Monthly Charges
tenure = st.sidebar.number_input(
    "Tenure (Months)", min_value=0, max_value=120, value=1, step=1
)
contract = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)

# CHANGED: Added step=5.0 so the +/- buttons jump by $5 at a time
monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)", min_value=0.0, value=105.0, step=5.0
)

paperless = st.sidebar.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
internet_service = st.sidebar.selectbox(
    "Internet Service", ["Fiber optic", "DSL", "No"]
)
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

# --- MAIN SCREEN ---
st.title("🔮 AI Customer Retention Dashboard")
st.subheader("Subscribed Services")

col1, col2, col3 = st.columns(3)
with col1:
    online_sec = st.checkbox("🔒 Online Security")
    online_backup = st.checkbox("💾 Online Backup")
with col2:
    device_prot = st.checkbox("📱 Device Protection")
    tech_support = st.checkbox("👨‍💻 Tech Support")
with col3:
    stream_tv = st.checkbox("📺 Streaming TV")
    stream_movies = st.checkbox("🎬 Streaming Movies")

total_extra_services = sum(
    [online_sec, online_backup, device_prot, tech_support, stream_tv, stream_movies]
)
st.markdown("<br>", unsafe_allow_html=True)


if st.button("Analyze Churn Risk & Generate Scenarios"):

    base_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "Total_Extra_Services": int(total_extra_services),
    }

    current_prob = predict_churn(base_data)

    st.session_state.history.insert(
        0,
        {
            "Tenure": tenure,
            "Contract": contract,
            "Monthly": f"${monthly_charges}",
            "Internet": internet_service,
            "Extra Services": int(total_extra_services),
            "Probability": f"{current_prob * 100:.1f}%",
        },
    )
    if len(st.session_state.history) > 5:
        st.session_state.history.pop()

    st.markdown("---")
    if current_prob > 0.5:
        st.error(f"🚨 HIGH RISK: Probability of leaving is {current_prob * 100:.1f}%")
    else:
        st.success(f"✅ LOW RISK: Probability of leaving is {current_prob * 100:.1f}%")

    st.markdown("### 📈 Risk Trend Analysis (Ceteris Paribus)")
    st.write("How changing *only one* variable affects this specific customer's risk:")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        tenure_range = range(1, 73, 5)
        tenure_probs = []
        for t in tenure_range:
            temp_data = base_data.copy()
            temp_data["tenure"] = t
            tenure_probs.append(predict_churn(temp_data) * 100)

        df_tenure = pd.DataFrame(
            {"Tenure (Months)": tenure_range, "Churn Risk (%)": tenure_probs}
        )
        chart_tenure = (
            alt.Chart(df_tenure)
            .mark_line(color="#ff4b4b", strokeWidth=4, point=True)
            .encode(
                x=alt.X(
                    "Tenure (Months):Q",
                    title="Tenure (Months)",
                    axis=alt.Axis(grid=False),
                ),
                y=alt.Y(
                    "Churn Risk (%):Q",
                    title="Churn Risk Probability (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=["Tenure (Months)", "Churn Risk (%)"],
            )
            .properties(title="Impact of Tenure on Risk", height=350)
        )

        # WARNING FIXED HERE:
        st.altair_chart(chart_tenure, width="stretch")

    with col_b:
        services_range = range(0, 7)
        services_probs = []
        for s in services_range:
            temp_data = base_data.copy()
            temp_data["Total_Extra_Services"] = s
            services_probs.append(predict_churn(temp_data) * 100)

        df_services = pd.DataFrame(
            {"Extra Services": services_range, "Churn Risk (%)": services_probs}
        )
        chart_services = (
            alt.Chart(df_services)
            .mark_line(color="#4F46E5", strokeWidth=4, point=True)
            .encode(
                x=alt.X(
                    "Extra Services:Q",
                    title="Total Extra Services",
                    axis=alt.Axis(grid=False, tickCount=6),
                ),
                y=alt.Y(
                    "Churn Risk (%):Q",
                    title="Churn Risk Probability (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=["Extra Services", "Churn Risk (%)"],
            )
            .properties(title="Impact of Services on Risk", height=350)
        )

        # WARNING FIXED HERE:
        st.altair_chart(chart_services, width="stretch")

    with col_c:
        charges_range = range(20, 121, 10)
        charges_probs = []
        for c in charges_range:
            temp_data = base_data.copy()
            temp_data["MonthlyCharges"] = c
            charges_probs.append(predict_churn(temp_data) * 100)

        df_charges = pd.DataFrame(
            {"Monthly Charge ($)": charges_range, "Churn Risk (%)": charges_probs}
        )
        chart_charges = (
            alt.Chart(df_charges)
            .mark_line(color="#11998e", strokeWidth=4, point=True)
            .encode(
                x=alt.X(
                    "Monthly Charge ($):Q",
                    title="Monthly Charge ($)",
                    axis=alt.Axis(grid=False),
                ),
                y=alt.Y(
                    "Churn Risk (%):Q",
                    title="Churn Risk Probability (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=["Monthly Charge ($)", "Churn Risk (%)"],
            )
            .properties(title="Impact of Bill on Risk", height=350)
        )

        # WARNING FIXED HERE:
        st.altair_chart(chart_charges, width="stretch")

# 5. DISPLAY HISTORY TABLE
st.markdown("---")
st.subheader("📝 Recent Prediction History")
if len(st.session_state.history) > 0:
    st.table(pd.DataFrame(st.session_state.history))
else:
    st.write("No predictions made yet.")
