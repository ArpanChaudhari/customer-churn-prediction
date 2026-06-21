import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration (Must be the very first line)
st.set_page_config(page_title="Churn Predictor AI", page_icon="🔮", layout="wide")

# 2. Inject Custom CSS for a Premium Look
st.markdown("""
<style>
    /* Style the Predict Button */
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    /* Custom Result Cards */
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .high-risk {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff9068 100%);
    }
    .low-risk {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Models
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('expected_columns.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")

# 4. SIDEBAR: Customer Demographics & Billing
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3126/3126647.png", width=100) # Cool icon
st.sidebar.title("Customer Profile")

tenure = st.sidebar.slider("Tenure (Months)", 0, 100, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=75.0)

# Auto-calculate total charges quietly in the background
total_charges = tenure * monthly_charges

paperless = st.sidebar.radio("Paperless Billing", ["Yes", "No"], horizontal=True)
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])


# 5. MAIN SCREEN: Extra Services & Dashboard
st.title("🔮 AI Customer Retention Dashboard")
st.markdown("Analyze customer risk and take proactive measures to prevent churn.")

st.subheader("Subscribed Services")
st.write("Does the customer use these additional features?")

# Nice 3-column layout for checkboxes
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

total_extra_services = sum([online_sec, online_backup, device_prot, tech_support, stream_tv, stream_movies])

st.markdown("<br>", unsafe_allow_html=True) # Add some spacing

# 6. The Prediction Engine
if st.button("Analyze Churn Risk"):
    
    numerical_data = {
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Total_Extra_Services': [total_extra_services]
    }
    
    input_df = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    for key, value in numerical_data.items():
        if key in input_df.columns:
            input_df[key] = value[0]
            
    categorical_mappings = {
        f"Contract_{contract}": 1,
        f"PaperlessBilling_{paperless}": 1,
        f"InternetService_{internet_service}": 1,
        f"PaymentMethod_{payment_method}": 1
    }
    
    for col_name, value in categorical_mappings.items():
        if col_name in input_df.columns:
            input_df[col_name] = 1
            
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1] 
    
    # 7. Beautiful Custom Results Output
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="result-card high-risk">
            <h2>🚨 HIGH CHURN RISK</h2>
            <p style="font-size: 20px;">Probability of leaving: <strong>{probability * 100:.1f}%</strong></p>
            <p><i>Recommendation: Reach out immediately with a retention offer.</i></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card low-risk">
            <h2>✅ LOYAL CUSTOMER</h2>
            <p style="font-size: 20px;">Probability of leaving: <strong>{probability * 100:.1f}%</strong></p>
            <p><i>Recommendation: Maintain standard service. No immediate action required.</i></p>
        </div>
        """, unsafe_allow_html=True)