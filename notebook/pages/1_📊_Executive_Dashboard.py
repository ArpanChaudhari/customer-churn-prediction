import streamlit as st
import pandas as pd
import joblib
import altair as alt
import os

st.set_page_config(page_title="Executive Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .metric-card {
        background: rgba(31, 41, 55, 0.4);
        border-left: 6px solid #8b5cf6;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Register custom premium theme for Altair
def premium_theme():
    return {
        'config': {
            'view': {'stroke': 'transparent'},
            'axis': {
                'domainColor': '#e5e7eb',
                'gridColor': '#f3f4f6',
                'labelColor': '#6b7280',
                'titleColor': '#374151',
                'titleFontWeight': 600,
            },
            'legend': {'labelColor': '#6b7280', 'titleColor': '#374151'},
            'title': {'color': '#f9fafb', 'fontSize': 16, 'fontWeight': 600}
        }
    }
alt.themes.register('premium', premium_theme)
alt.themes.enable('premium')

st.title("📊 Executive Churn Analytics Dashboard")

# 1. LOAD DATA & MODELS
@st.cache_data
def load_and_predict():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../data/Telco-Customer-Churn.csv")
    df = pd.read_csv(data_path)
    
    # Load models
    model_dir = os.path.dirname(os.path.dirname(__file__))
    model = joblib.load(os.path.join(model_dir, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    expected_columns = joblib.load(os.path.join(model_dir, "expected_columns.pkl"))
    
    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    extra_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Total_Extra_Services'] = (df[extra_services] == 'Yes').sum(axis=1)
    
    # Dummies
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Align columns
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    X = df_encoded[expected_columns]
    
    # Scale and predict
    scaled_data = scaler.transform(X)
    df['Risk_Probability'] = model.predict_proba(scaled_data)[:, 1]
    df['Risk_Category'] = pd.cut(df['Risk_Probability'], bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
    
    return df

try:
    df = load_and_predict()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

# SIDEBAR FILTERS
st.sidebar.header("Filters")
contract_filter = st.sidebar.multiselect("Contract Type", options=df['Contract'].unique(), default=df['Contract'].unique())
internet_filter = st.sidebar.multiselect("Internet Service", options=df['InternetService'].unique(), default=df['InternetService'].unique())

filtered_df = df[(df['Contract'].isin(contract_filter)) & (df['InternetService'].isin(internet_filter))]

# 2. KPIs
st.markdown("### Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_customers = len(filtered_df)
high_risk_customers = len(filtered_df[filtered_df['Risk_Category'] == 'High'])
avg_churn_risk = filtered_df['Risk_Probability'].mean()
revenue_at_risk = filtered_df[filtered_df['Risk_Category'] == 'High']['MonthlyCharges'].sum()

with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_customers:,}</div><div class='metric-label'>Total Customers</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{high_risk_customers:,}</div><div class='metric-label'>High Risk Customers</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_churn_risk:.1%}</div><div class='metric-label'>Average Churn Risk</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>${revenue_at_risk:,.2f}</div><div class='metric-label'>Monthly Revenue at Risk</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 3. CHARTS
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Risk Distribution")
    risk_counts = filtered_df['Risk_Category'].value_counts().reset_index()
    risk_counts.columns = ['Risk_Category', 'Count']
    chart_risk = alt.Chart(risk_counts).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X('Risk_Category', sort=['Low', 'Medium', 'High'], title="Risk Category", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Count', title="Number of Customers"),
        color=alt.Color('Risk_Category', scale=alt.Scale(domain=['Low', 'Medium', 'High'], range=['#10b981', '#f59e0b', '#ef4444']), legend=None),
        tooltip=['Risk_Category', 'Count']
    ).properties(height=320)
    st.altair_chart(chart_risk, use_container_width=True)

with col_b:
    st.subheader("Average Risk by Contract Type")
    contract_risk = filtered_df.groupby('Contract')['Risk_Probability'].mean().reset_index()
    chart_contract = alt.Chart(contract_risk).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X('Contract', title="Contract Type", sort='-y', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Risk_Probability', title="Average Risk Probability", axis=alt.Axis(format='%')),
        color=alt.Color('Contract', scale=alt.Scale(scheme='purples'), legend=None),
        tooltip=['Contract', alt.Tooltip('Risk_Probability', format='.1%')]
    ).properties(height=320)
    st.altair_chart(chart_contract, use_container_width=True)

# Demographics Row
st.markdown("---")
st.subheader("Risk Insights by Demographics")
col_c, col_d = st.columns(2)

with col_c:
    gender_risk = filtered_df.groupby('gender')['Risk_Probability'].mean().reset_index()
    chart_gender = alt.Chart(gender_risk).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Risk_Probability", type="quantitative"),
        color=alt.Color(field="gender", type="nominal"),
        tooltip=['gender', alt.Tooltip('Risk_Probability', format='.1%')]
    ).properties(height=300, title="Avg Risk by Gender")
    st.altair_chart(chart_gender, use_container_width=True)

with col_d:
    # Senior citizen mapped for better readability
    dem_df = filtered_df.copy()
    dem_df['SeniorCitizen_Label'] = dem_df['SeniorCitizen'].map({1: 'Senior', 0: 'Non-Senior'})
    senior_risk = dem_df.groupby('SeniorCitizen_Label')['Risk_Probability'].mean().reset_index()
    chart_senior = alt.Chart(senior_risk).mark_bar(opacity=0.8).encode(
        x=alt.X('SeniorCitizen_Label', title="Customer Type"),
        y=alt.Y('Risk_Probability', title="Avg Risk Probability", axis=alt.Axis(format='%')),
        color='SeniorCitizen_Label'
    ).properties(height=300, title="Avg Risk by Seniority")
    st.altair_chart(chart_senior, use_container_width=True)

# 4. HIGH RISK CUSTOMERS TABLE
st.markdown("---")
st.subheader("🚨 Intervention List (High Risk Customers)")
st.write("These customers have a risk probability > 70% and represent immediate revenue flight risk.")

high_risk_df = filtered_df[filtered_df['Risk_Category'] == 'High'].copy()
high_risk_df['Risk_Probability'] = high_risk_df['Risk_Probability'].apply(lambda x: f"{x:.1%}")

display_cols = ['customerID', 'Risk_Probability', 'MonthlyCharges', 'tenure', 'Contract', 'InternetService']
st.dataframe(high_risk_df[display_cols].sort_values(by='MonthlyCharges', ascending=False), use_container_width=True)

st.download_button(
    label="📥 Download Intervention List as CSV",
    data=high_risk_df[display_cols].to_csv(index=False).encode('utf-8'),
    file_name='high_risk_intervention_list.csv',
    mime='text/csv',
)
