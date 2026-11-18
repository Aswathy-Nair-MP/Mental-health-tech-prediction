import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MindMetrics AI | Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .hero-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.15);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-bottom: 0;
    }
    .section-header {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-card-risk {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 1.5rem;
        border-radius: 10px;
        color: #7f1d1d;
    }
    .result-card-safe {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 1.5rem;
        border-radius: 10px;
        color: #14532d;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.65rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #1d4ed8 100%, #1e40af 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD SAVED MODEL, SCALER & COLUMNS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("mental_health_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

try:
    model, scaler, model_columns = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files. Please ensure `.pkl` files are in the same folder as `app.py`. Details: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR: ABOUT PROJECT ONLY
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 MindMetrics AI")
    st.markdown("---")
    st.markdown("### About the Project")
    st.info(
        "**MindMetrics AI** is an intelligent machine learning web application designed to "
        "evaluate workplace mental health dynamics and predict the likelihood of an individual "
        "requiring mental health support or treatment.\n\n"
        "By analyzing demographic factors, work environment, organizational benefits, and "
        "workplace stigma indicators using a trained **Random Forest Classifier**, the system "
        "delivers instant, data-driven insights to promote proactive well-being."
    )
    st.caption("⚠️ *Disclaimer: This tool is for educational and informational screening purposes only and is not a substitute for professional medical advice.*")

# -----------------------------------------------------------------------------
# 4. MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="hero-box">
        <div class="hero-title">🧠 MindMetrics AI</div>
        <div class="hero-subtitle">Workplace Mental Health Risk & Support Assessment System</div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. USER INPUT FORM
# -----------------------------------------------------------------------------
with st.form("assessment_form"):
    # --- Section 1: Personal & Work Profile ---
    st.markdown('<div class="section-header">1. Personal & Work Profile</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other/Non-Binary"])
    
    with col2:
        self_employed = st.selectbox("Are you self-employed?", ["No", "Yes"])
        family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])
    
    with col3:
        remote_work = st.selectbox("Do you work remotely (50%+ of the time)?", ["No", "Yes"])
        tech_company = st.selectbox("Is your employer primarily a tech company?", ["Yes", "No"])

    # --- Section 2: Work Interference & Company Size ---
    st.markdown('<div class="section-header">2. Work Impact & Organization Details</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    
    with col4:
        work_interfere = st.selectbox(
            "Does mental health interfere with your work?",
            ["Never", "Rarely", "Sometimes", "Often", "Unknown"]
        )
    with col5:
        no_employees = st.selectbox(
            "Company Size (Number of Employees)",
            ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
        )
    with col6:
        obs_consequence = st.selectbox(
            "Observed negative consequences for coworkers with mental health conditions?",
            ["No", "Yes"]
        )

    # --- Section 3: Workplace Benefits & Support ---
    st.markdown('<div class="section-header">3. Workplace Mental Health Resources</div>', unsafe_allow_html=True)
    col7, col8, col9 = st.columns(3)
    
    with col7:
        benefits = st.selectbox("Does your employer provide mental health benefits?", ["Yes", "No", "Don't know"])
        care_options = st.selectbox("Do you know the options for mental health care provided?", ["Yes", "No", "Not sure"])
    
    with col8:
        wellness_program = st.selectbox("Has your employer discussed mental health in a wellness program?", ["No", "Yes", "Don't know"])
        seek_help = st.selectbox("Does your employer provide resources to seek help?", ["Yes", "No", "Don't know"])
    
    with col9:
        anonymity = st.selectbox("Is your anonymity protected if you take treatment?", ["Yes", "No", "Don't know"])
        leave = st.selectbox(
            "How easy is it to take medical leave for a mental health condition?",
            ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]
        )

    # --- Section 4: Workplace Culture & Communication ---
    st.markdown('<div class="section-header">4. Workplace Culture & Stigma</div>', unsafe_allow_html=True)
    col10, col11, col12 = st.columns(3)
    
    with col10:
        mental_health_consequence = st.selectbox("Negative consequence if discussing mental health with employer?", ["No", "Maybe", "Yes"])
        phys_health_consequence = st.selectbox("Negative consequence if discussing physical health with employer?", ["No", "Maybe", "Yes"])
    
    with col11:
        coworkers = st.selectbox("Would you discuss a mental health issue with coworkers?", ["Some of them", "No", "Yes"])
        supervisor = st.selectbox("Would you discuss a mental health issue with your direct supervisor?", ["Yes", "No", "Some of them"])
    
    with col12:
        mental_health_interview = st.selectbox("Would you bring up a mental health issue in an interview?", ["No", "Maybe", "Yes"])
        phys_health_interview = st.selectbox("Would you bring up a physical health issue in an interview?", ["Maybe", "No", "Yes"])
        mental_vs_physical = st.selectbox("Does employer take mental health as seriously as physical health?", ["Don't know", "Yes", "No"])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Analyze Mental Health Risk")

# -----------------------------------------------------------------------------
# 6. PREPROCESSING & PREDICTION LOGIC
# -----------------------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame(0.0, index=[0], columns=model_columns)

    # 1. Numeric & Binary Columns
    input_data["Age"] = float(age)
    input_data["self_employed"] = 1.0 if self_employed == "Yes" else 0.0
    input_data["family_history"] = 1.0 if family_history == "Yes" else 0.0
    input_data["remote_work"] = 1.0 if remote_work == "Yes" else 0.0
    input_data["tech_company"] = 1.0 if tech_company == "Yes" else 0.0
    input_data["obs_consequence"] = 1.0 if obs_consequence == "Yes" else 0.0

    # 2. One-Hot Encoded Columns mapping
    ohe_mappings = {
        f"Gender_{gender}": 1.0,
        f"work_interfere_{work_interfere}": 1.0,
        f"no_employees_{no_employees}": 1.0,
        f"benefits_{benefits}": 1.0,
        f"care_options_{care_options}": 1.0,
        f"wellness_program_{wellness_program}": 1.0,
        f"seek_help_{seek_help}": 1.0,
        f"anonymity_{anonymity}": 1.0,
        f"leave_{leave}": 1.0,
        f"mental_health_consequence_{mental_health_consequence}": 1.0,
        f"phys_health_consequence_{phys_health_consequence}": 1.0,
        f"coworkers_{coworkers}": 1.0,
        f"supervisor_{supervisor}": 1.0,
        f"mental_health_interview_{mental_health_interview}": 1.0,
        f"phys_health_interview_{phys_health_interview}": 1.0,
        f"mental_vs_physical_{mental_vs_physical}": 1.0,
    }

    for col_name, val in ohe_mappings.items():
        if col_name in input_data.columns:
            input_data[col_name] = val

    # 3. Scale 'Age' using the saved StandardScaler
    input_data[["Age"]] = scaler.transform(input_data[["Age"]])

    # 4. Predict
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    treatment_prob = probabilities[1] * 100

    # -------------------------------------------------------------------------
    # 7. DISPLAY RESULTS
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Assessment Result")
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
                <div class="result-card-risk">
                    <h3 style="margin-top:0;">⚠️ Elevated Risk — Support / Treatment Recommended</h3>
                    <p style="font-size:1.05rem; margin-bottom:0.5rem;">
                        Based on your responses, <strong>MindMetrics AI</strong> indicates a high likelihood 
                        (<strong>{treatment_prob:.1f}%</strong>) that you may benefit from mental health support or treatment.
                    </p>
                    <p style="margin-bottom:0; font-size:0.95rem;">
                        Consider exploring your organization's wellness resources, employee assistance programs (EAP), 
                        or consulting a licensed mental health professional.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-safe">
                    <h3 style="margin-top:0;">✅ Low Risk — Healthy Workplace Indicator</h3>
                    <p style="font-size:1.05rem; margin-bottom:0.5rem;">
                        Based on your responses, <strong>MindMetrics AI</strong> indicates a lower likelihood 
                        (<strong>{treatment_prob:.1f}%</strong> risk score) of requiring immediate mental health treatment.
                    </p>
                    <p style="margin-bottom:0; font-size:0.95rem;">
                        Continue maintaining healthy work-life boundaries and utilizing workplace well-being support whenever needed.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.metric(label="Predicted Support Probability", value=f"{treatment_prob:.1f}%")
        st.progress(int(treatment_prob))