import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Mental Health Tech Predictor",
    page_icon="🧠",
    layout="centered"
)

# Load Saved Model, Scaler, and Columns
@st.cache_resource
def load_artifacts():
    model = joblib.load('mental_health_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

# Custom Styling / Header
st.title("🧠 Mental Health Treatment Predictor in Tech")
st.markdown("This application predicts whether an individual in a tech workplace might require mental health treatment based on various work environment and personal factors.")
st.write("---")

# User Input Form Layout
st.subheader("📋 Enter Employee / Workplace Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=75, value=25)
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other/Non-Binary'])
    family_history = st.selectbox('Family History of Mental Illness?', ['Yes', 'No'])
    work_interfere = st.selectbox('Does work interfere with mental health?', ['Often', 'Sometimes', 'Rarely', 'Never', 'Unknown'])

with col2:
    remote_work = st.selectbox('Works Remotely?', ['Yes', 'No'])
    care_options = st.selectbox('Are mental health care options available?', ['Yes', 'No', 'Not sure'])
    benefits = st.selectbox('Does employer provide mental health benefits?', ['Yes', 'No', "Don't know"])
    anonymity = st.selectbox('Is anonymity protected?', ['Yes', 'No', "Don't know"])

st.write("---")

# Prediction Button
if st.button('Predict Treatment Need', use_container_width=True):
    # 1. Create a raw input dictionary matching original data categories
    input_data = {
        'Age': age,
        'Gender': gender,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'remote_work': remote_work,
        'care_options': care_options,
        'benefits': benefits,
        'anonymity': anonymity
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 2. Apply One-Hot Encoding (same as training data)
    input_df_encoded = pd.get_dummies(input_df)
    
    # 3. Reindex to match exact training columns (fill missing columns with 0)
    input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 4. Scale numerical column ('Age') using loaded scaler
    if 'Age' in input_df_encoded.columns:
        input_df_encoded['Age'] = scaler.transform(input_df_encoded[['Age']].values)
        
    # 5. Make Prediction
    prediction = model.predict(input_df_encoded)
    probability = model.predict_proba(input_df_encoded)
    
    # Display Results with UI feedback
    st.subheader("🎯 Prediction Result")
    
    if prediction[0] == 1:
        st.error("⚠️ **The model predicts that mental health treatment may be recommended.**")
        st.metric(label="Confidence Score", value=f"{probability[0][1]*100:.2f}%")
    else:
        st.success("✅ **The model predicts that mental health treatment is unlikely to be needed.**")
        st.metric(label="Confidence Score", value=f"{probability[0][0]*100:.2f}%")