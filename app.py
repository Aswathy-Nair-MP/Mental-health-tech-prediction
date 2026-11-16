import joblib
import pandas as pd
import streamlit as st

# Load the trained model (Ensure model.pkl is in the same folder)
model = joblib.load("model.pkl")

st.title("Mental Health Treatment Prediction App")
st.write(
    "Ee interface use cheythu input details nalki mental health treatment"
    " thevya undo enn predict cheyaam."
)

# Input fields organized cleanly
age = st.slider("Age", 18, 100, 25)

col1, col2 = st.columns(2)

with col1:
  self_employed = st.selectbox("Self Employed?", ["No", "Yes"])
  family_history = st.selectbox(
      "Family History of Mental Illness?", ["No", "Yes"]
  )
  remote_work = st.selectbox("Work Remotely?", ["No", "Yes"])
  tech_company = st.selectbox(
      "Is your company a tech company/startup?", ["No", "Yes"]
  )
  obs_consequence = st.selectbox(
      "Have you heard of or observed negative consequences?", ["No", "Yes"]
  )
  gender = st.selectbox("Gender", ["Male", "Female", "Other/Non-Binary"])
  work_interfere = st.selectbox(
      "Does mental health interfere with work?",
      ["Never", "Rarely", "Sometimes", "Often", "Unknown"],
  )
  no_employees = st.selectbox(
      "Number of employees in company",
      ["6-25", "26-100", "100-500", "500-1000", "More than 1000"],
  )
  benefits = st.selectbox(
      "Does your employer provide mental health benefits?",
      ["No", "Yes", "Don't know"],
  )
  care_options = st.selectbox(
      "Do you know the care options available?", ["No", "Yes", "Not sure"]
  )
  wellness_program = st.selectbox(
      "Has your employer ever discussed mental wellness?", ["No", "Yes"]
  )

with col2:
  seek_help = st.selectbox(
      "Does your employer provide resources to seek help?", ["No", "Yes"]
  )
  anonymity = st.selectbox(
      "Is your anonymity protected?", ["No", "Yes", "Don't know"]
  )
  leave = st.selectbox(
      "How easy is it to take medical leave for mental health?",
      ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult"],
  )
  mental_health_consequence = st.selectbox(
      "Discussing mental health has negative consequences?", ["No", "Yes"]
  )
  phys_health_consequence = st.selectbox(
      "Discussing physical health has negative consequences?", ["No", "Yes"]
  )
  coworkers = st.selectbox(
      "Would you discuss mental health with coworkers?",
      ["No", "Some of them", "Yes"],
  )
  supervisor = st.selectbox(
      "Would you discuss mental health with your supervisor?",
      ["No", "Some of them", "Yes"],
  )
  mental_health_interview = st.selectbox(
      "Would you bring up mental health in an interview?", ["No", "Maybe", "Yes"]
  )
  phys_health_interview = st.selectbox(
      "Would you bring up physical health in an interview?",
      ["No", "Maybe", "Yes"],
  )
  mental_vs_physical = st.selectbox(
      "Employer takes mental health as seriously as physical?",
      ["No", "Yes", "Don't know"],
  )

if st.button("Predict Treatment Requirement"):
  # Map inputs to match model training features
  input_data = {
      "Age": age,
      "self_employed": 1 if self_employed == "Yes" else 0,
      "family_history": 1 if family_history == "Yes" else 0,
      "remote_work": 1 if remote_work == "Yes" else 0,
      "tech_company": 1 if tech_company == "Yes" else 0,
      "obs_consequence": 1 if obs_consequence == "Yes" else 0,
      "Gender_Male": 1 if gender == "Male" else 0,
      "Gender_Other/Non-Binary": 1 if gender == "Other/Non-Binary" else 0,
      "work_interfere_Often": 1 if work_interfere == "Often" else 0,
      "work_interfere_Rarely": 1 if work_interfere == "Rarely" else 0,
      "work_interfere_Sometimes": 1 if work_interfere == "Sometimes" else 0,
      "work_interfere_Unknown": 1 if work_interfere == "Unknown" else 0,
      "no_employees_100-500": 1 if no_employees == "100-500" else 0,
      "no_employees_26-100": 1 if no_employees == "26-100" else 0,
      "no_employees_500-1000": 1 if no_employees == "500-1000" else 0,
      "no_employees_6-25": 1 if no_employees == "6-25" else 0,
      "no_employees_More than 1000": 1 if no_employees == "More than 1000"
      else 0,
      "benefits_No": 1 if benefits == "No" else 0,
      "benefits_Yes": 1 if benefits == "Yes" else 0,
      "care_options_Not sure": 1 if care_options == "Not sure" else 0,
      "care_options_Yes": 1 if care_options == "Yes" else 0,
      "wellness_program_No": 1 if wellness_program == "No" else 0,
      "wellness_program_Yes": 1 if wellness_program == "Yes" else 0,
      "seek_help_No": 1 if seek_help == "No" else 0,
      "seek_help_Yes": 1 if seek_help == "Yes" else 0,
      "anonymity_No": 1 if anonymity == "No" else 0,
      "anonymity_Yes": 1 if anonymity == "Yes" else 0,
      "leave_Somewhat difficult": 1 if leave == "Somewhat difficult" else 0,
      "leave_Somewhat easy": 1 if leave == "Somewhat easy" else 0,
      "leave_Very difficult": 1 if leave == "Very difficult" else 0,
      "leave_Very easy": 1 if leave == "Very easy" else 0,
      "mental_health_consequence_No": 1
      if mental_health_consequence == "No"
      else 0,
      "mental_health_consequence_Yes": 1
      if mental_health_consequence == "Yes"
      else 0,
      "phys_health_consequence_No": 1 if phys_health_consequence == "No" else 0,
      "phys_health_consequence_Yes": 1 if phys_health_consequence == "Yes"
      else 0,
      "coworkers_Some of them": 1 if coworkers == "Some of them" else 0,
      "coworkers_Yes": 1 if coworkers == "Yes" else 0,
      "supervisor_Some of them": 1 if supervisor == "Some of them" else 0,
      "supervisor_Yes": 1 if supervisor == "Yes" else 0,
      "mental_health_interview_No": 1 if mental_health_interview == "No" else 0,
      "mental_health_interview_Yes": 1 if mental_health_interview == "Yes"
      else 0,
      "phys_health_interview_No": 1 if phys_health_interview == "No" else 0,
      "phys_health_interview_Yes": 1 if phys_health_interview == "Yes" else 0,
      "mental_vs_physical_No": 1 if mental_vs_physical == "No" else 0,
      "mental_vs_physical_Yes": 1 if mental_vs_physical == "Yes" else 0,
  }

  input_df = pd.DataFrame([input_data])

  # Align with model feature names if saved in sklearn
  if hasattr(model, "feature_names_in_"):
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

  prediction = model.predict(input_df)

  st.subheader("Prediction Result:")
  if prediction[0] == 1:
    st.error(
        "Result: Treatment is recommended / likely needed based on the"
        " parameters."
    )
  else:
    st.success(
        "Result: Treatment is not required / less likely based on the"
        " parameters."
    )