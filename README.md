# 🧠 MindMetrics AI-Mental Health Predictor in Tech Workplace

A machine learning web application that predicts and analyzes mental health treatment needs and workplace interference based on employee survey data. Built with Python, Scikit-Learn, and Streamlit.

---

##  About the Project
This project utilizes the **Mental Health in Tech Survey** dataset to explore, clean, and model factors influencing mental health treatment seeking behavior in tech work environments. An end-to-end Machine Learning pipeline is built, trained using a tuned **Random Forest Classifier**, and deployed interactively via **Streamlit**.

---

##  Dataset & EDA Highlights
* **Source Dataset:** Mental Health in Tech Survey (`survey.csv`)
* **Initial Shape:** 1,259 rows and 27 columns[cite: 9]
* **Data Cleaning & Preprocessing:**
  * Cleaned the `Age` column by removing invalid outliers (ages < 18 or > 100) and dropping missing rows[cite: 9].
  * Standardized and mapped messy entries in the `Gender` column into three primary categories: `Male`, `Female`, and `Other/Non-Binary`[cite: 9].
  * Handled missing values and encoded categorical variables for machine learning compatibility.

---

##  Tech Stack & Libraries
* **Programming Language:** Python[cite: 9]
* **Data Analysis & Manipulation:** Pandas, NumPy[cite: 9]
* **Data Visualization:** Seaborn, Matplotlib[cite: 9]
* **Machine Learning:** Scikit-Learn (Random Forest, Standardizing, Encoding)
* **Web Framework / Deployment:** Streamlit
* **Development Environment:** VS Code, Jupyter Notebook

---

##  Project Structure
```text
├── survey.csv              # Original dataset
├── mental_health_model.pkl # Trained Random Forest model
├── scaler.pkl              # Feature standard scaler
├── columns.pkl             # Model feature columns mapping
├── app.py                  # Streamlit web application interface
└── README.md               # Project documentation
