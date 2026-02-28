import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("📊 Student Productivity Prediction App")

# Load model
model = joblib.load("best_model.pkl")

# Sidebar Inputs
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", 18, 40, 22)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
study_hours = st.sidebar.slider("Study Hours Per Day", 0.0, 15.0, 5.0)
sleep = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0)
phone = st.sidebar.slider("Phone Usage Hours", 0.0, 10.0, 3.0)
stress = st.sidebar.slider("Stress Level", 1, 10, 5)
focus = st.sidebar.slider("Focus Score", 1, 10, 6)

input_data = pd.DataFrame({
    "age":[age],
    "gender":[gender],
    "study_hours_per_day":[study_hours],
    "sleep_hours":[sleep],
    "phone_usage_hours":[phone],
    "social_media_hours":[2],
    "youtube_hours":[2],
    "gaming_hours":[1],
    "breaks_per_day":[3],
    "coffee_intake_mg":[100],
    "exercise_minutes":[30],
    "assignments_completed":[5],
    "attendance_percentage":[85],
    "stress_level":[stress],
    "focus_score":[focus],
    "final_grade":[75]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Productivity Score: {round(prediction[0],2)}")

# -----------------------------
# EDA Dashboard
# -----------------------------
st.header("📊 EDA Dashboard")

df = pd.read_csv(r"student_productivity_distraction_dataset_20000.csv")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
    st.pyplot(fig)

with col2:
    st.subheader("Productivity Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["productivity_score"], kde=True)

    st.pyplot(fig2)
