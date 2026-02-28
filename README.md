# 📊 Student Productivity Prediction (ML + EDA + Deployment)

## 🚀 Project Overview

This project builds an end-to-end Machine Learning pipeline to predict **Student Productivity Score** based on behavioral and academic features.

It includes:

- Data Cleaning & Preprocessing
- Gender Encoding
- Feature Scaling
- Multiple ML Algorithms
- Hyperparameter Tuning (GridSearchCV)
- Cross-Validation
- Model Evaluation Metrics
- Feature Importance Visualization
- EDA Dashboard Integration
- Streamlit Deployment
- Docker Containerization

---

## 📂 Dataset

File used:
```

student_productivity_distraction_dataset_20000.csv

```

Target Variable:
```

productivity_score

```

Important Features:
- age
- gender
- study_hours_per_day
- sleep_hours
- phone_usage_hours
- social_media_hours
- youtube_hours
- gaming_hours
- breaks_per_day
- coffee_intake_mg
- exercise_minutes
- assignments_completed
- attendance_percentage
- stress_level
- focus_score
- final_grade

---

## 🧠 Machine Learning Workflow

### 🔹 1. Preprocessing
- Drop unnecessary columns (student_id)
- OneHotEncoding for `gender`
- StandardScaler for numeric features
- ColumnTransformer integration

### 🔹 2. Models Applied
- Linear Regression
- Ridge
- Lasso
- Random Forest
- Gradient Boosting
- SVR

### 🔹 3. Hyperparameter Tuning
- GridSearchCV
- 5-Fold Cross Validation
- R² used as scoring metric

### 🔹 4. Evaluation Metrics
- R² Score
- MAE
- RMSE
- Cross-Validation Score

### 🔹 5. Best Model Saved
```

best_model.pkl

````

---

## 📊 EDA Dashboard Features

Integrated inside Streamlit:
- Correlation Heatmap
- Productivity Distribution Plot
- Dynamic UI Inputs

---

## 🖥️ How to Run Locally

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
````

### 2️⃣ Activate Environment

```bash
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train Model

```bash
python train.py
```

### 5️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t student-productivity-app .
```

### Run Container

```bash
docker run -p 8501:8501 student-productivity-app
```

Open:

```
http://localhost:8501
```

---

## 📈 Project Structure

```
├── train.py
├── app.py
├── best_model.pkl
├── student_productivity_distraction_dataset_20000.csv
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Recommended Python Version

```
Python 3.10
```

---

## 👤 Author

Aanjney Kumawat
Machine Learning & Data Science Enthusiast

````
requirements.txt

pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
streamlit==1.35.0
joblib==1.4.2
````

---

# 🔥 Important Note

If you are using:

```
Python 3.14
```

You may face compatibility issues.

Strongly recommended:

```
Python 3.10
```
