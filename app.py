import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Prediction System")

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.header("Enter Patient Details")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80,250,120)
chol = st.number_input("Cholesterol",100,600,200)
fbs = st.selectbox("Fasting Blood Sugar", [0,1])
restecg = st.selectbox("Rest ECG",[0,1,2])
thalach = st.number_input("Max Heart Rate",60,220,150)
exang = st.selectbox("Exercise Angina",[0,1])
oldpeak = st.number_input("Old Peak",0.0,10.0,1.0)
slope = st.selectbox("Slope",[0,1,2])
ca = st.selectbox("CA",[0,1,2,3,4])
thal = st.selectbox("Thal",[0,1,2,3])

if st.button("Predict"):

    data = pd.DataFrame([[

        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal

    ]], columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")

st.header("Reports")

st.image("reports/model_accuracy.png")

st.image("reports/correlation_heatmap.png")

st.image("reports/confusion_matrix.png")

st.image("reports/roc_curve.png")

st.image("reports/feature_importance.png")
