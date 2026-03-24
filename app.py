import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
import os

# =========================
# GROQ CLIENT
# =========================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))# =========================
# AI ANALYSIS FUNCTION
# =========================
def get_ai_analysis(patient_data, prediction, probability, model_name):

    risk = "High Risk" if prediction == 1 else "Low Risk"

    prompt = f"""
You are a medical assistant AI.

Patient Details:
{patient_data.to_string(index=False)}

Model Used: {model_name}
Prediction: {risk}
Risk Probability: {probability}%

Explain:
1. What this result means
2. Possible causes
3. Lifestyle advice
4. When to consult a doctor

Keep it simple and clear.
"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=800,
        top_p=1,
        stream=False
    )

    return completion.choices[0].message.content


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Lung Cancer AI Predictor",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 AI Lung Cancer Risk Predictor")

# =========================
# MODEL SELECTION
# =========================
st.sidebar.title("⚙️ Model Selection")

model_choice = st.sidebar.radio(
    "Choose Model",
    ["CTGAN + Random Forest", "SMOTE + Random Forest"]
)

# Load model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

if model_choice == "CTGAN + Random Forest":
    model = load_model("rf_model.pkl")
    st.markdown("### Model: CTGAN + Random Forest")
else:
    model = load_model("smote_rf_model.pkl")
    st.markdown("### Model: SMOTE + Random Forest")

st.write("Enter patient clinical details to estimate lung cancer risk.")

# =========================
# INPUT FIELDS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 50)

    smoking = st.selectbox("Smoking Habit", [1,2])
    alcohol = st.selectbox("Alcohol Consumption", [1,2])
    peer_pressure = st.selectbox("Peer Pressure", [1,2])
    anxiety = st.selectbox("Anxiety", [1,2])
    fatigue = st.selectbox("Fatigue", [1,2])
    chronic_disease = st.selectbox("Chronic Disease", [1,2])

with col2:
    st.subheader("Symptoms")

    yellow_fingers = st.selectbox("Yellow Fingers", [1,2])
    allergy = st.selectbox("Allergy", [1,2])
    wheezing = st.selectbox("Wheezing", [1,2])
    coughing = st.selectbox("Coughing", [1,2])
    short_breath = st.selectbox("Shortness of Breath", [1,2])
    swallowing = st.selectbox("Swallowing Difficulty", [1,2])
    chest_pain = st.selectbox("Chest Pain", [1,2])

# Encode gender
gender = 1 if gender == "Male" else 0

# Feature array
features = np.array([[gender, age, smoking, yellow_fingers,
                      anxiety, peer_pressure, chronic_disease,
                      fatigue, allergy, wheezing, alcohol,
                      coughing, short_breath, swallowing,
                      chest_pain]])

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Lung Cancer Risk"):

    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    probability_percent = round(prob * 100, 2)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Lung Cancer")
    else:
        st.success("✅ Low Risk of Lung Cancer")

    st.progress(prob)
    st.write(f"### Risk Probability: {probability_percent}%")

    # =========================
    # GAUGE CHART
    # =========================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability_percent,
        title={'text': "Cancer Risk Level"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "red"},
            'steps': [
                {'range':[0,30],'color':"green"},
                {'range':[30,70],'color':"yellow"},
                {'range':[70,100],'color':"red"}
            ]
        }
    ))
    st.plotly_chart(fig)

    # =========================
    # PATIENT SUMMARY
    # =========================
    patient_data = pd.DataFrame({
        "Feature":[
            "Gender","Age","Smoking","Yellow Fingers","Anxiety",
            "Peer Pressure","Chronic Disease","Fatigue","Allergy",
            "Wheezing","Alcohol","Coughing","Short Breath",
            "Swallowing Difficulty","Chest Pain"
        ],
        "Value":[
            gender,age,smoking,yellow_fingers,anxiety,
            peer_pressure,chronic_disease,fatigue,allergy,
            wheezing,alcohol,coughing,short_breath,
            swallowing,chest_pain
        ]
    })

    st.subheader("Patient Data Summary")
    st.dataframe(patient_data)

    # =========================
    # MODEL INFO
    # =========================
    st.subheader("Model Information")

    if model_choice == "CTGAN + Random Forest":
        st.write("""
        This model uses **CTGAN (synthetic data generation)** with a 
        **Random Forest classifier**.
        """)
    else:
        st.write("""
        This model uses **SMOTE** with a 
        **Random Forest classifier**.
        """)

    # =========================
    # 🤖 AI ANALYSIS
    # =========================
    st.subheader("🤖 AI Medical Analysis")

    with st.spinner("Analyzing with AI..."):
        ai_response = get_ai_analysis(
            patient_data,
            prediction[0],
            probability_percent,
            model_choice
        )

    st.write(ai_response)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("About This App")

st.sidebar.write("""
AI-powered lung cancer prediction system.

Features:
- Dual Model Comparison
- Machine Learning prediction
- AI Explanation (Groq)
- Risk visualization
""")

st.sidebar.write("Developer: AI Research Project")

# =========================
# DISCLAIMER
# =========================
st.warning("⚠️ This is an AI-based prediction tool. Not a medical diagnosis. Consult a doctor.")