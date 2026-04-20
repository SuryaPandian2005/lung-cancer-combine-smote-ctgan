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
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)# =========================
# AI ANALYSIS FUNCTION
# =========================
def get_ai_analysis(patient_data, prediction, probability, model_name):

    risk = "High Risk" if prediction == 1 else "Low Risk"

    prompt = f"""
You are a professional medical assistant.

Patient Data:
{patient_data.to_string(index=False)}

Prediction: {risk}
Risk: {probability}%

Give ONLY practical output in the exact format below:

🧠 Result Summary:
(Provide 1–2 short lines explaining the risk level, what it means for the patient, and overall health implication in simple terms)

⚠️ Key Risk Factors:
- (List up to 5 important risk factors based on patient data, habits, or symptoms)
- (Include likely causes or contributing lifestyle factors if relevant)
- (Keep each point short and clear)

✅ What To Do Now:
- (Provide clear, step-by-step practical actions the patient can follow immediately)
- (Include lifestyle improvements, precautions, or healthy habits)
- (Keep instructions simple and realistic)

🏥 When To See Doctor:
- (Mention specific symptoms, warning signs, or conditions that require medical attention)
- (Be clear about urgency if needed, e.g., “seek immediate care if…”)

Rules:
- Keep the response SHORT and structured
- Use bullet points where required
- Do NOT use medical jargon or complex terms
- Use simple, everyday language
- Be supportive and calm, not alarming
- Base all points on the given data (do not hallucinate unknown conditions)
- Do NOT include any text outside this format
- Do NOT add extra sections or explanations
🎯 Why this is better
✅ More professional tone
✅ Prevents AI hallucination
✅ Ensures consistent structured output
✅ Keeps it safe for medical-style UI
✅ Improves real-world usability
🔥 Bonus (optional upgrade)

If you want even stronger output control, add this line:

- If risk is low, focus more on prevention
- If risk is high, focus more on caution and medical consultation
🚀 Result

Your AI will now behave like:

🧑‍⚕️ Medical assistant
📊 Risk interpreter
🧾 Structured report generator

If you want next level:
👉 I can convert this into JSON + UI auto-render (very powerful for research/demo)
"""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=500,
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
@st.cache_resource
def load_ctgan():
    try:
        return joblib.load("ctgan_model.pkl")
    except:
        return None


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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Patient Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 50)

    smoking = st.selectbox("Smoking Habit", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    alcohol = st.selectbox("Alcohol Consumption", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    peer_pressure = st.selectbox("Peer Pressure", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    anxiety = st.selectbox("Anxiety", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    fatigue = st.selectbox("Fatigue", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    chronic_disease = st.selectbox("Chronic Disease", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 Symptoms")

    yellow_fingers = st.selectbox("Yellow Fingers", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    allergy = st.selectbox("Allergy", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    wheezing = st.selectbox("Wheezing", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    coughing = st.selectbox("Coughing", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    short_breath = st.selectbox("Shortness of Breath", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    swallowing = st.selectbox("Swallowing Difficulty", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
    chest_pain = st.selectbox("Chest Pain", [1, 2], format_func=lambda x: "No" if x == 1 else "Yes")

    st.markdown('</div>', unsafe_allow_html=True)
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
st.markdown("---")
predict_btn = st.button("🚀 Analyze Lung Cancer Risk")

if predict_btn:

    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    probability_percent = round(prob * 100, 2)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Lung Cancer")
    else:
        st.success("✅ Low Risk of Lung Cancer")

    st.progress(prob)
    st.write(f"### Risk Probability: {probability_percent}%")

    st.markdown('</div>', unsafe_allow_html=True)

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
# =========================
# 📌 SIDEBAR
# =========================
st.sidebar.title("🫁 Lung Cancer AI Dashboard")

st.sidebar.markdown("---")

# 🔍 About
st.sidebar.subheader("📖 About This App")
st.sidebar.write("""
This application uses **Machine Learning + AI** to estimate the risk of lung cancer 
based on patient clinical data and symptoms.

It provides:
- Instant risk prediction
- AI-powered medical insights
- Visual risk analysis
""")

# ⚙️ Features
st.sidebar.subheader("⚙️ Features")
st.sidebar.write("""
- 🔄 Dual Model Comparison (CTGAN & SMOTE)
- 🤖 AI Medical Explanation (Groq)
- 📊 Risk Visualization (Gauge Chart)
- 🧾 Patient Data Summary
""")

# 🧠 How it works
st.sidebar.subheader("🧠 How It Works")
st.sidebar.write("""
1. Enter patient details  
2. Click **Analyze Risk**  
3. Model predicts probability  
4. AI explains the result  
""")

# 🛑 Disclaimer
st.sidebar.subheader("⚠️ Disclaimer")
st.sidebar.warning("""
This is an AI-based prediction tool.  
It is **not a medical diagnosis**.  
Always consult a qualified doctor.
""")

# 👨‍💻 Developer
st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Developer")
st.sidebar.write("AI Research Project")
st.sidebar.write("Built with ❤️ using Streamlit")

# 📬 Optional Contact
st.sidebar.subheader("📬 Contact")
st.sidebar.write("For queries or collaboration:")
st.sidebar.write("📧 your-lungcancerprediction@gmail.com")
