# Dialysis_Status.py

import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv

# ------------------ Load Environment Variables ------------------
load_dotenv("setup.env", override=True)
GEMINI_API_KEY = os.getenv("AIzaSyBC7rQZBUW35PRDe-yZwiIh6naF5xyF4tE")

# ------------------ Configure Gemini ------------------
if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please check your setup.env file.")
    st.stop()
    
genai.configure(api_key="AIzaSyBC7rQZBUW35PRDe-yZwiIh6naF5xyF4tE")
model_gen = genai.GenerativeModel("gemini-1.5-flash")

# ------------------ Load Trained Model ------------------
MODEL_PATH = "https://github.com/Pavanfagare0023/Chronic_Kidney_Disease_Prediction/Dialysis_Status.pkl"
try:
    with open("https://github.com/Pavanfagare0023/Chronic_Kidney_Disease_Prediction/Dialysis_Status.pkl", 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please ensure it is in the project folder.")
    st.stop()

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="CKD_Dialysis_Prediction", layout="centered")
st.title("Chronic_Kidney_Disease - Dialysis_Prediction")
st.write("Fill in the medical test values to predict the need for dialysis.")

# ------------------ User Input ------------------
def get_user_input():
    return pd.DataFrame([{
        'age': st.slider("Age", 2, 90, 45),
        'bp': st.slider("Blood Pressure", 50, 180, 120),
        'sg': st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025]),
        'al': st.slider("Albumin", 0, 5, 0),
        'su': st.slider("Sugar", 0, 5, 0),
        'rbc': st.selectbox("Red Blood Cells", ["normal", "abnormal"]),
        'pc': st.selectbox("Pus Cell", ["normal", "abnormal"]),
        'pcc': st.selectbox("Pus Cell Clumps", ["present", "notpresent"]),
        'ba': st.selectbox("Bacteria", ["yes", "no"]),
        'bgr': st.slider("Blood Glucose Random", 22, 490, 150),
        'bu': st.slider("Blood Urea", 1, 400, 50),
        'sc': st.slider("Serum Creatinine", 0.4, 76.0, 1.2),
        'sod': st.slider("Sodium", 111, 163, 140),
        'pot': st.slider("Potassium", 2.5, 47.0, 4.5),
        'hemo': st.slider("Hemoglobin", 3.1, 17.8, 12.0),
        'pcv': st.slider("Packed Cell Volume", 9, 54, 38),
        'wc': st.slider("WBC Count (cells/cmm)", 3000, 18000, 8000),
        'rc': st.slider("RBC Count (millions/cmm)", 2.0, 8.0, 4.5),
        'htn': st.selectbox("Hypertension", ["yes", "no"]),
        'dm': st.selectbox("Diabetes Mellitus", ["yes", "no"]),
        'cad': st.selectbox("Coronary Artery Disease", ["yes", "no"]),
        'appet': st.selectbox("Appetite", ["good", "poor"]),
        'pe': st.selectbox("Pedal Edema", ["yes", "no"]),
        'ane': st.selectbox("Anemia", ["yes", "no"])

}])

input_data = get_user_input('age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba,'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo, 'pcv': pcv,'wc': wc, 'rc': rc, 'htn': htn, 'dm': dm, 'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane)


# ------------------ Encode Categorical Features ------------------
for col in input_data.columns:
    if input_data[col].dtype == object:
        le = LabelEncoder()
        input_data[col] = le.fit_transform(input_data[col])

# ------------------ Predict ------------------
if st.button("Predict Dialysis Need"):
    try:
        prediction = model.predict(input_data)[0]
        result_text = "Dialysis Required" if prediction == 1 else "Dialysis Not Required"
        st.subheader(f"Prediction Result: {result_text}")

        # ------------------ Gemini Explanation ------------------
        prompt = f"""
        Based on the following input data:
        {input_data.to_dict(orient='records')[0]}
        Explain in simple medical terms why this patient may or may not need dialysis.
        """
        response = model_gen.generate_content(prompt)
        st.markdown("### Explanation from Gemini:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
