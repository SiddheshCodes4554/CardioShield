# =========================================================
# 🛡️ CardioShield — Multimodal Cardiovascular AI Dashboard
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="CardioShield",
    page_icon="🫀",
    layout="wide"
)

st.title("🛡️ CardioShield — Cardiovascular Intelligence System")
st.markdown("AI-powered Clinical + ECG Risk Fusion Dashboard")
st.markdown("---")

# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_models():

    model_A = joblib.load("cardio_model_A.pkl")

    ecg_model = load_model(
        "ecg_autoencoder.h5",
        compile=False
    )

    scaler = joblib.load("ecg_scaler.pkl")

    return model_A, ecg_model, scaler


model_A, ecg_model, scaler = load_models()

# =====================================================
# 🔧 ECG PADDING FUNCTION (FIX LENGTH MISMATCH)
# =====================================================

def pad_ecg(signal, target_len):

    current_len = signal.shape[1]

    if current_len < target_len:

        pad_width = target_len - current_len

        padding = np.zeros(
            (signal.shape[0], pad_width)
        )

        signal = np.concatenate(
            [signal, padding],
            axis=1
        )

    return signal


# ---------------- LAYOUT ---------------- #

col1, col2 = st.columns(2)

# =====================================================
# 🩺 CLINICAL RISK PANEL
# =====================================================

with col1:

    st.subheader("🩺 Clinical Risk Assessment")

    age = st.number_input("Age", 1, 100, 45)
    gender = st.selectbox("Gender", [1, 2])
    height = st.number_input("Height (cm)", 140, 210, 170)
    weight = st.number_input("Weight (kg)", 40, 150, 70)

    ap_hi = st.number_input("Systolic BP", 80, 200, 120)
    ap_lo = st.number_input("Diastolic BP", 50, 130, 80)

    cholesterol = st.selectbox("Cholesterol Level", [1,2,3])
    gluc = st.selectbox("Glucose Level", [1,2,3])

    smoke = st.selectbox("Smoking", [0,1])
    alco = st.selectbox("Alcohol Intake", [0,1])
    active = st.selectbox("Physically Active", [0,1])

    clinical_risk = 0

    if st.button("Predict Clinical Risk"):

        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active
        }])

        clinical_risk = model_A.predict_proba(
            input_df
        )[:,1][0]

        st.metric(
            "Clinical Risk Score",
            f"{clinical_risk:.2f}"
        )

        if clinical_risk < 0.3:
            st.success("Low Cardiovascular Risk")

        elif clinical_risk < 0.6:
            st.warning("Moderate Cardiovascular Risk")

        else:
            st.error("High Cardiovascular Risk")

# =====================================================
# 🫀 ECG PANEL
# =====================================================

with col2:

    st.subheader("🫀 ECG Anomaly Detection")

    uploaded_file = st.file_uploader(
        "Upload ECG CSV",
        type=["csv"]
    )

    ecg_risk = 0

    if uploaded_file:

        ecg_data = pd.read_csv(uploaded_file)

        st.write("Uploaded ECG Shape:", ecg_data.shape)

        ecg_data = ecg_data.dropna()

        # Convert to array
        ecg_array = ecg_data.values

        # 🔥 FIX — pad to match scaler features
        ecg_array = pad_ecg(
            ecg_array,
            scaler.n_features_in_
        )

        # Scale
        ecg_scaled = scaler.transform(ecg_array)

        # -------- Segment Signals -------- #

        window_size = 500
        segments = []

        for signal in ecg_scaled:

            num_segments = len(signal) // window_size

            for i in range(num_segments):

                seg = signal[
                    i*window_size :
                    (i+1)*window_size
                ]

                segments.append(seg)

        segments = np.array(segments)

        segments = segments.reshape(
            segments.shape[0],
            segments.shape[1],
            1
        )

        # -------- Autoencoder Inference -------- #

        reconstructions = ecg_model.predict(
            segments,
            verbose=0
        )

        mse = np.mean(
            np.power(
                segments - reconstructions, 2
            ),
            axis=(1,2)
        )

        ecg_risk = float(np.mean(mse))

        st.metric(
            "ECG Anomaly Score",
            f"{ecg_risk:.4f}"
        )

        # -------- Signal Reconstruction -------- #

        st.subheader("Signal Reconstruction")

        i = 0
        original = segments[i].flatten()
        reconstructed = reconstructions[i].flatten()

        fig, ax = plt.subplots(figsize=(10,3))

        ax.plot(original, label="Original")
        ax.plot(reconstructed, label="Reconstructed")

        ax.legend()

        st.pyplot(fig)

        # -------- Heatmap -------- #

        st.subheader("Anomaly Heatmap")

        error = abs(original - reconstructed)

        heatmap = error.reshape(1,-1)

        fig2, ax2 = plt.subplots(figsize=(10,2))

        ax2.imshow(
            heatmap,
            cmap="hot",
            aspect="auto"
        )

        st.pyplot(fig2)

# =====================================================
# 🛡️ FUSION RISK INDEX
# =====================================================

st.markdown("---")
st.subheader("🛡️ CardioShield Fusion Risk Index")

final_risk = (clinical_risk + ecg_risk) / 2

st.metric(
    "Final Cardiovascular Risk",
    f"{final_risk:.2f}"
)

if final_risk < 0.3:
    st.success("Low Overall Risk")

elif final_risk < 0.6:
    st.warning("Moderate Overall Risk")

else:
    st.error("High Overall Risk")
