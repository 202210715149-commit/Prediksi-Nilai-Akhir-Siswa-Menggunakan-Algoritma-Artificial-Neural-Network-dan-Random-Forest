import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.title("🎓 Prediksi Nilai Akhir Siswa (G3)")
st.write("Aplikasi ini menggunakan 2 algoritma Machine Learning: ANN & Random Forest")

# ================================
# Load Models
# ================================
ann_model = load_model("model_ann_student_mat.h5")
rf_model = joblib.load("random_forest_student_mat.pkl")
scaler = joblib.load("scaler.pkl")

# ================================
# Input Features
# ================================

st.subheader("Masukkan Input Fitur Siswa")

# Daftar fitur (HARUS sesuai urutan training X!)
feature_names = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
    'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',

    # Fitur kategorik one-hot
    'school_MS', 'sex_M', 'address_U', 'famsize_LE3', 'Pstatus_T',
    'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
    'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
    'reason_home', 'reason_other', 'reason_reputation',
    'guardian_mother', 'guardian_other',
    'schoolsup_yes', 'famsup_yes', 'paid_yes', 'activities_yes', 'nursery_yes',
    'higher_yes', 'internet_yes', 'romantic_yes'
]

inputs = []

for feature in feature_names:
    val = st.number_input(feature, value=0.0, step=1.0)
    inputs.append(val)

inputs = np.array(inputs).reshape(1, -1)

# ================================
# Predict
# ================================
st.subheader("Pilih Algoritma")

algo = st.selectbox("Algoritma:", ["Artificial Neural Network (ANN)", "Random Forest"])

if st.button("Prediksi"):
    if algo == "Artificial Neural Network (ANN)":
        X_scaled = scaler.transform(inputs)
        pred = ann_model.predict(X_scaled)[0][0]
    else:
        pred = rf_model.predict(inputs)[0]

    st.success(f"🎯 Hasil Prediksi G3: **{pred:.2f}**")
