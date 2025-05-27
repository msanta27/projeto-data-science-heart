import streamlit as st
import pandas as pd
import joblib

# Carrega o modelo e o pré-processador
modelo = joblib.load("modelo_final.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("💓 Previsão de Doença Cardíaca")
st.markdown("Preencha os campos abaixo para saber o risco:")

# Entradas do usuário
age = st.slider("Idade", 25, 80, 50)
sex = st.selectbox("Sexo", ["M", "F"])
chest_pain = st.selectbox("Tipo de Dor no Peito", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Pressão em repouso", 80, 200, 120)
chol = st.number_input("Colesterol", 0, 600, 200)
fasting_bs = st.selectbox("Glicose em jejum > 120 mg/dl?", [0, 1])
resting_ecg = st.selectbox("ECG em repouso", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Frequência cardíaca máxima", 60, 220, 150)
ex_angina = st.selectbox("Angina induzida por exercício", ["Y", "N"])
oldpeak = st.number_input("Depressão ST", 0.0, 6.0, 0.0)
st_slope = st.selectbox("Inclinação ST", ["Up", "Flat", "Down"])

# Botão para previsão
if st.button("🔍 Avaliar"):
    dados = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": chol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": ex_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    # Derivar atributo extra igual ao notebook
    dados["age_over_50"] = dados["Age"].apply(lambda x: 1 if x > 50 else 0)

    # Pré-processar
    X_transf = preprocessor.transform(dados)

    # Prever
    pred = modelo.predict(X_transf)[0]
    prob = modelo.predict_proba(X_transf)[0][1]

    if pred == 1:
        st.error(f"🚨 Risco detectado! Probabilidade: {prob:.2%}")
    else:
        st.success(f"✅ Sem risco detectado. Probabilidade: {prob:.2%}")
