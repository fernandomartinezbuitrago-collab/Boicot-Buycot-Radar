%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE MODELO
# ==========================================
st.set_page_config(page_title="Brand Risk & Propensity Radar", page_icon="🛡️", layout="wide")

@st.cache_resource
def cargar_recursos():
    # Cargamos el pipeline entrenado
    path_modelo = '/content/drive/MyDrive/TFM Herramienta/models/modelo_boicot_rf.joblib'
    pipeline_rf = joblib.load(path_modelo)
    
    # Extraemos el modelo y preparamos el explicador SHAP
    modelo_puro = pipeline_rf[-1]
    explainer = shap.TreeExplainer(modelo_puro)
    return pipeline_rf, explainer

try:
    modelo_rf, explainer = cargar_recursos()
    columnas_modelo = modelo_rf[-1].feature_names_in_
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ==========================================
# 2. INTERFAZ DE USUARIO (SIDEBAR)
# ==========================================
st.title("🛡️ Brand Risk & Propensity Radar")
st.markdown("### Motor Predictivo de Riesgo de Boicot con IA Explicable (XAI)")

st.sidebar.header("⚙️ Simulador de Escenarios")

st.sidebar.subheader("1. Variables de Interés Político")
interes_campana = st.sidebar.selectbox("Interés en campañas (V241005)", ["Muy interesado (1)", "Poco interesado (3)", "Nada interesado (5)"])
importa_ganador = st.sidebar.selectbox("Importancia de quién gane (V241210)", ["Le importa mucho (1)", "Le da igual (5)"])

st.sidebar.subheader("2. Polarización Afectiva")
rechazo_dem = st.sidebar.radio("¿Desagrado por Partido Demócrata? (V241171)", ["Sí (1)", "No (5)"])
rechazo_rep = st.sidebar.radio("¿Desagrado por Partido Republicano? (V241175)", ["Sí (1)", "No (5)"])

st.sidebar.subheader("3. Agenda Medioambiental")
cambio_climatico = st.sidebar.selectbox("Impacto del Cambio Climático (V241365)", ["Extremadamente grave (1)", "Moderado (3)", "No es un problema (5)"])

# ==========================================
# 3. LÓGICA DE PROCESAMIENTO
# ==========================================
df_codificado = pd.DataFrame(0, index=[0], columns=columnas_modelo)

def extraer_codigo(texto):
    return int(texto.split("(")[1].split(")")[0])

variables_clave = {
    'V241005': extraer_codigo(interes_campana),
    'V241210': extraer_codigo(importa_ganador),
    'V241171': extraer_codigo(rechazo_dem),
    'V241175': extraer_codigo(rechazo_rep),
    'V241365': extraer_codigo(cambio_climatico)
}

for var, val in variables_clave.items():
    col_str = f"{var}_{val}"
    col_float = f"{var}_{float(val)}"
    if col_str in df_codificado.columns:
        df_codificado[col_str] = 1
    elif col_float in df_codificado.columns:
        df_codificado[col_float] = 1

probabilidad_boicot = modelo_rf.predict_proba(df_codificado)[0][1] * 100

# ==========================================
# 4. VISUALIZACIÓN DE RESULTADOS
# ==========================================
st.markdown("---")
st.subheader("📊 Análisis de Riesgo Predictivo")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="Probabilidad de Boicot", value=f"{probabilidad_boicot:.1f}%")
with col_m2:
    if probabilidad_boicot > 60:
        st.error("ESTADO: RIESGO ALTO 🚨")
    elif probabilidad_boicot > 30:
        st.warning("ESTADO: RIESGO MODERADO ⚠️")
    else:
        st.success("ESTADO: ESTABLE ✅")

# ==========================================
# 5. EXPLICABILIDAD SHAP
# ==========================================
st.markdown("---")
st.subheader("🧠 ¿Por qué este resultado? (Explicabilidad SHAP)")

shap_values_raw = explainer.shap_values(df_codificado)

# Manejo de dimensiones de SHAP (evita fallos de visualización)
if isinstance(shap_values_raw, list):
    valores_grafico = shap_values_raw[1][0]
elif len(np.array(shap_values_raw).shape) == 3:
    valores_grafico = np.array(shap_values_raw)[0, :, 1]
else:
    valores_grafico = np.array(shap_values_raw)[0]

diccionario_nombres = {
    'V241005': 'Interés en campañas', 'V241210': 'Importancia de quién gane',
    'V241171': 'Desagrado Demócrata', 'V241175': 'Desagrado Republicano',
    'V241365': 'Cambio Climático', 'V241178': 'Termómetro: D. Trump',
    'V241609': 'Confianza en Medios', 'V241045': 'Intención de Voto',
    'V241554': 'Confianza Científica', 'V241325': 'Postura sobre Aborto',
    'V241107': 'Identidad Hispana', 'V241247': 'Religiosidad'
}

nombres_amigables = []
for col in columnas_modelo:
    nombre_limpio = col
    for codigo, traduccion in diccionario_nombres.items():
        if col.startswith(codigo + "_"):
            valor_respuesta = col.split('_')[1]
            nombre_limpio = f"{traduccion} (Resp: {valor_respuesta})"
    nombres_amigables.append(nombre_limpio)

fig, ax = plt.subplots(figsize=(10, 4))
shap.bar_plot(valores_grafico, feature_names=nombres_amigables, max_display=10, show=False)
st.pyplot(fig)

st.info("💡 **Interpretación:** Las barras rojas (derecha) incrementan el riesgo; las azules (izquierda) lo mitigan.")

st.info("💡 Interpretación: Las barras hacia la derecha (rojo) aumentan el riesgo de que este perfil haga boicot, hacia la izquierda (azul) lo disminuyen.")
