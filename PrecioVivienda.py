import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="🏡 Predicción de Precio de Viviendas", layout="wide")

st.title("🏡 Predicción del Precio de una Vivienda")
st.markdown("Esta aplicación utiliza un **modelo de Regresión Lineal** entrenado para predecir el **precio estimado** de una vivienda usando características como tamaño, cantidad de cuartos, baños y ofertas.")

# ============================
# CARGA DEL MODELO
# ============================
try:
    with open("model4.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ No se encontró el archivo model4.pkl. Asegúrate de subirlo o colocarlo en la misma carpeta que app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ============================
# SIDEBAR — ENTRADA DE DATOS
# ============================
st.sidebar.header("📋 Ingrese las características")

pies = st.sidebar.number_input("Pies cuadrados:", min_value=200, max_value=10000, value=1500, step=50)
cuartos = st.sidebar.number_input("Número de cuartos:", min_value=1, max_value=10, value=3)
banos = st.sidebar.number_input("Número de baños:", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
ofertas = st.sidebar.number_input("Número de ofertas:", min_value=1, max_value=10, value=1)

# DataFrame de entrada
input_df = pd.DataFrame({
    "Piescuad": [pies],
    "Cuartos": [cuartos],
    "Baños": [banos],
    "Ofertas": [ofertas]
})

# ============================
# BOTÓN DE PREDICCIÓN
# ============================
if st.sidebar.button("🔍 Predecir Precio"):
    prediccion = model.predict(input_df)[0]
    prediccion = round(prediccion, 2)

    st.markdown(
        f"""
        <div style='background-color:#1b4332; padding:15px; border-radius:8px; text-align:center; color:white; font-size:22px;'>
        💰 <b>Precio estimado de la vivienda: ${prediccion:,.2f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ============================
    # VALIDACIÓN DE DATOS INGRESADOS
    # ============================
    st.subheader("📄 Datos utilizados en la predicción")

    st.table(input_df)

# ============================
# SECCIÓN DE ANÁLISIS (Opcional)
# ============================
st.subheader("📊 Panel de Análisis")

uploaded_file = st.file_uploader("📥 Sube un CSV para análisis (opcional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Vista previa del dataset")
    st.dataframe(df.head())

    st.write("### Estadísticas descriptivas")
    st.write(df.describe())

    # Distribuciones
    st.write("### Distribuciones")
    fig, ax = plt.subplots()
    df.hist(ax=ax)
    st.pyplot(fig)

# ============================
# IMPORTANCIA DE VARIABLES
# ============================
st.subheader("📌 Importancia de características (Regresión Lineal)")

try:
    coef = model.coef_
    variables = ["Piescuad", "Cuartos", "Baños", "Ofertas"]

    importancia_df = pd.DataFrame({
        "feature": variables,
        "coef": coef
    })

    st.table(importancia_df)
except:
    st.warning("No se pudieron mostrar los coeficientes del modelo.")
