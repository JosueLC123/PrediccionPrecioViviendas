import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="🏡 Predicción de Precio de Viviendas", layout="wide")

st.title("🏡 Predicción del Precio de una Vivienda")
st.markdown("Esta aplicación utiliza un **modelo de Regresión Lineal** para predecir el **precio estimado** de una vivienda.")

# ============================
# CARGA DEL MODELO CORRECTO
# ============================
try:
    with open("modeloPrecioVivienda.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ No se encontró el archivo modeloPrecioVivienda.pkl. Súbelo a la misma carpeta del proyecto.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

st.subheader("📋 Ingrese los datos de la vivienda")

# Entradas del usuario
tamanio = st.number_input("Tamaño (m²)", min_value=20, max_value=500, value=80)
cuartos = st.number_input("Número de cuartos", min_value=1, max_value=10, value=3)
banos = st.number_input("Número de baños", min_value=1, max_value=10, value=2)
ofertas = st.number_input("Número de ofertas", min_value=0, max_value=50, value=0)

if st.button("💰 Predecir Precio"):
    input_data = pd.DataFrame({
        'tamanio_m2': [tamanio],
        'cuartos': [cuartos],
        'banos': [banos],
        'ofertas': [ofertas]
    })

    prediccion = model.predict(input_data)[0]

    st.success(f"🏷️ Precio estimado: **S/. {prediccion:,.2f}**")

