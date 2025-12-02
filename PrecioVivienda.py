import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ===========================
# CONFIGURACIÓN DE LA PÁGINA
# ===========================
st.set_page_config(page_title="🏡 Predicción de Precio de Viviendas", layout="wide")

st.title("🏡 Predicción del Precio de una Vivienda")
st.markdown("""
Esta aplicación utiliza un modelo de **Regresión Lineal** entrenado para predecir el **precio estimado**
de una vivienda usando características como:
- Tamaño en pies cuadrados  
- Número de cuartos  
- Número de baños  
- Número de ofertas  
""")

# ===========================
# CARGA DEL MODELO
# ===========================
try:
    with open("modelo_vivienda.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ No se encontró el archivo **modeloPrecioVivienda.pkl**. Súbelo a la misma carpeta.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ===========================
# SIDEBAR – ENTRADA DE DATOS
# ===========================
st.sidebar.header("📌 Ingrese los datos de la vivienda")

pies = st.sidebar.number_input("Pies cuadrados", min_value=200, max_value=10000, value=1800)
cuartos = st.sidebar.number_input("Cuartos", min_value=1, max_value=10, value=3)
banos = st.sidebar.number_input("Baños", min_value=1, max_value=10, value=2)
ofertas = st.sidebar.number_input("Ofertas recibidas", min_value=0, max_value=20, value=2)

# Crear dataframe del input
input_data = pd.DataFrame({
    'Piescuad': [pies],
    'Cuartos': [cuartos],
    'Baños': [banos],
    'Ofertas': [ofertas]
})

# ===========================
# BOTÓN DE PREDICCIÓN
# ===========================
if st.sidebar.button("🔍 Predecir precio"):

    try:
        prediccion = model.predict(input_data)[0]
        prediccion = round(prediccion, 2)
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.stop()

    # Mostrar tarjeta elegante
    st.markdown("""
        <h3>🏠 Precio Estimado</h3>
        """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style='background-color:#4CAF50;padding:20px;border-radius:10px;color:white;text-align:center'>
            <h2>💲 {prediccion:,.2f}</h2>
            <p>Precio aproximado según las características ingresadas</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===========================
    # GRÁFICO DE ENTRADAS
    # ===========================
    st.markdown("### 📊 Características ingresadas")
    fig, ax = plt.subplots()
    ax.bar(input_data.columns, input_data.iloc[0])
    ax.set_title("Valores ingresados")
    st.pyplot(fig)


# ===========================
# SECCIÓN: VISUALIZACIÓN DEL MODELO
# ===========================
st.header("📈 Visualización del Modelo de Regresión Lineal")

coef = model.coef_
intercepto = model.intercept_
features = ['Piescuad', 'Cuartos', 'Baños', 'Ofertas']

st.subheader("📌 Coeficientes del modelo")
coef_df = pd.DataFrame({
    "Variable": features,
    "Coeficiente": coef
})
st.table(coef_df)

st.success(f"**Intercepto:** {intercepto:,.2f}")

# Gráfico de importancia de variables
st.subheader("📊 Importancia de las Variables (Coeficientes)")

fig2, ax2 = plt.subplots()
ax2.bar(features, coef)
ax2.set_title("Importancia de cada variable en la predicción")
ax2.set_ylabel("Valor del Coeficiente")
st.pyplot(fig2)

# ===========================
# SECCIÓN: COMPARACIÓN REAL VS PREDICHO
# ===========================

st.header("📊 Comparación Real vs Predicho")

uploaded = st.file_uploader("Sube el archivo con tus datos originales (para mostrar comparación)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Verifica columnas mínimas
    if all(col in df.columns for col in ["Piescuad", "Cuartos", "Baños", "Ofertas", "Precio"]):

        X = df[["Piescuad", "Cuartos", "Baños", "Ofertas"]]
        y_real = df["Precio"]
        y_pred = model.predict(X)

        df_compare = pd.DataFrame({
            "Precio Real": y_real,
            "Precio Predicho": y_pred
        }).head(20)

        st.subheader("📄 Tabla (primeros 20 valores)")
        st.dataframe(df_compare)

        # Gráfico comparativo
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(df_compare["Precio Real"].values, label="Real")
        ax3.plot(df_compare["Precio Predicho"].values, label="Predicho")
        ax3.set_title("📈 Real vs Predicho (primeros 20)")
        ax3.legend()
        st.pyplot(fig3)

    else:
        st.error("El archivo debe contener: Piescuad, Cuartos, Baños, Ofertas y Precio.")
