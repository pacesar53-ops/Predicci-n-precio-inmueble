import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os
import pydeck as pdk

# Configuración de página
st.set_page_config(
    page_title="Valuación Inmobiliaria AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = "modelo_inmobiliario.pkl"

@st.cache_resource
def cargar_modelo():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

modelo = cargar_modelo()

# --- FORMULARIO EN LA BARRA LATERAL ---
with st.sidebar.form(key="property_form"):
    st.header("⚙️ Parámetros de la Propiedad")

    city_selected = st.selectbox("Ciudad", ["Quito", "Guayaquil", "Manta", "Otra"])

    st.subheader("Especificaciones")
    bedrooms = st.slider("Dormitorios", 1, 10, 3)
    bathrooms = st.slider("Baños", 1, 10, 2)
    parking = st.slider("Estacionamientos", 0, 10, 2)
    area = st.number_input("Área Construida (m²)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)

    st.subheader("Coordenadas")
    lat = st.number_input("Latitud", value=-0.180000, format="%.6f")
    lon = st.number_input("Longitud", value=-78.480000, format="%.6f")

    # Botón único que procesa el formulario
    btn_predict = st.form_submit_button("🚀 Calcular Valuación", type="primary", use_container_width=True)

# --- PANEL PRINCIPAL ---
st.title("🏢 Estimacion de Precio de Inmueble")
st.caption("Aplicando Machine Learning.")

st.divider()

if modelo is None:
    st.error(f"⚠️ No se encontró el archivo del modelo ('{MODEL_PATH}').")
else:
    # Solo ejecuta la predicción y actualización de UI al hacer clic
    if btn_predict:
        data_dict = {
            'BEDROOMS': bedrooms,
            'BATHROOMS': bathrooms,
            'PARKING_SPOTS': parking,
            'CONSTRUCTION_AREA_SQM': area,
            'LATITUDE': lat,
            'LONGITUDE': lon,
            'CITY_Guayaquil': 1 if city_selected == "Guayaquil" else 0,
            'CITY_Manta': 1 if city_selected == "Manta" else 0,
            'CITY_Quito': 1 if city_selected == "Quito" else 0
        }

        input_df = pd.DataFrame([data_dict])
        if hasattr(modelo, "feature_names_in_"):
            input_df = input_df.reindex(columns=modelo.feature_names_in_, fill_value=0)

        try:
            prediccion = modelo.predict(input_df)[0]
            precio_m2 = prediccion / area if area > 0 else 0

            # Métricas Principales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Valor Estimado de Mercado",
                    value=f"${prediccion:,.2f}"
                )
            
            with col2:
                st.metric(
                    label="Precio Promedio por m²",
                    value=f"${precio_m2:,.2f} / m²"
                )

            with col3:
                st.metric(
                    label="Ubicación Seleccionada",
                    value=f"{city_selected}"
                )

            st.divider()

            # Pestañas
            tab_map, tab_details = st.tabs(["🗺️ Mapa Interactivo", "📋 Resumen del Inmueble"])

            with tab_map:
                # Configuración de vista inicial
                view_state = pdk.ViewState(
                    latitude=lat,
                    longitude=lon,
                    zoom=14,
                    pitch=45
                )

                # Capa del marcador (punto rojo)
                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=pd.DataFrame([{'lat': lat, 'lon': lon}]),
                    get_position='[lon, lat]',
                    get_color='[225, 45, 45, 200]',
                    get_radius=80,
                    pickable=True
                )

                # Renderizado con mapa libre (CARTO Light)
                st.pydeck_chart(pdk.Deck(
                    map_provider="carto",
                    map_style="light",
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={"text": f"Ubicación del inmueble\nPrecio Est.: ${prediccion:,.2f}"}
                ))

            with tab_details:
                st.markdown("### Resumen de Variables de Entrada")
                df_resumen = pd.DataFrame({
                    "Atributo": ["Ciudad", "Área Total", "Dormitorios", "Baños", "Estacionamientos", "Coordenadas"],
                    "Valor": [city_selected, f"{area} m²", bedrooms, bathrooms, parking, f"{lat}, {lon}"]
                })
                st.dataframe(df_resumen, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
    else:
        st.info("👈 Ajuste los parámetros de la propiedad en la barra lateral y presione **'🚀 Calcular Valuación'** para generar el reporte.")