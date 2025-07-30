import streamlit as st
import numpy as np
from scipy.special import expit
import joblib
import matplotlib.pyplot as plt
from streamlit_geolocation import streamlit_geolocation
from datetime import datetime
from streamlit_folium import st_folium
import folium




# Load pretrained Bayesian GMM model
gmm = joblib.load("bayesian_gmm.pkl")

st.title("¿Sufres sobrecalentamiento en tu vivienda durante olas de calor en Pamplona, Navarra?")

st.markdown("Responde a las siguientes preguntas para saber si podrías estar en riesgo de sufrir de sobrecalentamiento:")

# User inputs
temp = st.slider("Temperatura del termostato (°C)", 22.0, 31.0, 24.0)
has_ac = st.selectbox("¿Tienes al menos un aparato de aire acondicionado instalado en la casa?", ["Sí", "No"])
hw = st.selectbox("¿Está Pamplona sufriendo una ola de calor?", ["No", "Sí"])
gender = st.selectbox("Sexo", ["Mujer", "Hombre"])
shading = st.selectbox("¿Sientes ahora la necesidad de utilizar dispositivos de sombreado (como toldos, cortinas, persianas)?", ["Sí", "No"])


# Encode inputs
x = [
    temp,
    1 if has_ac == "No" else 0,
    1 if hw == "Sí" else 0,
    1 if gender == "Mujer" else 0,
    1 if shading == "No" else 0
]

# Logistic Regression with updated coefficients
intercept = -19.3390 
coeffs = [0.6400, 2.2284, 0.9783, 1.0670, -2.0731]
linear_combination = intercept + np.dot(coeffs, x)
prob = expit(linear_combination)

# Bayesian GMM
prob_array = np.array([[prob]])
posterior = gmm.predict_proba(prob_array)
cluster = gmm.predict(prob_array)[0]

# Get probabilities for each of the 2 clusters
prob_cluster_0 = posterior[0][0]
prob_cluster_1 = posterior[0][1]

# Definir cuál es el clúster de alto riesgo: asumimos que cluster 1 es alto riesgo (ajusta si necesario)
cluster_info = {
    0: {'label': 'Bajo riesgo de sobrecalentamiento', 'color': 'blue', 'icon': '🟦'},
    1: {'label': 'Alto riesgo de sobrecalentamiento', 'color': 'red', 'icon': '🟥'},
    'intermedio': {'label': 'Riesgo intermedio', 'color': 'orange', 'icon': '⚪'}
}

# Asignar etiqueta basada en la mayor probabilidad, pero considerar solapamiento
confidence_threshold = 0.8  # Puedes ajustar este umbral
if max(prob_cluster_0, prob_cluster_1) < confidence_threshold:
    most_likely_cluster = 'intermedio'
else:
    most_likely_cluster = int(np.argmax([prob_cluster_0, prob_cluster_1]))

risk_label = cluster_info[most_likely_cluster]['icon'] + " " + cluster_info[most_likely_cluster]['label']





# Mostrar resultados
st.subheader("Tu riesgo de sobrecalentamiento:")
st.write(f"Probabilidad estimada de sobrecalentamiento: **{prob:.2f}**")
st.success(f"**Tu perfil pertenece al grupo de: {cluster_info[most_likely_cluster]['icon']} {cluster_info[most_likely_cluster]['label']}**")


cluster_probs = [prob_cluster_0, prob_cluster_1]

with st.expander("Detalles del riesgo"):
    st.markdown("### Probabilidades de pertenencia a cada grupo:")
    
    # Mostrar las probabilidades por grupo
    for i in [0, 1]:
        info = cluster_info[i]
        st.write(f"{info['icon']} {info['label']} (Cluster {i}): **{cluster_probs[i]:.2f}**")
    
    st.write(f"🔶 Nivel de confianza en la clasificación: **{max(cluster_probs):.2f}**")






st.markdown("### Ubicación")
st.write(f"Conocer tu ubicación nos ayuda a entender las diferencias en sobrecalentamiento según barrios, tipologías edificatorias, etc. ")

# Intentar detectar ubicación automáticamente
location = streamlit_geolocation()
lat = location.get("latitude")
lon = location.get("longitude")

if lat is not None and lon is not None:
    st.success(f"Ubicación detectada: Lat {lat:.4f}, Lon {lon:.4f}")
else:
    st.info("Tu navegador debe permitir el acceso a la ubicación.")


# Mostrar mapa si hay coordenadas válidas
if lat is not None and lon is not None:
    st.markdown("#### Mapa de tu ubicación")
    import pandas as pd
    location_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(location_df)



st.markdown("### Comparte tu experiencia térmica actual:")

# Thermal sensation
sensation_labels = {
    -3: "[-3] Muy fría",
    -2: "[-2] Fría",
    -1: "[-1] Ligeramente fría",
     0: "[0] Neutral",
     1: "[+1] Ligeramente caliente",
     2: "[+2] Caliente",
     3: "[+3] Muy caliente"
}
sensation = st.select_slider(
    "¿Cómo sientes la temperatura de la casa ahora?",
    options=list(sensation_labels.keys()),
    value=0,
    format_func=lambda x: sensation_labels[x],
    help="De -3 (mucho frío) a +3 (mucho calor)"
)


# Thermal satisfaction: -3 (very unsatisfied) to +3 (very satisfied)
satisfaction_labels = {
    -3: "[-3] Muy insatisfecho/a",
    -2: "[-2] Insatisfecho/a",
    -1: "[-1] Ligeramente insatisfecho/a",
     0: "[0] Ni satisfecho ni insatisfecho (neutral)",
     1: "[+1] Ligeramente satisfecho/a",
     2: "[+2] Satisfecho/a",
     3: "[+3] Muy satisfecho/a"
}
satisfaction = st.select_slider(
    "¿Qué tan satisfecho/a estás con la temperatura actual de la casa?",
    options=list(satisfaction_labels.keys()),
    value=0,
    format_func=lambda x: satisfaction_labels[x],
    help="De -3 (muy insatisfecho/a) a +3 (muy satisfecho/a)"
)

# Thermal preference: -1 (prefer cooler) to +1 (prefer warmer)
preference_labels = {
    -1: "[-1] Preferiría que fuera más fría",
     0: "[0] Está bien así",
     1: "[+1] Preferiría que fuera más cálida"
}
preference = st.select_slider(
    "¿Preferirías que la temperatura fuera diferente?",
    options=list(preference_labels.keys()),
    value=0,
    format_func=lambda x: preference_labels[x],
    help="Preferencias térmicas según ASHRAE"
)






if st.button("Enviar respuestas"):
    import pandas as pd
    import os

    # Strip icons from label
    risk_label_clean = risk_label.split(" ", 1)[-1]  # Gets the text after the icon

    # Obtener fecha y hora actual
    timestamp = datetime.now().isoformat()


    # Prepare data
    new_data = {
        "timestamp": timestamp,
        "temp_termostato": temp,
        "sin_AC": 1 if has_ac == "No" else 0,
        "ola_calor": 1 if hw == "Sí" else 0,
        "mujer": 1 if gender == "Mujer" else 0,
        "sin_sombreado": 1 if shading == "No" else 0,
        "prob_sobrec": prob,
        "prob_cluster_0": prob_cluster_0,
        "prob_cluster_1": prob_cluster_1,
        "riesgo": cluster_info[most_likely_cluster]['label'],
        "sensacion": sensation,
        "satisfaccion": satisfaction,
        "preferencia": preference,
        "latitud": lat,
        "longitud": lon
    }

    # Convert to DataFrame
    new_df = pd.DataFrame([new_data])

    # Save to file (append if file exists)
    filename = "respuestas_sobrecalentamiento.csv"
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    updated_df.to_csv(filename, index=False)
    st.success("¡Gracias! Tus respuestas han sido guardadas correctamente.")

