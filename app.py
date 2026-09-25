import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# ---------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Detector de Posición",
    page_icon="🕹️",
    layout="centered"
)

# ---------------------------------------------------------
# Estilos decorativos (CSS embebido)
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 100%);
    }
    .title-container {
        text-align: center;
        padding: 10px 0 0 0;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 16px;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .feedback-text {
        text-align: center;
        font-size: 17px;
        font-weight: 600;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cabecera
# ---------------------------------------------------------
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.title("🕹️ Detector de Posición")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Reconocimiento de posiciones con un modelo entrenado en Teachable Machine</p>',
    unsafe_allow_html=True
)

st.write("🐍 Versión de Python:", platform.python_version())

# ---------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# ---------------------------------------------------------
# Imagen principal e info lateral
# ---------------------------------------------------------
image = Image.open('manscreen.jpg')
st.image(image, width=350)

with st.sidebar:
    st.subheader("ℹ️ Acerca de esta app")
    st.write(
        "Usando un modelo entrenado en **Teachable Machine**, "
        "esta app identifica la posición que muestres frente a la cámara. 📸"
    )
    st.write("Posiciones que reconoce:")
    st.markdown("- ⬅️ Izquierda\n- ⬆️ Arriba")

# ---------------------------------------------------------
# Estado para guardar la predicción y el feedback
# ---------------------------------------------------------
if "ultima_prediccion" not in st.session_state:
    st.session_state.ultima_prediccion = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None

# ---------------------------------------------------------
# Captura de foto
# ---------------------------------------------------------
st.markdown("### 📷 Toma una foto para detectar tu posición")
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalización
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Inferencia
    prediction = model.predict(data)
    print(prediction)

    resultado = None
    emoji = ""
    probabilidad = 0.0

    if prediction[0][0] > 0.5:
        resultado = "Izquierda"
        emoji = "⬅️"
        probabilidad = prediction[0][0]
    if prediction[0][1] > 0.5:
        resultado = "Arriba"
        emoji = "⬆️"
        probabilidad = prediction[0][1]
    #if prediction[0][2] > 0.5:
    #    resultado = "Derecha"
    #    emoji = "➡️"
    #    probabilidad = prediction[0][2]

    # Guardamos la predicción actual en el estado (y reseteamos feedback previo)
    if resultado != st.session_state.ultima_prediccion:
        st.session_state.feedback = None
    st.session_state.ultima_prediccion = resultado

    # ---------------------------------------------------------
    # Resultado decorado
    # ---------------------------------------------------------
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    if resultado:
        st.markdown(f"## {emoji} {resultado}")
        st.progress(min(float(probabilidad), 1.0))
        st.write(f"Probabilidad: **{probabilidad:.2%}**")
    else:
        st.markdown("## 🤔 No se detectó ninguna posición clara")
        st.write("Intenta acercarte más a la cámara o mejorar la iluminación.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Feedback del usuario
    # ---------------------------------------------------------
    st.markdown("### ¿Esta predicción está correcta?")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Sí"):
            st.session_state.feedback = "si"
    with col2:
        if st.button("❌ No"):
            st.session_state.feedback = "no"
    with col3:
        if st.button("😐 Más o menos"):
            st.session_state.feedback = "masomenos"

    if st.session_state.feedback == "si":
        st.markdown(
            '<p class="feedback-text">🎉 ¡Genial! Gracias por confirmar la predicción.</p>',
            unsafe_allow_html=True
        )
    elif st.session_state.feedback == "no":
        st.markdown(
            '<p class="feedback-text">📝 Gracias por avisar, tomaremos en cuenta este error '
            'para mejorar el modelo.</p>',
            unsafe_allow_html=True
        )
    elif st.session_state.feedback == "masomenos":
        st.markdown(
            '<p class="feedback-text">🤏 Entendido, parece que el modelo tuvo dudas. '
            'Seguiremos ajustándolo.</p>',
            unsafe_allow_html=True
        )
