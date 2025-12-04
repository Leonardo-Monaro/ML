import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# 1) ➤ st.set_page_config DEVE vir logo após os imports
# ---------------------------------------------------------
st.set_page_config(page_title="Rock Paper Scissors Classifier", layout="centered")

# ---------------------------------------------------------
# 2) Carregar modelo
# ---------------------------------------------------------
MODEL_PATH = "rock_paper_scissors_model.keras"  # ou .h5

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ['paper', 'rock', 'scissors']

# ---------------------------------------------------------
# 3) Função de predição
# ---------------------------------------------------------
def predict_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    label_index = np.argmax(preds[0])
    confidence = preds[0][label_index]
    return class_names[label_index], confidence

# ---------------------------------------------------------
# 4) Interface Streamlit
# ---------------------------------------------------------
st.title("✋✊✌ Classificador Pedra / Papel / Tesoura")
st.write("Envie uma imagem para o modelo classificar.")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagem enviada", use_column_width=True)

    if st.button("Classificar"):
        label, conf = predict_image(img)
        st.subheader(f"Predição: **{label.upper()}**")
        st.write(f"Confiança: **{conf * 100:.2f}%**")
