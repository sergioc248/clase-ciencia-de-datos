from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image


IMG_HEIGHT = 180
IMG_WIDTH = 180
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artefactos_parcial1" / "flower_classifier.keras"
DEFAULT_CLASSES_PATH = Path(__file__).resolve().parent / "artefactos_parcial1" / "class_names.json"


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names(classes_path: str) -> list[str]:
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("class_names.json no contiene una lista valida de clases.")
    return class_names


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(image, dtype=np.float32)

    # Misma normalizacion usada en el notebook: [0, 255] -> [0, 1]
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def main() -> None:
    st.set_page_config(page_title="Clasificador de Flores", layout="wide")

    st.title("Clasificador de Tipos de Flores")
    st.write(
        "Carga una imagen desde tu equipo o desde una URL para predecir la clase de flor y ver la distribucion de probabilidades."
    )

    st.sidebar.header("Configuracion")
    model_path = st.sidebar.text_input("Ruta del modelo", value=str(DEFAULT_MODEL_PATH))
    classes_path = st.sidebar.text_input("Ruta del archivo de clases", value=str(DEFAULT_CLASSES_PATH))

    try:
        model = load_model(model_path)
        class_names = load_class_names(classes_path)
    except Exception as exc:
        st.error(f"Error cargando modelo o clases: {exc}")
        st.stop()

    st.subheader("Catalogo de clases")
    st.write(", ".join(class_names))

    source = st.radio("Fuente de imagen", options=["Archivo local", "URL"], horizontal=True)

    image: Image.Image | None = None

    if source == "Archivo local":
        uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        image_url = st.text_input("Pega la URL de la imagen")
        if image_url:
            try:
                image = load_image_from_url(image_url)
            except Exception as exc:
                st.error(f"No se pudo descargar/cargar la imagen desde la URL: {exc}")

    if image is None:
        st.info("Carga una imagen para ejecutar la prediccion.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Imagen de entrada")
        st.image(image, use_container_width=True)

    input_tensor = preprocess_image(image)
    probs = model.predict(input_tensor, verbose=0)[0]

    if len(probs) != len(class_names):
        st.error(
            "El numero de clases del modelo no coincide con class_names.json. "
            f"Modelo: {len(probs)}, clases: {len(class_names)}"
        )
        st.stop()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_prob = float(probs[pred_idx])

    with col2:
        st.subheader("Prediccion")
        st.success(f"Clase mas probable: {pred_class} ({pred_prob:.2%})")

    st.subheader("Distribucion de probabilidades por clase")

    data = []
    for idx, (label, prob) in enumerate(zip(class_names, probs)):
        data.append({
            "clase": label,
            "probabilidad": float(prob),
            "es_prediccion": idx == pred_idx,
        })

    # Tabla ordenada por probabilidad para lectura rapida
    ranked = sorted(data, key=lambda x: x["probabilidad"], reverse=True)
    st.dataframe(
        [
            {
                "Clase": row["clase"],
                "Probabilidad": f"{row['probabilidad']:.2%}",
                "Top": "SI" if row["es_prediccion"] else "",
            }
            for row in ranked
        ],
        use_container_width=True,
        hide_index=True,
    )

    # Grafica de barras con resaltado automatico de la clase top
    chart_labels = [row["clase"] for row in data]
    chart_values = [row["probabilidad"] for row in data]
    chart_colors = ["#b0bec5" for _ in data]
    chart_colors[pred_idx] = "#ff7043"

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(chart_labels, chart_values, color=chart_colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidad")
    ax.set_title("Probabilidad por clase")
    ax.tick_params(axis="x", rotation=45)

    for bar, value in zip(bars, chart_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
