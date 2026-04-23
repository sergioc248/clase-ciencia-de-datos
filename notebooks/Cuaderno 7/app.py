import io
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw


st.set_page_config(page_title="Detector CIFAR-10", page_icon="🧠", layout="centered")

CLASSES = [
	"avion",
	"automovil",
	"pajaro",
	"gato",
	"ciervo",
	"perro",
	"rana",
	"caballo",
	"barco",
	"camion o bus",
]


@st.cache_resource(show_spinner=False)
def load_detector_model() -> tf.keras.Model:
	# Try common locations to make the app easier to run from different folders.
	candidate_paths = [
		Path(__file__).resolve().parent / "detector.keras",
		Path.cwd() / "detector.keras",
	]

	model_path = next((p for p in candidate_paths if p.exists()), None)
	if model_path is None:
		searched = "\n".join(str(p) for p in candidate_paths)
		raise FileNotFoundError(
			"No se encontro detector.keras. Ubica el archivo en una de estas rutas:\n"
			f"{searched}"
		)

	return tf.keras.models.load_model(model_path, compile=False)


def read_image_from_url(url: str) -> Image.Image:
	response = requests.get(url, timeout=15)
	response.raise_for_status()
	return Image.open(io.BytesIO(response.content)).convert("RGB")


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
	image_rgb = image.convert("RGB")
	image_resized = image_rgb.resize((32, 32), Image.Resampling.BILINEAR)

	# Match training preprocessing: pixel values in [0, 1].
	image_array = np.array(image_resized, dtype=np.float32) / 255.0
	image_batch = np.expand_dims(image_array, axis=0)
	return image_batch, image_rgb


def draw_prediction_label(image: Image.Image, label: str, confidence: float) -> Image.Image:
	output = image.copy()
	draw = ImageDraw.Draw(output)
	text = f"Prediccion: {label} ({confidence:.2%})"

	# Draw a solid bar at the top to place the prediction text.
	bar_height = 40
	draw.rectangle([(0, 0), (output.width, bar_height)], fill=(0, 0, 0))
	draw.text((10, 10), text, fill=(255, 255, 255))
	return output


def predict(image: Image.Image, model: tf.keras.Model) -> tuple[str, np.ndarray, Image.Image]:
	image_batch, original_image = preprocess_image(image)
	probs = model.predict(image_batch, verbose=0)[0]
	pred_idx = int(np.argmax(probs))
	pred_label = CLASSES[pred_idx]
	return pred_label, probs, original_image


st.title("Clasificador de Imagenes con detector.keras")
st.write(
	"Sube una imagen desde tu PC, pega una URL o toma una foto con la webcam para clasificarla."
)

try:
	model = load_detector_model()
except Exception as exc:
	st.error(f"No fue posible cargar el modelo: {exc}")
	st.stop()

source = st.radio(
	"Fuente de imagen",
	["URL", "Archivo local", "Webcam"],
	horizontal=True,
)

input_image = None

if source == "URL":
	image_url = st.text_input("URL de la imagen", placeholder="https://.../imagen.jpg")
	if image_url:
		try:
			input_image = read_image_from_url(image_url)
		except Exception as exc:
			st.error(f"No se pudo cargar la imagen desde la URL: {exc}")

elif source == "Archivo local":
	uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"])
	if uploaded is not None:
		try:
			input_image = Image.open(uploaded).convert("RGB")
		except Exception as exc:
			st.error(f"No se pudo leer el archivo: {exc}")

else:
	camera_file = st.camera_input("Toma una foto")
	if camera_file is not None:
		try:
			input_image = Image.open(camera_file).convert("RGB")
		except Exception as exc:
			st.error(f"No se pudo leer la foto de webcam: {exc}")

if input_image is not None:
	predicted_label, probabilities, original_image = predict(input_image, model)
	top_confidence = float(np.max(probabilities))
	labeled_image = draw_prediction_label(original_image, predicted_label, top_confidence)

	st.image(
		labeled_image,
		caption=f"Clase predicha: {predicted_label}",
		use_container_width=True,
	)

	st.subheader("Probabilidades por clase")
	prob_table = {
		"clase": CLASSES,
		"probabilidad": [float(p) for p in probabilities],
	}
	st.dataframe(prob_table, use_container_width=True, hide_index=True)

	st.bar_chart(prob_table, x="clase", y="probabilidad", use_container_width=True)
else:
	st.info("Selecciona una fuente y proporciona una imagen para obtener la prediccion.")
