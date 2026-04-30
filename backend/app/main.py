import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from ultralytics import YOLO

from app.detector import detect_plates

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/best.pt")
FALLBACK_MODEL = "yolo11n.pt"

# shared state loaded once at startup
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load YOLO — use custom weights if available, otherwise download the base model
    if MODEL_PATH.exists():
        logger.info("Loading custom model from %s", MODEL_PATH)
        state["model"] = YOLO(str(MODEL_PATH))
    else:
        logger.warning("Custom model not found at %s, falling back to %s", MODEL_PATH, FALLBACK_MODEL)
        state["model"] = YOLO(FALLBACK_MODEL)

    # gpu=False keeps the container lean; flip to True if a CUDA runtime is present
    state["reader"] = easyocr.Reader(["en"], gpu=False)
    logger.info("Model and OCR reader ready")

    yield

    state.clear()


app = FastAPI(title="License Plate Detection API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image")

    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    plates = detect_plates(state["model"], state["reader"], img)
    return {"plates": plates}
