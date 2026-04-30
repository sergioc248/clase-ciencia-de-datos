import cv2
import numpy as np


def detect_plates(model, reader, img_bgr: np.ndarray) -> list[dict]:
    results = model.predict(img_bgr, imgsz=640, conf=0.5, verbose=False)

    plates = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            # crop the detected plate region
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # upscale 2x for better OCR accuracy
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Otsu binarization normalises varying lighting conditions
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            texts = reader.readtext(thresh, detail=0, paragraph=True)
            text = " ".join(texts).strip()

            plates.append({
                "text": text,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
            })

    return plates
