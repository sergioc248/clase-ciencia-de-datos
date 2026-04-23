import warnings
# Suprimir advertencia protobuf temprano
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
import tempfile
import atexit
import shutil

# Intentaremos usar pydub solo si está disponible para convertir mp3->wav
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

# --------------------------
# CONFIGURACIÓN DE NOTAS
# --------------------------
mano_derecha = ['C4', 'D4', 'E4', 'F4', 'G4']
mano_izquierda = ['A4', 'B4', 'C5', 'D5', 'E5']
# Orden: meñique izq -> ... -> pulgar izq -> pulgar der -> ... -> meñique der
nota_por_dedo = mano_izquierda[::-1] + mano_derecha  # 10 notas
ruta_sonidos = "sonidos"

# --------------------------
# PYGAME / CARGA SAMPLES
# --------------------------
try:
    pygame.mixer.init()
except Exception as e:
    print("⚠️ pygame.mixer.init() falló:", e)
    raise

TMP_DIR = tempfile.mkdtemp(prefix="piano_sounds_")
atexit.register(lambda: shutil.rmtree(TMP_DIR, ignore_errors=True))

def ffmpeg_available():
    return shutil.which("ffmpeg") is not None or shutil.which("ffmpeg.exe") is not None

def try_load(path):
    try:
        return pygame.mixer.Sound(path)
    except Exception:
        return None

sonidos = {}
for i, nombre in enumerate(nota_por_dedo):
    mp3_path = os.path.join(ruta_sonidos, f"{nombre}.mp3")
    wav_path = os.path.join(ruta_sonidos, f"{nombre}.wav")
    loaded = False

    if os.path.exists(mp3_path):
        snd = try_load(mp3_path)
        if snd:
            sonidos[i] = snd
            loaded = True
        else:
            if PYDUB_AVAILABLE and ffmpeg_available():
                try:
                    audio = AudioSegment.from_file(mp3_path, format="mp3")
                    out_wav = os.path.join(TMP_DIR, f"{nombre}.wav")
                    audio.export(out_wav, format="wav")
                    snd = try_load(out_wav)
                    if snd:
                        sonidos[i] = snd
                        loaded = True
                    else:
                        print(f"⚠️ No se pudo cargar el WAV convertido para {mp3_path}")
                except Exception as e:
                    print(f"⚠️ Error convirtiendo {mp3_path} a WAV: {e}")
            else:
                print(f"⚠️ No se pudo cargar {mp3_path} directamente. Para conversión automática instala pydub y ffmpeg.")
    if not loaded and os.path.exists(wav_path):
        snd = try_load(wav_path)
        if snd:
            sonidos[i] = snd
            loaded = True

    if not loaded:
        print(f"⚠️ Sonido no encontrado o no cargable para nota {nombre}. Buscados: {mp3_path} , {wav_path}")

# --------------------------
# MEDIAPIPE (solo puntas de dedo)
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Índices estrictos de las puntas (usar SOLO estos)
FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

# --------------------------
# DETECCIÓN MOVIMIENTO POR PUNTA (normalizado)
# --------------------------
prev_y = [None] * 10             # valor smoothed anterior por dedo (normalizado 0..1)
alpha_smooth = 0.6               # EMA alpha (0..1), mayor = menos suavizado
UMBRAL_MOV_REL = 0.03            # umbral relativo (fracción de la altura). ~3%
COOLDOWN = 0.18                  # segundos entre retriggers por dedo
last_play = {i: 0 for i in range(10)}

def tocar_sonido_idx(idx):
    if idx not in sonidos:
        return
    ahora = time.time()
    if ahora - last_play.get(idx, 0) < COOLDOWN:
        return
    try:
        sonidos[idx].play()
        last_play[idx] = ahora
    except Exception as e:
        print("Error reproduciendo sonido:", e)

# --------------------------
# UTIL: obtener puntas en orden consistente usando handedness
# --------------------------
def obtener_tips_ordenados(results, frame_width, frame_height):
    """
    Retorna lista de 10 tuples (x_px, y_px, y_norm) en el orden:
    meñique izq -> ... -> pulgar izq -> pulgar der -> ... -> meñique der.
    Si una mano no está presente, sus 5 entradas serán (None, None, None).

    IMPORTANTE: emparejamos multi_hand_landmarks con multi_handedness
    usando zip como dicta la API de MediaPipe.
    """
    salida = []

    if not results.multi_hand_landmarks:
        return [(None, None, None)] * 10

    # Asegurar que tenemos handedness y landmarks emparejados
    # La API garantiza que multi_hand_landmarks[i] corresponde a multi_handedness[i]
    hands_by_label = {}
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' o 'Right' (respecto a la persona)
            hands_by_label[label] = lm

    # Mano izquierda: queremos meñique->...->pulgar (reversed tips)
    if 'Left' in hands_by_label:
        lm = hands_by_label['Left']
        for tip in reversed(FINGER_TIPS):  # pinky -> thumb
            x = int(lm.landmark[tip].x * frame_width)
            y = int(lm.landmark[tip].y * frame_height)
            y_norm = lm.landmark[tip].y
            salida.append((x, y, y_norm))
    else:
        salida.extend([(None, None, None)] * 5)

    # Mano derecha: thumb->...->pinky
    if 'Right' in hands_by_label:
        lm = hands_by_label['Right']
        for tip in FINGER_TIPS:
            x = int(lm.landmark[tip].x * frame_width)
            y = int(lm.landmark[tip].y * frame_height)
            y_norm = lm.landmark[tip].y
            salida.append((x, y, y_norm))
    else:
        salida.extend([(None, None, None)] * 5)

    # seguridad: devolver exactamente 10 entradas
    if len(salida) != 10:
        # completar con None si algo extraño sucedió
        salida = (salida + [(None, None, None)] * 10)[:10]
    return salida

# --------------------------
# BUCLE PRINCIPAL
# --------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se puede abrir la cámara.")
    exit(1)

print("\n🎹 Piano activado — usando SOLO puntas y landmarks de ambas manos correctamente emparejados")
print("Mueve las puntas de tus dedos hacia abajo para tocar.\n")

# Nota sobre flip horizontal:
# - En este script flippeamos el frame antes de procesar, por lo que la handedness
#   devuelta por MediaPipe corresponde a la imagen que ves en pantalla.
# - Si prefieres procesar sin flip y sólo voltear para mostrar, mueve el cv2.flip
#   después del procesamiento. Ambas opciones son válidas, solo cambia la referencia visual.

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error leyendo frame.")
        break

    # Volteamos horizontalmente para comportamiento "espejo" (opcional).
    # Esto también cambiará lo que ve Mediapipe, pero la API devuelve handedness
    # consistente con la imagen procesada.
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Si no hay detecciones, resetear historial para evitar retriggers
    if not results.multi_hand_landmarks:
        prev_y = [None] * 10
        cv2.imshow("Piano - puntas (pygame) - ambas manos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    tips = obtener_tips_ordenados(results, w, h)  # lista de 10 (x,y,y_norm)

    # Procesar solo usando las puntas (y_norm), con suavizado EMA
    for i in range(10):
        x_px, y_px, y_norm = tips[i]
        if y_norm is None:
            prev_y[i] = None
            continue

        # aplicar suavizado EMA
        if prev_y[i] is None:
            y_smooth = y_norm
        else:
            y_smooth = alpha_smooth * y_norm + (1 - alpha_smooth) * prev_y[i]

        # Dibujar punto en pixel (opcional)
        cv2.circle(frame, (x_px, int(y_px)), 6, (255, 0, 0), -1)

        # calcular desplazamiento en coordenadas normalizadas (positivo hacia abajo)
        if prev_y[i] is not None:
            dy_norm = y_smooth - prev_y[i]
            if dy_norm > UMBRAL_MOV_REL:
                tocar_sonido_idx(i)
                nota = nota_por_dedo[i]
                cv2.putText(frame, nota, (x_px - 10, int(y_px) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        prev_y[i] = y_smooth

    cv2.imshow("Piano - puntas (pygame) - ambas manos", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()