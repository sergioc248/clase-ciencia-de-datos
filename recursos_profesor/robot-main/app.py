"""
Aplicación de Detección de Pose con MediaPipe
Detecta landmarks de pose humana, calcula ángulos y muestra estados en tiempo real
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import os
from collections import deque

# ==================== CONSTANTES ====================

# Thresholds para detección de estados
EXTENDED_ELBOW_ANGLE = 150
EXTENDED_HOR_RATIO = 0.55
DOWN_ANGLE_THRESHOLD = -60

# Sistema anti-ruido
CONFIRM_FRAMES = 3  # Frames necesarios para confirmar cambio de estado
ALPHA = 0.3  # Factor de suavizado exponencial (0.0 - 1.0)

# Configuración MediaPipe
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Tamaño de ventanas
ROBOT_WINDOW_WIDTH = 640
ROBOT_WINDOW_HEIGHT = 480

# ==================== INICIALIZACIÓN MEDIAPIPE ====================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ==================== CLASES AUXILIARES ====================

class SmoothedValue:
    """Suavizado exponencial para valores numéricos"""
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value


class StateConfirmation:
    """Sistema de confirmación de estados con histeresis"""
    def __init__(self, confirm_frames=CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self.current_state = "unknown"
        self.candidate_state = "unknown"
        self.counter = 0
    
    def update(self, detected_state):
        if detected_state == self.candidate_state:
            self.counter = min(self.counter + 1, self.confirm_frames)
        else:
            self.candidate_state = detected_state
            self.counter = 0
        
        # Cambiar estado solo si hay suficientes confirmaciones
        if self.counter >= self.confirm_frames:
            self.current_state = self.candidate_state
        
        return self.current_state


# ==================== FUNCIONES DE CÁLCULO ====================

def angle_between(a, b, c):
    """
    Calcula el ángulo en el punto b formado por los puntos a-b-c
    Usa producto punto y arccos
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    
    # Evitar división por cero
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product < 1e-8:
        return 0.0
    
    cosang = np.dot(ba, bc) / norm_product
    # Clamp para evitar errores numéricos con arccos
    cosang = np.clip(cosang, -1.0, 1.0)
    
    return np.degrees(np.arccos(cosang))


def distance(p1, p2):
    """Distancia euclidiana entre dos puntos"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def shoulder_to_wrist_angle(shoulder, wrist):
    """
    Calcula el ángulo de la dirección hombro->muñeca
    0° = derecha horizontal, 90° = arriba, -90° = abajo
    """
    vx = wrist[0] - shoulder[0]
    vy = shoulder[1] - wrist[1]  # Invertir Y (coordenadas de píxel)
    
    return math.degrees(math.atan2(vy, vx))


def calculate_horizontal_ratio(shoulder, elbow, wrist):
    """
    Calcula la ratio horizontal normalizada del brazo
    Ayuda a distinguir brazo extendido horizontal de brazo pegado al cuerpo
    """
    arm_len = distance(shoulder, elbow) + distance(elbow, wrist)
    if arm_len < 1.0:
        return 0.0
    
    horizontal_distance = abs(wrist[0] - shoulder[0])
    return horizontal_distance / arm_len


# ==================== EXTRACCIÓN DE LANDMARKS ====================

def get_landmarks_coords(landmarks, width, height, indices):
    """
    Convierte landmarks normalizados a coordenadas de píxeles
    """
    coords = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        px = int(lm.x * width)
        py = int(lm.y * height)
        coords.append((px, py))
    
    return coords


def extract_arm_data(landmarks, width, height, side='right'):
    """
    Extrae datos del brazo (hombro, codo, muñeca, cadera)
    side: 'right' o 'left'
    """
    if side == 'right':
        shoulder_idx = mp_holistic.PoseLandmark.RIGHT_SHOULDER.value
        elbow_idx = mp_holistic.PoseLandmark.RIGHT_ELBOW.value
        wrist_idx = mp_holistic.PoseLandmark.RIGHT_WRIST.value
        hip_idx = mp_holistic.PoseLandmark.RIGHT_HIP.value
    else:
        shoulder_idx = mp_holistic.PoseLandmark.LEFT_SHOULDER.value
        elbow_idx = mp_holistic.PoseLandmark.LEFT_ELBOW.value
        wrist_idx = mp_holistic.PoseLandmark.LEFT_WRIST.value
        hip_idx = mp_holistic.PoseLandmark.LEFT_HIP.value
    
    coords = get_landmarks_coords(landmarks, width, height, 
                                   [shoulder_idx, elbow_idx, wrist_idx, hip_idx])
    
    return {
        'shoulder': coords[0],
        'elbow': coords[1],
        'wrist': coords[2],
        'hip': coords[3]
    }


# ==================== DETECCIÓN DE ESTADOS ====================

def analyze_arm(arm_data):
    """
    Analiza el estado de un brazo individual
    Retorna: dict con elbow_angle, hor_ratio, direction_angle, is_extended, is_up, is_down
    """
    shoulder = arm_data['shoulder']
    elbow = arm_data['elbow']
    wrist = arm_data['wrist']
    hip = arm_data['hip']
    
    # Cálculos biomecánicos
    elbow_angle = angle_between(shoulder, elbow, wrist)
    hor_ratio = calculate_horizontal_ratio(shoulder, elbow, wrist)
    direction_angle = shoulder_to_wrist_angle(shoulder, wrist)
    
    # Criterios de estado
    is_extended = elbow_angle > EXTENDED_ELBOW_ANGLE and hor_ratio > EXTENDED_HOR_RATIO
    is_up = wrist[1] < shoulder[1]  # wrist_y < shoulder_y
    is_down = wrist[1] > hip[1] or direction_angle < DOWN_ANGLE_THRESHOLD
    
    return {
        'elbow_angle': elbow_angle,
        'hor_ratio': hor_ratio,
        'direction_angle': direction_angle,
        'is_extended': is_extended,
        'is_up': is_up,
        'is_down': is_down
    }


def detect_combined_state(right_analysis, left_analysis):
    """
    Detecta el estado combinado de ambos brazos
    Retorna el nombre del estado detectado
    """
    # Extraer estados individuales
    r_ext = right_analysis['is_extended']
    r_up = right_analysis['is_up']
    r_down = right_analysis['is_down']
    
    l_ext = left_analysis['is_extended']
    l_up = left_analysis['is_up']
    l_down = left_analysis['is_down']
    
    # Estados combinados (prioridad en orden)
    if r_down and l_down:
        return "both_down"
    if r_up and l_up:
        return "both_up"
    if r_ext and l_ext:
        return "both_extended"
    
    # Combinaciones asimétricas
    if r_up and l_down:
        return "right_up_left_down"
    if r_up and l_ext:
        return "right_up_left_extended"
    if r_ext and l_down:
        return "right_extended_left_down"
    
    if l_up and r_down:
        return "left_up_right_down"
    if l_up and r_ext:
        return "left_up_right_extended"
    if l_ext and r_down:
        return "left_extended_right_down"
    
    return "neutral"


# ==================== VISUALIZACIÓN ====================

def draw_text_with_background(img, text, pos, font_scale=0.6, thickness=2):
    """Dibuja texto con fondo semitransparente para mejor legibilidad"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Obtener tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Dibujar rectángulo de fondo
    x, y = pos
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y + 5), 
                  (0, 0, 0), -1)
    
    # Dibujar texto
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)


def draw_debug_overlay(img, right_analysis, left_analysis, state):
    """Dibuja overlay de debug con valores y estado actual"""
    height, width = img.shape[:2]
    
    # Título
    draw_text_with_background(img, "Pose Detection Debug", (10, 30), 0.7, 2)
    
    # Estado actual
    draw_text_with_background(img, f"State: {state}", (10, 70), 0.8, 2)
    
    # Datos brazo derecho
    y_offset = 110
    draw_text_with_background(img, "RIGHT ARM:", (10, y_offset), 0.6, 1)
    draw_text_with_background(img, f"  Elbow: {right_analysis['elbow_angle']:.1f}deg", 
                            (10, y_offset + 25), 0.5, 1)
    draw_text_with_background(img, f"  H-Ratio: {right_analysis['hor_ratio']:.2f}", 
                            (10, y_offset + 50), 0.5, 1)
    draw_text_with_background(img, f"  Dir: {right_analysis['direction_angle']:.1f}deg", 
                            (10, y_offset + 75), 0.5, 1)
    
    # Datos brazo izquierdo
    y_offset = 210
    draw_text_with_background(img, "LEFT ARM:", (10, y_offset), 0.6, 1)
    draw_text_with_background(img, f"  Elbow: {left_analysis['elbow_angle']:.1f}deg", 
                            (10, y_offset + 25), 0.5, 1)
    draw_text_with_background(img, f"  H-Ratio: {left_analysis['hor_ratio']:.2f}", 
                            (10, y_offset + 50), 0.5, 1)
    draw_text_with_background(img, f"  Dir: {left_analysis['direction_angle']:.1f}deg", 
                            (10, y_offset + 75), 0.5, 1)
    
    # Instrucciones
    draw_text_with_background(img, "Press 'q' or ESC to quit", 
                            (10, height - 20), 0.5, 1)


def load_robot_images(robot_dir='robot/robot'):
    """Carga todas las imágenes del robot desde el directorio"""
    robot_images = {}
    
    if not os.path.exists(robot_dir):
        print(f"Warning: Directorio {robot_dir}/ no encontrado")
        return robot_images
    
    # Mapeo de estados en inglés a nombres de archivo en español
    state_mapping = {
        "both_down": "ambos_abajo.png",
        "both_up": "ambos_arriba.png",
        "both_extended": "ambos_extendida.png",
        "right_up_left_down": "derecha_arriba_izq_abajo.png",
        "right_up_left_extended": "derecha_arriba_izq_extendida.png",
        "right_extended_left_down": "derecha_extendida_izq_abajo.png",
        "left_up_right_down": "izquierda_arriba_der_abajo.png",
        "left_up_right_extended": "izquierda_arriba_der_extendida.png",
        "left_extended_right_down": "izquierda_extendida_der_abajo.png",
        "neutral": "ambos_abajo.png"  # Usar ambos_abajo como neutral por defecto
    }
    
    for state, filename in state_mapping.items():
        img_path = os.path.join(robot_dir, filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Redimensionar a tamaño de ventana
                img = cv2.resize(img, (ROBOT_WINDOW_WIDTH, ROBOT_WINDOW_HEIGHT))
                robot_images[state] = img
                print(f"Loaded: {state} -> {img_path}")
    
    return robot_images


def get_robot_display(robot_images, state):
    """Obtiene la imagen del robot para el estado actual"""
    if state in robot_images:
        return robot_images[state].copy()
    
    # Imagen por defecto si no existe
    default_img = np.zeros((ROBOT_WINDOW_HEIGHT, ROBOT_WINDOW_WIDTH, 3), dtype=np.uint8)
    cv2.putText(default_img, f"State: {state}", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(default_img, "(Image not found)", (150, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    return default_img


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Función principal de la aplicación"""
    
    # Parsear argumentos
    parser = argparse.ArgumentParser(description='Detección de Pose con MediaPipe')
    parser.add_argument('--video', type=str, default=None, 
                       help='Ruta al archivo de video (opcional, por defecto usa webcam)')
    args = parser.parse_args()
    
    # Configurar captura de video
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Cargando video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Usando webcam")
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la fuente de video")
        return
    
    # Cargar imágenes del robot
    robot_images = load_robot_images('robot/')
    
    # Inicializar sistemas de suavizado y confirmación
    state_confirmer = StateConfirmation(CONFIRM_FRAMES)
    
    # Suavizadores para ángulos (uno por brazo)
    right_elbow_smooth = SmoothedValue(ALPHA)
    right_ratio_smooth = SmoothedValue(ALPHA)
    left_elbow_smooth = SmoothedValue(ALPHA)
    left_ratio_smooth = SmoothedValue(ALPHA)
    
    # Inicializar MediaPipe
    with mp_holistic.Holistic(
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as holistic:
        
        print("\n=== Aplicación iniciada ===")
        print("Controles: 'q' o ESC para salir\n")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Fin del video o error de lectura")
                break
            
            # Obtener dimensiones
            height, width, _ = frame.shape
            
            # Convertir BGR a RGB para MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = holistic.process(image_rgb)
            
            # Dibujar landmarks en la imagen
            annotated_image = frame.copy()
            
            if results.pose_landmarks:
                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Extraer datos de brazos
                right_arm = extract_arm_data(results.pose_landmarks, width, height, 'right')
                left_arm = extract_arm_data(results.pose_landmarks, width, height, 'left')
                
                # Analizar brazos
                right_analysis = analyze_arm(right_arm)
                left_analysis = analyze_arm(left_arm)
                
                # Aplicar suavizado a valores críticos
                right_analysis['elbow_angle'] = right_elbow_smooth.update(right_analysis['elbow_angle'])
                right_analysis['hor_ratio'] = right_ratio_smooth.update(right_analysis['hor_ratio'])
                left_analysis['elbow_angle'] = left_elbow_smooth.update(left_analysis['elbow_angle'])
                left_analysis['hor_ratio'] = left_ratio_smooth.update(left_analysis['hor_ratio'])
                
                # Re-evaluar condiciones con valores suavizados
                right_analysis['is_extended'] = (right_analysis['elbow_angle'] > EXTENDED_ELBOW_ANGLE and 
                                                 right_analysis['hor_ratio'] > EXTENDED_HOR_RATIO)
                left_analysis['is_extended'] = (left_analysis['elbow_angle'] > EXTENDED_ELBOW_ANGLE and 
                                                left_analysis['hor_ratio'] > EXTENDED_HOR_RATIO)
                
                # Detectar estado combinado
                detected_state = detect_combined_state(right_analysis, left_analysis)
                
                # Aplicar confirmación con histeresis
                confirmed_state = state_confirmer.update(detected_state)
                
                # Dibujar debug overlay
                draw_debug_overlay(annotated_image, right_analysis, left_analysis, confirmed_state)
                
                # Obtener imagen del robot
                robot_display = get_robot_display(robot_images, confirmed_state)
            else:
                # No se detectaron landmarks
                draw_text_with_background(annotated_image, "No pose detected", (10, 30), 0.8, 2)
                robot_display = get_robot_display(robot_images, "neutral")
            
            # Mostrar ventanas
            cv2.imshow('Camera Feed - Pose Detection', annotated_image)
            cv2.imshow('Robot State', robot_display)
            
            # Controles de teclado
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == 27:  # 'q' o ESC
                print("Saliendo...")
                break
    
    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    print("Aplicación cerrada")


if __name__ == "__main__":
    main()

