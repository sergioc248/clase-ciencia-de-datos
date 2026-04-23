# Piano con las manos — Proyecto de examen (Ciencia de Datos)

Resumen
-------
Este proyecto implementa un "piano virtual" controlado por las puntas de los dedos usando la cámara. Utiliza MediaPipe Hands (modelo de IA para detección de manos y 21 landmarks) para detectar las puntas de los dedos (landmarks 4, 8, 12, 16, 20) y pygame para reproducir samples (MP3/WAV). Al mover la punta de un dedo hacia abajo se interpreta como una pulsación y se reproduce la nota mapeada.

Modo de uso del examen (modo directo, sin calibración)
-----------------------------------------------------
Esta guía describe el modo "directo" del programa — no incluye calibración interactiva.  
Comportamiento del programa:
- El programa muestra la imagen RAW (sin flip). La posición en pantalla corresponde a la posición física.
- El mapeo de notas se realiza por posición horizontal en la imagen: la mano que aparece más a la izquierda reproducirá las notas más graves; la que aparece más a la derecha reproducirá las notas más agudas.
- No hay pasos de calibración; basta colocar las manos de forma que la izquierda física quede a la izquierda en el encuadre y la derecha física a la derecha.

Contexto técnico
----------------
- MediaPipe Hands es un modelo de IA que detecta 21 landmarks por mano (x,y,z normalizados).  
- Usamos exclusivamente las puntas: 4 (thumb tip), 8 (index tip), 12 (middle tip), 16 (ring tip), 20 (pinky tip).  
- La medición de movimiento usa la coordenada y normalizada (0..1) y un umbral relativo para independencia del tamaño del frame.  
- Se aplica un suavizado (EMA) y un cooldown por dedo para reducir falsos positivos y retriggers.  
- Mapeo: mano más a la izquierda → notas graves; mano más a la derecha → notas agudas.

Requisitos recomendados (librerías)
-----------------------------------
- Python 3.8 - 3.11 (recomendado 3.9/3.10 para compatibilidad con mediapipe en algunos sistemas)
- Paquetes Python (agregar en `requirements.txt`):
  - opencv-python
  - mediapipe
  - numpy
  - pygame
  - pydub (opcional, para convertir MP3→WAV)
  - protobuf==3.20.3

requirements.txt sugerido
-------------------------
```text
opencv-python
mediapipe
numpy
pygame
pydub
protobuf==3.20.3
```

Estructura del proyecto (esperada)
----------------------------------
```
proyecto_piano/
├─ piano_raw_direct.py          # script principal (modo directo, sin calibración)
├─ notas_config.py              # (opcional) mapeo notas / helper
├─ requirements.txt
├─ sonidos/
│  ├─ C4.mp3
│  ├─ D4.mp3
│  ├─ E4.mp3
│  ├─ F4.mp3
│  ├─ G4.mp3
│  ├─ A4.mp3
│  ├─ B4.mp3
│  ├─ C5.mp3
│  ├─ D5.mp3
│  └─ E5.mp3
└─ README.md
```

Instalación (pasos resumidos)
-----------------------------
1. Clonar o copiar el repositorio del proyecto.
2. Crear y activar un entorno virtual:
   - Windows (PowerShell):
     - python -m venv .venv
     - .\.venv\Scripts\Activate.ps1
   - macOS / Linux:
     - python3 -m venv .venv
     - source .venv/bin/activate
3. Actualizar pip:
   - python -m pip install --upgrade pip setuptools wheel
4. Instalar dependencias:
   - pip install -r requirements.txt
5. (Opcional) Si usas MP3 y pydub: instalar ffmpeg en el sistema (añadir a PATH).

Ejecución y controles
---------------------
1. Ejecutar:
   - python piano_raw_direct.py
2. Ventana del programa:
   - La imagen se muestra RAW (sin flip). Asegúrate de que tu mano física izquierda esté a la izquierda del encuadre y la derecha a la derecha.
3. Controles:
   - ESC → salir

Pruebas rápidas (qué verificar)
-------------------------------
- Mueve el meñique de la mano izquierda (física): debe sonar la nota más baja (p. ej. C4 según configuración).
- Mueve el meñique de la mano derecha (física): debe sonar la nota más alta (p. ej. E5).
- Si los sonidos no se reproducen, revisa mensajes en la consola sobre archivos no encontrados y verifica `sonidos/`.

Explicación breve del algoritmo
-------------------------------
- MediaPipe devuelve landmarks normalizados. Tomamos la coordenada y (vertical) de cada tip.
- Suavizamos la coordenada con EMA
