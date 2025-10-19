"""

%------- PLANTILLA DE CÓDIGO ----------------------------------------------
%------- Coceptos básicos de PDI-------------------------------------------
%------- Por: Juan Diego Cabrera Moncada  juan.cabrera@udea.edu.co --------
%-------      CC 1005234094, Wpp 3103557657 -------------------------------
%-------      Santiago Pereira Ramírez  santiago.pereira@udea.edu.co-------
%-------      CC 1001478542, Wpp 3017323150 -------------------------------
%------- Curso Básico de Procesamiento de Imágenes y Visión Artificial-----
%------- Octubre de 2025---------------------------------------------------
%--------------------------------------------------------------------------

=========================================================
EXPLICACIÓN DETALLADA DE TÉCNICAS DE PDI UTILIZADAS
=========================================================

Este programa combina el modelo de reconocimiento de manos de Mediapipe
con técnicas clásicas de Procesamiento Digital de Imágenes (PDI)
para generar una máscara binaria de la mano y segmentarla del fondo.

A continuación se detallan las etapas y técnicas empleadas:

1. Conversión de espacio de color (BGR → RGB)
------------------------------------------------
- Se usa cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
- OpenCV usa BGR por defecto, pero Mediapipe necesita imágenes RGB.
- Esta es una transformación de espacio de color, una operación básica de PDI.

2️. Creación de máscara binaria
-------------------------------
- Se crea una matriz negra (todo 0) del tamaño del fotograma.
- Luego se “dibuja” sobre ella en blanco (valor 255) la forma de la mano.
- Es una representación binaria: objeto = 1 (blanco), fondo = 0 (negro).
- Se usa para segmentar el objeto de interés (la mano).

3. Dibujo de polígonos y contornos
-----------------------------------
- Con cv2.polylines() se dibujan los dedos.
- Con cv2.fillPoly() se rellena la palma.
- Se utilizan las coordenadas de los landmarks de Mediapipe.
- Esto permite reconstruir la geometría de la mano.
- Es un ejemplo de modelado de contornos y geometría digital.

4. Operaciones morfológicas (Cierre morfológico)
-------------------------------------------------
- cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
- El cierre = dilatación + erosión.
- Rellena huecos pequeños y suaviza los bordes.
- Une regiones separadas (por ejemplo, dedos y palma).
- Es una transformación morfológica clásica.

5. Filtrado Gaussiano (Suavizado)
----------------------------------
- cv2.GaussianBlur(mask, (9, 9), 0)
- Aplica una convolución con una función gaussiana 2D.
- Suaviza bordes y reduce ruido de los contornos.
- Es un filtrado espacial lineal de suavizado.

6. Operación lógica Bitwise (Segmentación)
--------------------------------------------
- cv2.bitwise_and(frame, frame, mask=mask)
- Combina la máscara binaria con la imagen original.
- Muestra solo la parte de la mano (donde mask = 255).
- Es una segmentación por operaciones lógicas punto a punto.

7. Visualización múltiple
--------------------------
- Se muestran dos ventanas:
    • “1️. Mano Binaria con Dedos” → donde ocurre todo el PDI
    • “2️. Juego con Fondo” → donde se visualiza el juego
- Permite separar el análisis visual (PDI) del entorno interactivo (juego).

RESUMEN DE TÉCNICAS DE PDI APLICADAS

| Nº | Técnica PDI               | Tipo / Categoría                 | Función Principal                         |
|----|---------------------------|----------------------------------|-------------------------------------------|
| 1  | Conversión BGR → RGB      | Transformación                   | Adaptar formato de color                  |
| 2  | Creación de máscara       | Representación binaria           | Base para segmentar la mano               |
| 3  | Polilíneas y polígonos    | Geometría digital                | Modelar contornos de dedos y palma        |
| 4  | Cierre morfológico        | Transformación morfológica       | Rellenar huecos y unir regiones           |
| 5  | Filtro Gaussiano          | Filtrado espacial                | Suavizar bordes, eliminar ruido           |
| 6  | Operación lógica AND      | Segmentación / Lógica binaria    | Combinar máscara con fondo original       |
| 7  | Visualización múltiple    | Análisis visual                  | Separar etapas del procesamiento          |

=========================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import os
import time

# =============================
# Configuración inicial
# =============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

base_path = os.path.dirname(os.path.abspath(__file__))

# =============================
# Función segura para cargar imágenes
# =============================
def load_image(filename, size=None):
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Error al cargar la imagen: {path}")
    if size is not None:
        img = cv2.resize(img, size)
    return img

# =============================
# Carga de imágenes
# =============================
nave_img = load_image("nave.png", (80, 80))
enemigo_img = load_image("enemigo.png", (50, 50))
aliado_img = load_image("aliado.png", (50, 50))

# 🔄 Rotar imágenes 90° (ajusta el sentido según necesites)
nave_img = cv2.rotate(nave_img, cv2.ROTATE_90_CLOCKWISE)
enemigo_img = cv2.rotate(enemigo_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
aliado_img = cv2.rotate(aliado_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

fondo_img = load_image("Fondo.jpg")

# =============================
# Función para superponer imágenes con canal alfa
# =============================
def overlay_image(background, overlay, x, y):
    """Dibuja una imagen con canal alfa sobre otra."""
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c] + (1 - alpha) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = overlay
    return background

# =============================
# Variables del juego
# =============================
objetos = []
puntaje = 0
ultimo_spawn = time.time()
fin_del_juego = False
juego_iniciado = False

# =============================
# Función para dibujar botones
# =============================
def dibujar_boton(img, texto, x, y, w, h, mouse_x=None, mouse_y=None):
    """Dibuja un botón interactivo y retorna si fue presionado."""
    hover = False
    if mouse_x is not None and mouse_y is not None:
        if x < mouse_x < x + w and y < mouse_y < y + h:
            hover = True
    
    color = (100, 200, 100) if hover else (50, 150, 50)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)
    
    text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(img, texto, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    return hover

# =============================
# Función para callback del mouse
# =============================
mouse_x, mouse_y = -1, -1
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

# =============================
# Función de mensaje final
# =============================
def mostrar_mensaje_final(texto, color):
    """Muestra una pantalla centrada con el mensaje final."""
    # Ventana más grande para mensajes largos
    ventana = np.zeros((500, 1200, 3), dtype=np.uint8)
    
    # Fondo degradado opcional (puedes comentar estas líneas si prefieres fondo negro)
    for i in range(500):
        intensidad = int(30 + (i / 500) * 20)
        ventana[i, :] = [intensidad, intensidad, intensidad]
    
    # Dividir el texto en líneas si es muy largo (por si usas \n)
    lineas = texto.split('\n')
    
    # Configuración de texto
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    tamanio_fuente = 2.0
    grosor = 6
    
    # Calcular altura total del texto
    altura_linea = 80
    altura_total = len(lineas) * altura_linea
    y_inicial = (500 - altura_total) // 2 + 60
    
    # Dibujar cada línea centrada
    for i, linea in enumerate(lineas):
        text_size = cv2.getTextSize(linea, fuente, tamanio_fuente, grosor)[0]
        text_x = (1200 - text_size[0]) // 2
        text_y = y_inicial + (i * altura_linea)
        
        # Sombra del texto
        cv2.putText(ventana, linea, (text_x + 4, text_y + 4),
                    fuente, tamanio_fuente, (0, 0, 0), grosor, cv2.LINE_AA)
        # Texto principal
        cv2.putText(ventana, linea, (text_x, text_y),
                    fuente, tamanio_fuente, color, grosor, cv2.LINE_AA)
    
    cv2.imshow("Resultado Final", ventana)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

# =============================
# Configurar ventana y callback del mouse
# =============================
cv2.namedWindow("2️. Juego con Fondo")
cv2.setMouseCallback("2️. Juego con Fondo", mouse_callback)

# =============================
# Bucle principal
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # ==========================================================
    # Pantalla de inicio del juego
    # ==========================================================
    if not juego_iniciado:
        fondo = cv2.resize(fondo_img, (w, h))
        pantalla_inicio = fondo.copy()
        
        # Título
        cv2.putText(pantalla_inicio, "Galaxy Guardian", (w//2 - 250, h//2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
        
        # Instrucciones
        cv2.putText(pantalla_inicio, "Destruye enemigos y evita aliados", (w//2 - 280, h//2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Dibujar botón de inicio
        boton_x, boton_y = w//2 - 100, h//2 + 50
        boton_w, boton_h = 200, 80
        
        boton_hover = dibujar_boton(pantalla_inicio, "INICIAR", boton_x, boton_y, 
                                    boton_w, boton_h, mouse_x, mouse_y)
        
        # Verificar si se hizo clic en el botón
        if mouse_clicked and boton_hover:
            juego_iniciado = True
            mouse_clicked = False
            # Reiniciar variables del juego
            objetos = []
            puntaje = 0
            ultimo_spawn = time.time()
        
        mouse_clicked = False
        cv2.imshow("2️. Juego con Fondo", pantalla_inicio)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # ==========================================================
    # 1. Detección de mano y creación de máscara binaria
    # ==========================================================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = None, None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            pts = []
            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)

            dedos = [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [0, 9, 10, 11, 12],
                [0, 13, 14, 15, 16],
                [0, 17, 18, 19, 20]
            ]

            for dedo in dedos:
                contorno = pts[dedo]
                cv2.polylines(mask, [contorno], isClosed=False, color=255, thickness=20)

            palma_pts = np.array([pts[0], pts[5], pts[9], pts[13], pts[17]], dtype=np.int32)
            cv2.fillPoly(mask, [palma_pts], 255)

            base = np.array([pts[0], pts[5], pts[9], pts[13], pts[17]])
            cx, cy = np.mean(base[:, 0]), np.mean(base[:, 1])

    # Aplicar cierre morfológico + suavizado gaussiano (PDI)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)

    # ======================================================
    # 6️. Operación lógica Bitwise (Segmentación de la mano real)
    # ======================================================
    segmented_hand = cv2.bitwise_and(frame, frame, mask=mask)

    # Convertir la máscara a color y aumentar su brillo
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored = cv2.convertScaleAbs(mask_colored, alpha=1.8, beta=50)
    # alpha > 1 aumenta contraste, beta > 0 aumenta brillo

    # ======================================================
    # Fusión visual: realce de la máscara sobre la textura real
    # ======================================================
    # Se mezcla la mano segmentada con una máscara más blanca
    highlight = cv2.addWeighted(segmented_hand, 0.7, mask_colored, 0.9, 0)

    # Mostrar resultado
    cv2.imshow("2️. Mano Segmentada con Bitwise", highlight)

    # ==========================================================
    # 2️. JUEGO CON IMÁGENES
    # ==========================================================
    fondo = cv2.resize(fondo_img, (w, h))
    juego = fondo.copy()

    # Generar enemigos o aliados
    if time.time() - ultimo_spawn > 1.2:
        tipo = random.choice(["enemigo", "aliado"])
        x = w
        y = random.randint(50, h - 50)
        objetos.append([x, y, tipo])
        ultimo_spawn = time.time()

    nuevos_objetos = []
    for (x, y, tipo) in objetos:
        x -= 18  # velocidad
        if cx is not None and cy is not None:
            dist = np.sqrt((cx - x)**2 + (cy - y)**2)
            if dist < 50:
                if tipo == "aliado":
                    puntaje -= 1
                else:
                    puntaje += 1
                continue

        # Si se sale por la izquierda
        if x < -60:
            if tipo == "enemigo":
                puntaje -= 1
            continue

        nuevos_objetos.append([x, y, tipo])

        # Dibujar objeto
        if tipo == "enemigo":
            juego = overlay_image(juego, enemigo_img, int(x), int(y))
        else:
            juego = overlay_image(juego, aliado_img, int(x), int(y))

    objetos = nuevos_objetos

    # Dibujar la nave
    if cx is not None and cy is not None:
        juego = overlay_image(juego, nave_img, int(cx) - 40, int(cy) - 40)

    # Mostrar puntaje
    cv2.putText(juego, f"Puntaje: {puntaje}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("2️. Juego con Fondo", juego)

    # Condiciones de fin del juego
    if puntaje >= 10:
        fin_del_juego = True
        mensaje = ("GG Has salvado la galaxia!!!", (0, 255, 0))
        break
    elif puntaje <= -10:
        fin_del_juego = True
        mensaje = ("El imperio ha tomado la galaxia :(", (0, 0, 255))
        break

    if cv2.waitKey(1) & 0xFF == 27:
        fin_del_juego = False
        break

cap.release()
cv2.destroyAllWindows()

# Mostrar mensaje final si corresponde
if fin_del_juego:
    texto, color = mensaje
    mostrar_mensaje_final(texto, color)

"""
%--------------------------------------------------------------------------
%---------------------------  FIN DEL PROGRAMA ----------------------------
%--------------------------------------------------------------------------
"""
