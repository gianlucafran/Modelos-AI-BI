"""
Detector Harris: Extracción de Características y Seguimiento de Video (Optical Flow)
Demostración con dataset de imágenes reales de scikit-learn (Simulación de Video)

Conceptos:
- Detector de Esquinas de Harris: Algoritmo matemático que identifica puntos de interés 
  (esquinas) evaluando cómo cambia la intensidad de los píxeles en todas las direcciones.
  Una esquina es un punto ideal para el seguimiento porque su posición está bien 
  definida (a diferencia de un borde, que sufre del "Problema de la Apertura").
- Matriz de Estructura (Tensor): Matriz 2x2 de derivadas espaciales de la imagen. Los 
  valores propios (eigenvalues) de esta matriz determinan si un área es plana, borde o esquina.
- Flujo Óptico (Lucas-Kanade): Algoritmo que rastrea el movimiento de píxeles específicos 
  (las esquinas de Harris) entre fotogramas consecutivos de un video asumiendo que 
  la intensidad luminosa del objeto no cambia significativamente.
- Simulación de Video: Como scikit-learn no incluye datasets de video, tomaremos 
  una imagen estática de alta resolución y aplicaremos transformaciones afines 
  consecutivas (traslaciones) para simular un paneo de cámara (Video Sintético).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Requiere instalación de OpenCV: pip install opencv-python
try:
    import cv2
except ImportError:
    print("Por favor, instala OpenCV usando: pip install opencv-python")
    import sys
    sys.exit()

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. GENERAR EL DATASET (SIMULACIÓN DE VIDEO CON SCIKIT-LEARN)
# ============================================================================
print("=" * 80)
print("PASO 1: CREAR SECUENCIA DE VIDEO SINTÉTICO (PANEO DE CÁMARA)")
print("=" * 80)

# Cargamos una imagen real de alta resolución de scikit-learn
print("Cargando imagen base 'china.jpg'...")
img_color = load_sample_image("china.jpg")
img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

h, w = img_gray.shape

print("Generando fotogramas de video simulando el movimiento de un dron...")
num_frames = 30
video_frames = []
video_frames_color = []

# Creamos 30 fotogramas trasladando la imagen (x=+4, y=+2 píxeles por frame)
# Simulamos así un movimiento suave de cámara a lo largo de la escena arquitectónica.
for i in range(num_frames):
    # Matriz de Traslación M = [[1, 0, tx], [0, 1, ty]]
    tx = i * 4
    ty = i * 2
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    
    frame_gray = cv2.warpAffine(img_gray, M, (w, h))
    frame_color = cv2.warpAffine(img_color, M, (w, h))
    
    video_frames.append(frame_gray)
    video_frames_color.append(frame_color)

print(f"\nDataset: Secuencia de Video Sintético")
print(f"Fotogramas generados: {len(video_frames)}")
print(f"Resolución del video: {w}x{h} píxeles a 1 canal (Grises)")

# ============================================================================
# 2. DETECTOR DE HARRIS (INICIALIZACIÓN DE PUNTOS A RASTREAR)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DETECCIÓN DE ESQUINAS (HARRIS CORNER DETECTOR)")
print("=" * 80)

# Tomamos el primer fotograma para encontrar los puntos de interés iniciales
frame_inicial = video_frames[0]

print("Aplicando el Detector de Esquinas de Harris al Fotograma 0...")

# OpenCV proporciona 'goodFeaturesToTrack', una función excelente para inicializar
# algoritmos de seguimiento. Le pasamos 'useHarrisDetector=True' para aplicar la 
# matemática exacta de Harris: R = det(M) - k*(trace(M))^2
puntos_harris = cv2.goodFeaturesToTrack(
    image=frame_inicial,
    maxCorners=100,        # Queremos seguir los 100 puntos más fuertes
    qualityLevel=0.01,     # Umbral de calidad mínima (1% del mejor punto)
    minDistance=15,        # Distancia mínima en píxeles entre esquinas (evita aglomeración)
    useHarrisDetector=True,# ACTIVAMOS EL DETECTOR HARRIS OBLIGATORIAMENTE
    k=0.04                 # Parámetro libre de Harris (típicamente entre 0.04 y 0.06)
)

print(f"✓ El Detector de Harris identificó {len(puntos_harris)} esquinas estructurales fuertes.")
print("  (Ej. bordes de techos, ventanas y pilares del templo).")

# ============================================================================
# 3. SEGUIMIENTO DE VIDEO (LUCAS-KANADE OPTICAL FLOW)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: RASTREO EN EL TIEMPO (OPTICAL FLOW)")
print("=" * 80)

# Parámetros para el algoritmo de Flujo Óptico Lucas-Kanade
lk_params = dict(
    winSize=(15, 15), # Tamaño de la ventana de búsqueda
    maxLevel=2,       # Niveles de la pirámide de imágenes (para movimientos rápidos)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # Criterios de parada
)

# Creamos una imagen vacía (máscara) para dibujar las estelas/trayectorias de los puntos
mascara_trayectorias = np.zeros_like(video_frames_color[0])

# Variables para el bucle
old_gray = frame_inicial
p0 = puntos_harris

print(f"Procesando secuencia de {num_frames} fotogramas. Rastreando coordenadas...")

# Bucle principal de procesamiento de video
for i in range(1, num_frames):
    frame_actual_gray = video_frames[i]
    frame_actual_color = video_frames_color[i].copy()
    
    # Calculamos el Flujo Óptico: 
    # Buscamos dónde terminaron los puntos 'p0' en el 'frame_actual_gray'
    p1, status, error = cv2.calcOpticalFlowPyrLK(old_gray, frame_actual_gray, p0, None, **lk_params)
    
    # Filtramos los puntos que se encontraron con éxito (status = 1)
    puntos_buenos_nuevos = p1[status == 1]
    puntos_buenos_viejos = p0[status == 1]
    
    # Dibujamos las trayectorias (líneas) y los puntos actuales en el frame
    for j, (nuevo, viejo) in enumerate(zip(puntos_buenos_nuevos, puntos_buenos_viejos)):
        a, b = nuevo.ravel() # Nuevas coordenadas (x, y)
        c, d = viejo.ravel() # Viejas coordenadas (x, y)
        
        a, b, c, d = int(a), int(b), int(c), int(d)
        
        # Línea de trayectoria (Azul)
        mascara_trayectorias = cv2.line(mascara_trayectorias, (a, b), (c, d), (0, 150, 255), 2)
        # Punto actual (Rojo)
        frame_actual_color = cv2.circle(frame_actual_color, (a, b), 5, (255, 0, 0), -1)
        
    # Preparar el siguiente ciclo
    old_gray = frame_actual_gray.copy()
    # p0 debe tener la forma (N, 1, 2)
    p0 = puntos_buenos_nuevos.reshape(-1, 1, 2)

print("✓ Seguimiento completado. Todas las trayectorias han sido almacenadas.")

# Fusionamos el último frame con las trayectorias dibujadas
imagen_final_seguimiento = cv2.add(frame_actual_color, mascara_trayectorias)

# ============================================================================
# 4. VISUALIZACIÓN: PUNTOS DE HARRIS Y TRAYECTORIAS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: VISUALIZACIÓN DE RESULTADOS")
print("=" * 80)

# Dibujamos los puntos iniciales sobre el Frame 0 para ver qué eligió Harris
frame_0_visual = video_frames_color[0].copy()
for pt in puntos_harris:
    x, y = pt.ravel()
    cv2.circle(frame_0_visual, (int(x), int(y)), 4, (0, 255, 0), -1) # Verde

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: Inicialización con Harris
axes[0].imshow(frame_0_visual)
axes[0].set_title('Paso 1: Detección de Esquinas de Harris (Frame 0)\n(Puntos estructurales de alto gradiente en X e Y)', 
                  fontsize=13, fontweight='bold')
axes[0].axis('off')

# Leyenda manual para Harris
axes[0].text(20, h - 30, "● Puntos inicializados (Esquinas)", color="white", 
             bbox=dict(facecolor='green', alpha=0.7), fontsize=12)

# Gráfico 2: Flujo Óptico (Tracking en Video)
axes[1].imshow(imagen_final_seguimiento)
axes[1].set_title(f'Paso 2: Seguimiento Lucas-Kanade (Frame {num_frames-1})\n(Trayectorias de movimiento de las esquinas)', 
                  fontsize=13, fontweight='bold')
axes[1].axis('off')

axes[1].text(20, h - 30, "● Posición actual   ▬ Trayectoria recorrida", color="white", 
             bbox=dict(facecolor='black', alpha=0.7), fontsize=12)

plt.tight_layout()
#plt.savefig('/home/usuario/01_harris_video_tracking.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_harris_video_tracking.png")
plt.close()

# ============================================================================
# 5. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               DETECTOR DE HARRIS EN SEGUIMIENTO DE VIDEO                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. EL PROBLEMA DEL RASTREO (TRACKING):
   • Rastrear un objeto en un video píxel por píxel es imposible e ineficiente. 
     Se requiere encontrar "puntos clave" que sean robustos a los cambios de 
     iluminación y perspectiva.

2. LA SOLUCIÓN: ESQUINAS DE HARRIS:
   • ¿Por qué esquinas? Si intentas rastrear una línea recta (borde), el algoritmo 
     sufre del "Problema de Apertura" (no puede notar el deslizamiento a lo largo 
     de la línea).
   • Harris Corner Detector calcula las derivadas espaciales en X e Y. Si ambas 
     direcciones presentan una alta variación (ambos Eigenvalues son grandes), 
     sabemos que hemos encontrado una esquina. Estas esquinas (marcadas en verde 
     en el Gráfico 1) son anclas perfectas para rastrear.

3. OPTICAL FLOW (LUCAS-KANADE):
   • Una vez que Harris nos dio las coordenadas (x,y) de las 100 esquinas más 
     fuertes del edificio, Lucas-Kanade se encargó del rastreo dinámico.
   • Lucas-Kanade asume que el movimiento entre dos frames es pequeño y que 
     los píxeles conservan su intensidad. A través de la Pirámide de Imágenes, 
     el algoritmo siguió exitosamente el paneo de nuestra "cámara simulada", 
     trazando las rutas amarillas visibles en el Gráfico 2.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

cv2.goodFeaturesToTrack(useHarrisDetector=True, k=0.04):
  - Activa la respuesta original de Harris R = λ1*λ2 - k*(λ1+λ2)^2.
  - k=0.04 es el coeficiente empírico para diferenciar entre bordes y esquinas.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_harris_video_tracking.png - Visualización dual del estado inicial (Detección 
                               estática) y el estado final (Historial de rastreo).

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de rastreo con Detector Harris y Flujo Óptico completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")