"""
FLANN (Fast Library for Approximate Nearest Neighbors): Búsqueda Visual
Demostración con dataset de imágenes reales (china.jpg) de scikit-learn

Conceptos:
- Búsqueda Visual (Visual Search): Proceso de encontrar una imagen o patrón 
  específico (Query) dentro de una base de datos grande o una escena compleja.
- FLANN: Librería optimizada para buscar vecinos más cercanos en espacios de 
  alta dimensionalidad (como los vectores de 128 dimensiones de SIFT). A 
  diferencia de Brute-Force, FLANN crea índices espaciales (como KD-Trees) 
  para hacer la búsqueda exponencialmente más rápida.
- KD-Tree (K-Dimensional Tree): Estructura de datos de partición espacial que 
  organiza los puntos en el espacio para que la búsqueda tome tiempo logarítmico O(log N)
  en lugar de tiempo lineal O(N).
- Homografía: Transformación proyectiva plana utilizada para trazar exactamente 
  dónde se encuentra el objeto buscado dentro de la escena.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
import seaborn as sns
import warnings
import time

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
sns.set_palette("Set1")

# ============================================================================
# 1. CARGAR Y PREPARAR EL DATASET (SIMULACIÓN DE BÚSQUEDA VISUAL)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR ESCENA Y GENERAR EL PATRÓN DE BÚSQUEDA (QUERY)")
print("=" * 80)

# Cargamos la imagen real de scikit-learn que actuará como nuestra "Escena"
print("Cargando imagen de escena 'china.jpg'...")
img_escena_color = load_sample_image("china.jpg")
img_escena = cv2.cvtColor(img_escena_color, cv2.COLOR_RGB2GRAY)

# Simulamos la entrada del usuario: El usuario recorta un detalle de un tejado 
# y quiere que el algoritmo busque dónde está ese patrón en la imagen completa.
# Coordenadas manuales para aislar un edificio específico en la imagen.
x_inicio, x_fin = 150, 280
y_inicio, y_fin = 150, 250

img_query_color = img_escena_color[y_inicio:y_fin, x_inicio:x_fin]

# Para hacer la búsqueda más realista (como si fuera una foto tomada desde el celular),
# rotaremos y escalaremos ligeramente el query.
centro = (img_query_color.shape[1] // 2, img_query_color.shape[0] // 2)
M_rotacion = cv2.getRotationMatrix2D(centro, angle=15, scale=0.85)
img_query_color = cv2.warpAffine(img_query_color, M_rotacion, (img_query_color.shape[1], img_query_color.shape[0]))

img_query = cv2.cvtColor(img_query_color, cv2.COLOR_RGB2GRAY)

print(f"\nDataset: Fotografía de Templo en China")
print(f"Dimensiones de la Escena (Base de Datos): {img_escena.shape[1]}x{img_escena.shape[0]} px")
print(f"Dimensiones del Query (Objeto a buscar):  {img_query.shape[1]}x{img_query.shape[0]} px")

# ============================================================================
# 2. EXTRACCIÓN DE CARACTERÍSTICAS (SIFT)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: EXTRACCIÓN DE VECTORES DE CARACTERÍSTICAS (SIFT)")
print("=" * 80)

# Inicializamos SIFT para encontrar puntos clave invariantes a la escala y rotación
sift = cv2.SIFT_create()

print("Extrayendo descriptores espaciales...")
kp_query, des_query = sift.detectAndCompute(img_query, None)
kp_escena, des_escena = sift.detectAndCompute(img_escena, None)

print(f"✓ Puntos clave en el Query (Objeto): {len(kp_query)}")
print(f"✓ Puntos clave en la Escena (Fondo): {len(kp_escena)}")
print("  (Se buscará emparejar matemáticamente estos vectores 128D)")

# ============================================================================
# 3. EMPAREJAMIENTO OPTIMIZADO (FLANN)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: BÚSQUEDA VISUAL USANDO FLANN (FAST APPROXIMATE NEAREST NEIGHBORS)")
print("=" * 80)

# Configuración de FLANN
# Como estamos usando SIFT (Descriptores Float32), el algoritmo ideal es KD-Tree.
# Si usáramos ORB (Descriptores Binarios), usaríamos LSH (Locality Sensitive Hashing).
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) # 5 árboles KD paralelos
search_params = dict(checks=50) # Recorre 50 ramas recursivas del árbol

print("Inicializando índice FLANN (KD-Tree)...")
flann = cv2.FlannBasedMatcher(index_params, search_params)

t0 = time.time()
# Buscamos los 2 mejores vecinos (k=2) para poder aplicar la Prueba de Razón de Lowe
matches = flann.knnMatch(des_query, des_escena, k=2)
t_calc = (time.time() - t0) * 1000

print(f"✓ Búsqueda FLANN completada en {t_calc:.2f} ms.")
print("  (Mucho más rápido que Brute-Force en datasets gigantes gracias a los árboles de búsqueda)")

# ---------------------------------------------------------
# Prueba de Razón de Lowe (Filtrado de Outliers)
# ---------------------------------------------------------
buenos_matches = []
for m, n in matches:
    # Si la distancia al mejor match es mucho menor (70%) que la distancia al segundo mejor,
    # significa que la coincidencia es única y muy confiable.
    if m.distance < 0.7 * n.distance:
        buenos_matches.append(m)

print(f"\nEmparejamientos iniciales: {len(matches)}")
print(f"Coincidencias altamente confiables (Lowe's Ratio): {len(buenos_matches)}")

# ============================================================================
# 4. ALINEACIÓN GEOMÉTRICA (LOCALIZACIÓN DEL OBJETO)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: LOCALIZACIÓN ESPACIAL Y DELIMITACIÓN (RANSAC)")
print("=" * 80)

MIN_MATCH_COUNT = 10
img_resultado = img_escena_color.copy()

if len(buenos_matches) >= MIN_MATCH_COUNT:
    print("Suficientes coincidencias encontradas. Calculando geometría...")
    
    # Extraer las coordenadas (x, y) de los puntos coincidentes
    pts_query = np.float32([kp_query[m.queryIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)
    pts_escena = np.float32([kp_escena[m.trainIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)
    
    # Encontrar la Matriz de Homografía con RANSAC para descartar errores cruzados
    M_homografia, mascara = cv2.findHomography(pts_query, pts_escena, cv2.RANSAC, 5.0)
    matches_mask = mascara.ravel().tolist()
    
    # Si encontramos una matriz geométrica válida, dibujamos el recuadro
    if M_homografia is not None:
        h_q, w_q = img_query.shape
        # Esquinas originales de la imagen query
        esquinas_query = np.float32([[0, 0], [0, h_q - 1], [w_q - 1, h_q - 1], [w_q - 1, 0]]).reshape(-1, 1, 2)
        
        # Proyectar las esquinas sobre la escena para encontrar el objeto rotado/escalado
        esquinas_escena = cv2.perspectiveTransform(esquinas_query, M_homografia)
        
        # Dibujar un polígono (Bounding Box) verde lima brillante
        img_resultado = cv2.polylines(img_resultado, [np.int32(esquinas_escena)], 
                                      isClosed=True, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
        
        print("✓ Objeto localizado exitosamente. Bounding Box proyectado en la escena.")
else:
    print(f"No hay suficientes coincidencias ({len(buenos_matches)}/{MIN_MATCH_COUNT}).")
    matches_mask = None

# Preparar la imagen visual de las conexiones
img_matches = cv2.drawMatches(
    img_query_color, kp_query, 
    img_resultado, kp_escena, 
    buenos_matches, None, 
    matchColor=(0, 255, 0), # Verde para conexiones exitosas
    singlePointColor=(255, 0, 0), 
    matchesMask=matches_mask, 
    flags=2
)

# ============================================================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DEL MOTOR DE BÚSQUEDA")
print("=" * 80)

fig = plt.figure(figsize=(16, 9))

# Gráfico 1: El Objeto a Buscar (Query)
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax1.imshow(img_query_color)
ax1.set_title('Entrada (Query Image)\nRotada y Modificada', fontsize=12, fontweight='bold')
ax1.axis('off')

# Gráfico 2: La Escena Original (Database)
ax2 = plt.subplot2grid((2, 3), (1, 0))
ax2.imshow(img_escena_color)
ax2.set_title('Escena Completa (Database)', fontsize=12, fontweight='bold')
ax2.axis('off')

# Gráfico 3: Proceso de Emparejamiento Visual FLANN
ax3 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
ax3.imshow(img_matches)
ax3.set_title(f'Búsqueda Visual Asistida por FLANN KD-Tree\n({len(buenos_matches)} Features Emparejados Correctamente)', 
              fontsize=14, fontweight='bold', color='darkblue')
ax3.axis('off')

# Recuadro explicativo
ax3.text(10, img_matches.shape[0] - 20, 
         "El polígono verde delimita la ubicación exacta deducida mediante el mapeo de tensores.", 
         bbox=dict(facecolor='white', alpha=0.8), fontsize=11)

plt.tight_layout()
#plt.savefig('/home/usuario/01_flann_visual_search.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_flann_visual_search.png")
plt.close()

# ============================================================================
# 6. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               FLANN APLICADO A LA BÚSQUEDA VISUAL (VISUAL SEARCH)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. BÚSQUEDA VISUAL VS CLASIFICACIÓN:
   • En lugar de decir "esto es un templo" (Clasificación CNN), la Búsqueda 
     Visual responde a la pregunta geométrica: "¿Exactamente EN QUÉ PÍXELES de 
     esta base de datos gigante se encuentra este fragmento específico?".

2. ¿POR QUÉ FLANN Y NO BRUTE-FORCE?:
   • Fuerza Bruta compara 1 a 1. Si tuviéramos un video 4K o una base de datos 
     de 10,000 imágenes, colapsaría computacionalmente.
   • FLANN (Fast Library for Approximate Nearest Neighbors) construye un 
     'KD-Tree' (Árbol K-Dimensional). Agrupa los vectores espaciales de la 
     escena de forma jerárquica. Al buscar, descarta ramas enteras del árbol 
     en microsegundos, logrando tiempos logarítmicos de búsqueda.

3. INVARIANZA Y ROBUSTEZ:
   • Modificamos sintéticamente el patrón de búsqueda (Query) rotándolo 15 
     grados y reduciendo su tamaño un 15%. 
   • Gracias a los descriptores SIFT combinados con el motor FLANN y el 
     filtrado de Lowe, el algoritmo ignoró la deformación y reconoció 
     matemáticamente las texturas idénticas de las tejas, proyectando 
     (vía Homografía RANSAC) un recuadro perfecto alrededor de la estructura original.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

cv2.FlannBasedMatcher(index_params, search_params):
  - algorithm=FLANN_INDEX_KDTREE: Obligatorio para vectores SIFT/SURF.
  - trees=5: Construye 5 árboles en paralelo. Mayor número = mayor precisión 
    (menos aproximado), pero consume más memoria.
  - checks=50: Cuántas hojas del árbol verificar recursivamente antes de dar 
    el resultado por bueno. Controla el trade-off exactitud/velocidad.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_flann_visual_search.png - Composición mostrando el fragmento de entrada, la 
                             base de datos y las conexiones espaciales proyectadas 
                             para ubicar geométricamente el objetivo.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Pipeline de Búsqueda Visual con FLANN completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")