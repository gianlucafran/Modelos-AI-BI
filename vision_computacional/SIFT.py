"""
SIFT (Scale-Invariant Feature Transform): Stitching de Imágenes (Panorámicas)
Demostración con dataset de imágenes reales (china.jpg) de scikit-learn y OpenCV

Conceptos:
- SIFT: Algoritmo de Visión por Computadora que detecta y describe características 
  locales en imágenes. Es invariante a la escala, rotación y parcialmente a la iluminación.
- Puntos Clave (Keypoints): Coordenadas espaciales donde hay características 
  distintivas (ej. esquinas, bordes de alto contraste).
- Descriptores: Vectores matemáticos (128 dimensiones en SIFT) que describen la 
  apariencia visual de un punto clave para que pueda ser reconocido en otra imagen.
- Feature Matching (Emparejamiento): Búsqueda de los descriptores más similares 
  entre dos imágenes usando distancia Euclidiana y la "Prueba de Razón de Lowe".
- Homografía y RANSAC: Cálculo de la matriz de transformación perspectiva que 
  alinea una imagen con la otra, descartando emparejamientos erróneos (Outliers).
- Image Stitching: Proceso de fusionar múltiples imágenes con solapamiento 
  en una única imagen panorámica.

Nota: Scikit-Learn proporciona las imágenes de prueba, pero SIFT es un algoritmo 
nativo de OpenCV.
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

# Configurar estilo general
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# 1. CARGAR Y PREPARAR EL DATASET (SIMULACIÓN DE DOS FOTOS SOLAPADAS)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR IMAGEN Y SIMULAR FOTOS CON SOLAPAMIENTO")
print("=" * 80)

# Cargamos una imagen real de alta resolución de scikit-learn
print("Cargando imagen 'china.jpg'...")
img_original = load_sample_image("china.jpg")

# Para demostrar el "Stitching" (Creación de panorámicas), dividiremos 
# esta imagen en dos mitades (Izquierda y Derecha) con un 30% de solapamiento.
h, w, canales = img_original.shape

# Imagen 1 (Izquierda): Desde el inicio hasta el 65% del ancho
img_izquierda = img_original[:, :int(w * 0.65)]

# Imagen 2 (Derecha): Desde el 35% del ancho hasta el final
img_derecha = img_original[:, int(w * 0.35):]

print(f"\nDataset Original: Fotografía del Templo ({w}x{h})")
print(f"Generada Foto 1 (Izquierda): {img_izquierda.shape[1]}x{img_izquierda.shape[0]} px")
print(f"Generada Foto 2 (Derecha):   {img_derecha.shape[1]}x{img_derecha.shape[0]} px")
print("El objetivo es que el algoritmo SIFT encuentre automáticamente cómo volver a unirlas.")

# Convertimos a escala de grises (requerimiento matemático para detectar features)
gris_izq = cv2.cvtColor(img_izquierda, cv2.COLOR_RGB2GRAY)
gris_der = cv2.cvtColor(img_derecha, cv2.COLOR_RGB2GRAY)

# ============================================================================
# 2. EXTRACCIÓN DE CARACTERÍSTICAS (SIFT)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DETECCIÓN DE PUNTOS CLAVE Y DESCRIPTORES (SIFT)")
print("=" * 80)

# Inicializar el detector SIFT
# Nota: Desde OpenCV 4.4, SIFT es gratuito (la patente expiró) y está en la rama principal.
sift = cv2.SIFT_create()

print("Buscando Puntos Clave (Keypoints) y calculando Descriptores...")
# detectAndCompute devuelve:
# 1. keypoints (Posición x, y, tamaño, ángulo)
# 2. descriptores (Matriz donde cada fila es un vector de 128D que describe el keypoint)
kp1, des1 = sift.detectAndCompute(gris_izq, None)
kp2, des2 = sift.detectAndCompute(gris_der, None)

print(f"✓ SIFT encontró {len(kp1)} características en la Foto Izquierda.")
print(f"✓ SIFT encontró {len(kp2)} características en la Foto Derecha.")
print(f"  El descriptor es una matriz de tamaño: {des1.shape} (128 variables por punto)")

# Visualización opcional de los Puntos Clave
img_kp = cv2.drawKeypoints(img_izquierda, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0))

fig_kp, ax_kp = plt.subplots(figsize=(10, 6))
ax_kp.imshow(img_kp)
ax_kp.set_title('Puntos Clave SIFT Detectados (Posición, Escala y Orientación)', fontsize=13, fontweight='bold')
ax_kp.axis('off')
#plt.savefig('/home/usuario/01_sift_keypoints.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 01_sift_keypoints.png")
plt.close()

# ============================================================================
# 3. EMPAREJAMIENTO DE CARACTERÍSTICAS (FEATURE MATCHING)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: EMPAREJAMIENTO Y FILTRADO DE OUTLIERS")
print("=" * 80)

# Usaremos un Emparejador de Fuerza Bruta (Brute-Force Matcher)
# Compara el vector 128D de cada punto en la Foto 1 con todos los puntos de la Foto 2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Usamos k=2 para encontrar los dos mejores candidatos para cada punto.
matches = bf.knnMatch(des1, des2, k=2)

print(f"Total de emparejamientos iniciales intentados: {len(matches)}")

# ---------------------------------------------------------
# PRUEBA DE RAZÓN DE LOWE (Lowe's Ratio Test)
# ---------------------------------------------------------
# Para descartar emparejamientos erróneos (falsos positivos), David Lowe 
# (creador de SIFT) demostró que un match es bueno solo si la distancia al 
# mejor candidato (m) es significativamente menor que al segundo mejor (n).
buenos_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        buenos_matches.append(m)

print(f"✓ Emparejamientos confiables tras la Prueba de Lowe: {len(buenos_matches)}")

# Dibujamos las líneas conectando los puntos coincidentes (Solo los 100 mejores para no saturar)
buenos_matches_ordenados = sorted(buenos_matches, key=lambda x: x.distance)
img_matches = cv2.drawMatches(
    img_izquierda, kp1, img_derecha, kp2, 
    buenos_matches_ordenados[:100], None, 
    matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2
)

fig_m, ax_m = plt.subplots(figsize=(16, 6))
ax_m.imshow(img_matches)
ax_m.set_title('Emparejamiento de Puntos (Feature Matching)\n(Conectando características visuales idénticas)', 
               fontsize=13, fontweight='bold')
ax_m.axis('off')
#plt.savefig('/home/usuario/02_sift_matches.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 02_sift_matches.png")
plt.close()

# ============================================================================
# 4. CÁLCULO DE HOMOGRAFÍA Y STITCHING (RANSAC)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: ALINEACIÓN MATEMÁTICA Y FUSIÓN DE IMÁGENES")
print("=" * 80)

# Para calcular la Homografía (Cómo debe deformarse/moverse la Foto 2 para encajar 
# sobre la Foto 1), necesitamos extraer las coordenadas exactas de los buenos matches.
puntos_izq = np.float32([kp1[m.queryIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)
puntos_der = np.float32([kp2[m.trainIdx].pt for m in buenos_matches]).reshape(-1, 1, 2)

print("Calculando Matriz de Homografía mediante RANSAC...")
# RANSAC (Random Sample Consensus) probará combinaciones aleatorias de 4 puntos 
# para encontrar el modelo geométrico que mejor se ajusta a la mayoría de los puntos, 
# descartando cualquier match que haya fallado matemáticamente.
M_homografia, mascara = cv2.findHomography(puntos_der, puntos_izq, cv2.RANSAC, 5.0)

matches_mask = mascara.ravel().tolist()
inliers = sum(matches_mask)
print(f"✓ Homografía calculada exitosamente.")
print(f"  - Inliers (Puntos geométricamente correctos): {inliers} de {len(buenos_matches)}")
print("\nMatriz de Transformación 3x3 generada:")
print(np.round(M_homografia, 3))

# ---------------------------------------------------------
# FUSIÓN (WARP PERSPECTIVE)
# ---------------------------------------------------------
print("\nAplicando transformación perspectiva a la Foto Derecha...")

# Calculamos el tamaño del "lienzo" final (Suficientemente ancho para ambas fotos)
ancho_total = img_izquierda.shape[1] + img_derecha.shape[1]
alto_total = img_izquierda.shape[0]

# Deformamos/Trasladamos la imagen derecha a las coordenadas de la izquierda
panorama = cv2.warpPerspective(img_derecha, M_homografia, (ancho_total, alto_total))

# Superponemos la imagen izquierda en su posición original (0,0)
panorama[0:img_izquierda.shape[0], 0:img_izquierda.shape[1]] = img_izquierda

# ============================================================================
# 5. VISUALIZACIÓN FINAL DEL RESULTADO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DE LA PANORÁMICA GENERADA")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Mostrar las partes originales
ax1 = axes[0]
ax1.imshow(np.hstack((img_izquierda, np.zeros((h, 20, 3), dtype=np.uint8), img_derecha)))
ax1.set_title('Las Dos Fotos Originales (Foto Izquierda | Foto Derecha)\nNótese el solapamiento del 30%', 
              fontsize=13, fontweight='bold')
ax1.axis('off')

# Mostrar el Panorama fusionado
ax2 = axes[1]
# Recortar el área negra sobrante del lienzo
panorama_recortado = panorama[:, :int(w * 1.05)] # Aproximamos el ancho real reconstituido
ax2.imshow(panorama_recortado)
ax2.set_title('Imagen Panorámica Generada Automáticamente\n(Algoritmo SIFT + Homografía)', 
              fontsize=14, fontweight='bold', color='darkblue')
ax2.axis('off')

plt.tight_layout()
#plt.savefig('/home/usuario/03_sift_stitching_panorama.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 03_sift_stitching_panorama.png")
plt.close()

# ============================================================================
# 6. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               SIFT: ALINEACIÓN Y STITCHING DE IMÁGENES (PANORÁMICAS)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. ¿QUÉ HACE ESPECIAL A SIFT?:
   • Antes de SIFT, los algoritmos se perdían si una foto estaba tomada desde 
     más lejos o ligeramente rotada. 
   • SIFT (Scale-Invariant Feature Transform) calcula el "gradiente de 
     orientación" alrededor de cada punto, creando un descriptor de 128D que 
     sigue siendo el mismo sin importar el tamaño o giro de la imagen.

2. EL PIPELINE DE STITCHING:
   • Detección: SIFT encontró los detalles arquitectónicos del templo (esquinas 
     de los techos, bordes de los árboles).
   • Matching: El filtro K-Nearest Neighbors conectó los puntos idénticos de 
     la Foto 1 con la Foto 2.
   • Razón de Lowe: Eliminó coincidencias dudosas (donde el algoritmo estaba 
     confundido entre dos hojas o tejas similares).
   • RANSAC: Resolvió la geometría. Al encontrar un modelo matemático unificado 
     (Homografía), ignoró cualquier coincidencia cruzada errónea restante.

3. LA FUSIÓN FINAL (WARPING):
   • La matriz de Homografía actuó como unas "instrucciones matemáticas" que 
     le dijeron a la computadora: "Para que la Foto 2 encaje, trasládala X 
     píxeles a la derecha". 
   • Al fusionar la matriz original y la trasladada, se generó un panorama 
     perfectamente continuo. Este es el principio fundacional de cómo funcionan 
     las cámaras de los smartphones al crear fotos panorámicas.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_sift_keypoints.png         - Demostración de las características detectadas 
                                y su invarianza de escala (círculos de colores).
02_sift_matches.png           - Mapeo de vectores entre las dos imágenes.
03_sift_stitching_panorama.png- El resultado final: La reconstitución perfecta 
                                del templo mediante álgebra lineal.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Generación de panorámica mediante SIFT completada exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")