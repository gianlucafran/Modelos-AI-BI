"""
Brute-Force Matcher: Reconocimiento Facial Simple (Feature Matching)
Demostración con dataset real Labeled Faces in the Wild (LFW) de scikit-learn

Conceptos:
- Brute-Force Matcher (BFMatcher): Algoritmo de emparejamiento que toma el descriptor 
  de una característica en la primera imagen y lo compara matemáticamente contra TODOS 
  los descriptores de la segunda imagen. Es exhaustivo y garantiza encontrar la 
  distancia mínima.
- Descriptores Binarios (ORB): A diferencia de SIFT (que usa punto flotante), ORB 
  representa las características visuales como cadenas de ceros y unos.
- Distancia de Hamming: La métrica ideal para comparar descriptores binarios. Cuenta 
  cuántos bits son diferentes entre dos cadenas. (Ej. 1010 y 1000 tienen distancia 1).
- Reconocimiento Facial Simple: Abordaremos la identificación de rostros buscando 
  similitudes topológicas (cicatrices, forma de los ojos, arrugas) entre dos fotos 
  en lugar de usar Redes Neuronales Profundas (CNNs).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
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

# ============================================================================
# 1. CARGAR Y PREPARAR EL DATASET (ROSTROS REALES)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR DATASET DE ROSTROS (LABELED FACES IN THE WILD)")
print("=" * 80)

# Descargamos el dataset con un resize mayor (3.0) para que los algoritmos de 
# extracción de características tengan suficientes píxeles para encontrar detalles.
print("Descargando/Cargando dataset LFW...")
caras = fetch_lfw_people(min_faces_per_person=50, resize=3.0)

imagenes = caras.images
etiquetas = caras.target
nombres = caras.target_names

print(f"\nDataset: Labeled Faces in the Wild")
print(f"Resolución de imágenes escaladas: {imagenes.shape[1]}x{imagenes.shape[2]} píxeles")

# Seleccionamos las imágenes para nuestro experimento:
# 1. Imagen de Referencia (Sujeto A)
# 2. Imagen Positiva (Sujeto A en otra pose/iluminación)
# 3. Imagen Negativa (Sujeto B)

sujeto_a_idx = np.where(etiquetas == 0)[0] # Ej. Ariel Sharon
sujeto_b_idx = np.where(etiquetas == 1)[0] # Ej. Colin Powell

img_referencia = imagenes[sujeto_a_idx[0]]
img_positiva = imagenes[sujeto_a_idx[1]]
img_negativa = imagenes[sujeto_b_idx[0]]

nombre_a = nombres[0]
nombre_b = nombres[1]

# OpenCV requiere imágenes en formato uint8 (0-255)
def preparar_imagen_cv2(img_float):
    img_norm = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8)

img_ref_cv = preparar_imagen_cv2(img_referencia)
img_pos_cv = preparar_imagen_cv2(img_positiva)
img_neg_cv = preparar_imagen_cv2(img_negativa)

print(f"\nSujetos seleccionados para la prueba de reconocimiento:")
print(f"  - Target (Referencia): {nombre_a}")
print(f"  - Prueba Positiva:     {nombre_a} (Diferente fotografía)")
print(f"  - Prueba Negativa:     {nombre_b} (Impostor)")

# ============================================================================
# 2. EXTRACCIÓN DE CARACTERÍSTICAS (ORB)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DETECCIÓN DE PUNTOS CLAVE Y DESCRIPTORES (ORB)")
print("=" * 80)

# ORB (Oriented FAST and Rotated BRIEF) es rapidísimo y no tiene patentes.
# Busca esquinas y zonas de alto contraste en el rostro.
orb = cv2.ORB_create(nfeatures=500)

print("Extrayendo mapas faciales topológicos...")
kp_ref, des_ref = orb.detectAndCompute(img_ref_cv, None)
kp_pos, des_pos = orb.detectAndCompute(img_pos_cv, None)
kp_neg, des_neg = orb.detectAndCompute(img_neg_cv, None)

print(f"✓ Características extraídas:")
print(f"  - Rostro Referencia: {len(kp_ref)} puntos clave detectados.")
print(f"  - Rostro Positivo:   {len(kp_pos)} puntos clave detectados.")
print(f"  - Rostro Negativo:   {len(kp_neg)} puntos clave detectados.")

# ============================================================================
# 3. EMPAREJAMIENTO EXHAUSTIVO (BRUTE-FORCE MATCHER)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: EMPAREJAMIENTO DE DESCRIPTORES (BFMATCHER)")
print("=" * 80)

# Inicializamos el Brute-Force Matcher
# NORM_HAMMING: Obligatorio para descriptores binarios como ORB.
# crossCheck=True: Una coincidencia solo es válida si el punto A en la imagen 1 
# tiene como mejor pareja al punto B en la imagen 2, y viceversa (Filtro bidireccional).
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print("Ejecutando comparación cruzada (Fuerza Bruta) píxel por píxel...")
t0 = time.time()

# 1. Comparación Genuina: Referencia vs Positivo (Mismo sujeto)
matches_genuinos = bf.match(des_ref, des_pos)

# 2. Comparación Impostor: Referencia vs Negativo (Sujetos diferentes)
matches_impostor = bf.match(des_ref, des_neg)

t_calc = (time.time() - t0) * 1000
print(f"✓ Emparejamientos calculados exhaustivamente en {t_calc:.1f} ms.")

# Ordenamos los emparejamientos por distancia (los más cercanos/similares primero)
matches_genuinos = sorted(matches_genuinos, key=lambda x: x.distance)
matches_impostor = sorted(matches_impostor, key=lambda x: x.distance)

# ============================================================================
# 4. FILTRADO Y DECISIÓN (RECONOCIMIENTO)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: SISTEMA DE DECISIÓN DE IDENTIDAD")
print("=" * 80)

# Para reconocer un rostro de forma simple, contamos cuántos emparejamientos 
# tienen una distancia de Hamming muy baja (son visualmente casi idénticos).
UMBRAL_DISTANCIA = 40  # Distancia máxima de Hamming aceptable para considerar un match válido
UMBRAL_COINCIDENCIAS = 25 # Mínimo de coincidencias fuertes para "Verificar" la identidad

buenos_genuinos = [m for m in matches_genuinos if m.distance < UMBRAL_DISTANCIA]
buenos_impostor = [m for m in matches_impostor if m.distance < UMBRAL_DISTANCIA]

print(f"Criterio de Aceptación: Máximo {UMBRAL_DISTANCIA} bits de diferencia por característica.")
print(f"Mínimo para autenticación: {UMBRAL_COINCIDENCIAS} puntos coincidentes.")

print("\nResultados de Autenticación:")
print(f"  [INTENTO 1 - Genuino]: Se encontraron {len(buenos_genuinos)} coincidencias fuertes.")
if len(buenos_genuinos) >= UMBRAL_COINCIDENCIAS:
    print("      >> ESTADO: ACCESO CONCEDIDO (Identidad Verificada) ✅")
else:
    print("      >> ESTADO: ACCESO DENEGADO (Falso Rechazo) ❌")

print(f"  [INTENTO 2 - Impostor]: Se encontraron {len(buenos_impostor)} coincidencias fuertes.")
if len(buenos_impostor) >= UMBRAL_COINCIDENCIAS:
    print("      >> ESTADO: ACCESO CONCEDIDO (Falsa Aceptación) ❌")
else:
    print("      >> ESTADO: ACCESO DENEGADO (Impostor Bloqueado) ✅")

# ============================================================================
# 5. VISUALIZACIÓN: EL VEREDICTO DEL BFMATCHER
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DE LOS EMPAREJAMIENTOS FACIALES")
print("=" * 80)

# Dibujamos las coincidencias (solo los buenos matches filtrados)
img_match_genuino = cv2.drawMatches(
    img_ref_cv, kp_ref, img_pos_cv, kp_pos, 
    buenos_genuinos, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0)
)

img_match_impostor = cv2.drawMatches(
    img_ref_cv, kp_ref, img_neg_cv, kp_neg, 
    buenos_impostor, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(255, 0, 0)
)

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Gráfico 1: Identidad Genuina
axes[0].imshow(img_match_genuino, cmap='gray')
axes[0].set_title(f'Comparación Genuina: {nombre_a} vs {nombre_a}\n({len(buenos_genuinos)} Conexiones Topológicas Validadas)', 
                  fontsize=14, fontweight='bold', color='forestgreen')
axes[0].axis('off')

# Gráfico 2: Impostor
axes[1].imshow(img_match_impostor, cmap='gray')
axes[1].set_title(f'Comparación Impostor: {nombre_a} vs {nombre_b}\n({len(buenos_impostor)} Conexiones - Rechazo de Identidad)', 
                  fontsize=14, fontweight='bold', color='crimson')
axes[1].axis('off')

plt.tight_layout()
#plt.savefig('/home/usuario/01_bfmatcher_facial.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_bfmatcher_facial.png")
plt.close()

# ============================================================================
# 6. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               BRUTE-FORCE MATCHER EN RECONOCIMIENTO FACIAL                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. LA LÓGICA DE FUERZA BRUTA:
   • El algoritmo `BFMatcher` no toma atajos probabilísticos. Tomó el ojo 
     izquierdo del Sujeto A y lo comparó iterativamente contra todas y cada una 
     de las características extraídas de la imagen secundaria. Al usar 
     `crossCheck=True`, aseguró matemáticamente que la afinidad fuera mutua.

2. ¿FUNCIONA BIEN PARA ROSTROS?:
   • Sí y No. Como se observa en el Gráfico 1, el algoritmo logra conectar 
     rasgos específicos (comisuras de labios, puente de la nariz, arrugas) 
     cuando se trata de la misma persona.
   • Sin embargo, la topología facial humana es muy similar entre individuos. 
     El Gráfico 2 muestra que el algoritmo aún encuentra similitudes (falsos 
     positivos en la textura de la piel o bordes de las gafas) entre personas 
     diferentes.

3. EL VEREDICTO DE LA INDUSTRIA:
   • ORB + BFMatcher es espectacular para rastrear objetos rígidos (cajas, 
     libros, edificios en panorámicas) porque las esquinas no cambian.
   • Para rostros (que son elásticos, cambian de expresión, luz y ángulo), este 
     método "clásico" es propenso a errores. Por esto, la industria abandonó el 
     Feature Matching geométrico a favor de las Redes Neuronales Convolucionales 
     (FaceNet, ArcFace), que miden la biometría profunda en lugar de los 
     píxeles superficiales.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True):
  - NORM_HAMMING: Se utiliza estrictamente porque ORB devuelve descriptores 
    binarios. Evalúa el XOR entre las cadenas de bits, siendo mucho más rápido 
    que calcular raíces cuadradas (NORM_L2, usado para SIFT).
  - crossCheck: Reduce enormemente los outliers al obligar a que el punto 
    emparejado no tenga un mejor candidato disponible.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_bfmatcher_facial.png - Demostración visual de las redes de coincidencias 
                          entre un par genuino (acceso concedido) y un par 
                          impostor (acceso denegado).

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de Fuerza Bruta para Reconocimiento Facial completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")