"""
LoG (Laplacian of Gaussian): Filtro de Contornos (Edge Detection)
Demostración con dataset de imágenes reales (china.jpg) de scikit-learn

Conceptos:
- Detección de Contornos (Edge Detection): Proceso de identificar puntos en una imagen 
  donde el brillo cambia drásticamente.
- Operador Laplaciano: Calcula la segunda derivada espacial de la imagen. Los 
  "cruces por cero" (zero-crossings) de esta derivada indican los bordes.
- El problema del ruido: El Laplaciano es extremadamente sensible al ruido. Si se 
  aplica directamente, los píxeles ruidosos generarán falsos contornos.
- Laplacian of Gaussian (LoG): Soluciona el problema aplicando primero un filtro 
  Gaussiano para suavizar la imagen (eliminar ruido) y luego el Laplaciano. 
  Matemáticamente, ambas operaciones se combinan en un solo filtro.
- Sigma (σ): Controla el radio del desenfoque gaussiano. A mayor sigma, se detectan 
  solo los contornos estructurales más gruesos y se ignoran las texturas finas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from scipy.ndimage import gaussian_laplace
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. CARGAR Y PREPARAR EL DATASET (IMAGEN REAL DE SCIKIT-LEARN)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR LA IMAGEN Y CONVERTIR A ESCALA DE GRISES")
print("=" * 80)

# Utilizamos una imagen real de alta resolución proporcionada por scikit-learn
print("Cargando imagen 'china.jpg' del dataset de scikit-learn...")
imagen_color = load_sample_image("china.jpg")

# Los filtros de contorno como LoG operan sobre la intensidad luminosa, 
# por lo que debemos convertir la imagen RGB a Escala de Grises.
# Usamos la fórmula estándar de luminancia: Y = 0.299*R + 0.587*G + 0.114*B
def rgb_a_escala_grises(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

imagen_grises = rgb_a_escala_grises(imagen_color)

print(f"\nDataset: Fotografía del Templo (China)")
print(f"Dimensiones originales (RGB): {imagen_color.shape}")
print(f"Dimensiones en escala de grises: {imagen_grises.shape}")

# ============================================================================
# 2. APLICACIÓN DEL FILTRO LoG (LAPLACIAN OF GAUSSIAN)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: APLICACIÓN DEL FILTRO LoG CON DIFERENTES ESCALAS (SIGMA)")
print("=" * 80)

# El parámetro Sigma (σ) define la desviación estándar de la campana de Gauss.
# Probaremos tres niveles para ver cómo cambia la detección de bordes.

# Sigma = 1.0 (Bordes muy finos y texturas)
print("Calculando LoG con Sigma = 1.0 (Detecta detalles finos y texturas)...")
log_fino = gaussian_laplace(imagen_grises, sigma=1.0)

# Sigma = 2.0 (Bordes intermedios, supresión moderada de ruido)
print("Calculando LoG con Sigma = 2.0 (Balance entre detalle y estructura)...")
log_medio = gaussian_laplace(imagen_grises, sigma=2.0)

# Sigma = 3.5 (Solo bordes estructurales fuertes)
print("Calculando LoG con Sigma = 3.5 (Suprime textura, deja bordes estructurales gruesos)...")
log_grueso = gaussian_laplace(imagen_grises, sigma=3.5)

print("\n✓ Filtros matemáticos de segunda derivada aplicados exitosamente.")

# ============================================================================
# 3. EXTRACCIÓN DE CONTORNOS (CRUCE POR CERO Y UMBRALIZACIÓN)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: PROCESAMIENTO VISUAL DE LOS BORDES")
print("=" * 80)

# En la teoría pura, los contornos son los "cruces por cero" de la imagen LoG.
# Para visualización práctica en visión por computadora, solemos tomar el valor 
# absoluto de la respuesta del filtro y aplicar una mejora de contraste o umbral.

# Tomamos el valor absoluto para que todos los cambios fuertes sean positivos (blancos)
contornos_fino = np.abs(log_fino)
contornos_medio = np.abs(log_medio)
contornos_grueso = np.abs(log_grueso)

# Función para normalizar visualmente (0 a 255)
def normalizar_visual(img_log):
    # Usamos el percentil 98 para evitar que píxeles anómalos oscurezcan el resto
    max_val = np.percentile(img_log, 98) 
    img_norm = np.clip(img_log / max_val, 0, 1)
    return img_norm

contornos_fino_vis = normalizar_visual(contornos_fino)
contornos_medio_vis = normalizar_visual(contornos_medio)
contornos_grueso_vis = normalizar_visual(contornos_grueso)

print("Los mapas de bordes han sido normalizados para su visualización óptica.")

# ============================================================================
# 4. VISUALIZACIÓN: COMPARATIVA DE NIVELES DE DETECCIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: GENERACIÓN DE GALERÍA DE RESULTADOS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# 1. Imagen Original
axes[0].imshow(imagen_grises, cmap='gray')
axes[0].set_title('Imagen Original (Escala de Grises)', fontsize=13, fontweight='bold')
axes[0].axis('off')

# 2. LoG Sigma = 1.0
axes[1].imshow(contornos_fino_vis, cmap='gray')
axes[1].set_title('Filtro LoG (Sigma = 1.0)\nTexturas Finas (Hojas, tejas)', 
                  fontsize=13, fontweight='bold')
axes[1].axis('off')

# 3. LoG Sigma = 2.0
axes[2].imshow(contornos_medio_vis, cmap='gray')
axes[2].set_title('Filtro LoG (Sigma = 2.0)\nBalance Detalle-Estructura', 
                  fontsize=13, fontweight='bold')
axes[2].axis('off')

# 4. LoG Sigma = 3.5
axes[3].imshow(contornos_grueso_vis, cmap='gray')
axes[3].set_title('Filtro LoG (Sigma = 3.5)\nEstructura Arquitectónica Principal', 
                  fontsize=13, fontweight='bold')
axes[3].axis('off')

# Añadir un recuadro explicativo general
fig.text(0.5, 0.02, "Nota: A medida que Sigma (σ) aumenta, el filtro ignora el ruido y "
                    "las texturas superficiales, resaltando solo la silueta estructural del edificio.", 
         ha="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 1])
#plt.savefig('/home/usuario/01_filtro_log_contornos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_filtro_log_contornos.png")
plt.close()

# ============================================================================
# 5. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               LoG (LAPLACIAN OF GAUSSIAN) EN VISIÓN POR COMPUTADORA          ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. FUNDAMENTO MATEMÁTICO:
   • El filtro LoG es el santo grial clásico de la detección de contornos. 
   • Primero, la operación Gaussiana "desenfoca" la imagen. Luego, el Operador 
     Laplaciano (∇²f) busca cambios bruscos en la intensidad de los píxeles 
     (donde la pendiente de la luz sube o baja rápidamente).

2. EL PODER DEL PARÁMETRO SIGMA (σ):
   • En el análisis "Multi-escala" (Paso 4):
     - Sigma = 1.0: Extrae información de alta frecuencia. El filtro marca 
       cada hoja de los árboles y cada línea diminuta de las tejas del templo.
     - Sigma = 3.5: Actúa como un filtro de baja frecuencia. Ignora las hojas 
       y los detalles finos (los difumina antes de aplicar el Laplaciano), 
       dejando únicamente los bordes macizos de las paredes, ventanas y tejados.

3. ¿POR QUÉ NO USAR SOLO EL LAPLACIANO?:
   • Si aplicáramos la segunda derivada (Laplaciano) directamente sin el 
     suavizado Gaussiano previo, el algoritmo detectaría el más mínimo ruido 
     del sensor de la cámara como si fuera un "contorno", arruinando por 
     completo la capacidad de extraer formas reconocibles.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           LIBRERÍAS UTILIZADAS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

- Scikit-Learn: Proporciona la imagen real de prueba (`load_sample_image`).
- SciPy (`scipy.ndimage.gaussian_laplace`): Ejecuta el convolucionado matemático 
  multidimensional, aplicando el kernel de LoG a la matriz de píxeles 2D.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_filtro_log_contornos.png - Matriz de imágenes comparando la fotografía original 
                              en escala de grises con sus mapas de contornos a 
                              tres escalas diferentes de desviación estándar (Sigma).

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de contornos mediante Laplacian of Gaussian completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")