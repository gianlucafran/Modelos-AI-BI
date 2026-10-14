"""
SSD (Single Shot MultiBox Detector): Visión en Drones y Monitoreo Aéreo
Demostración con dataset de imágenes reales de scikit-learn y PyTorch Torchvision

Conceptos:
- SSD (Single Shot MultiBox Detector): Arquitectura de detección de objetos en una sola 
  etapa (Single Shot). A diferencia de los modelos de dos etapas como Faster R-CNN, 
  SSD discretiza el espacio de salida de las cajas en un conjunto de "default boxes" 
  sobre diferentes relaciones de aspecto y escalas por mapa de características.
- Visión en Drones (Aerial/Drone Vision): Aplicación crítica donde los objetos de interés 
  (vehículos, personas, obstáculos) aparecen muy pequeños debido a la altitud del vuelo. 
  SSD utiliza múltiples capas convolucionales de diferentes escalas (Feature Pyramids) 
  para detectar simultáneamente objetos grandes y objetos diminutos.
- Imágenes RGB: Esenciales en visión aérea para distinguir contrastes de terreno 
  (carreteras, techos, vegetación) mediante la combinación cromática completa.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings
import time

# Scikit-Learn (Dataset de imágenes reales)
from sklearn.datasets import load_sample_images

# PyTorch y Torchvision (SSD con backbone MobileNetV3 / ResNet)
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms import functional as F

warnings.filterwarnings('ignore')

# Configurar estilo visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. CARGAR DATASET DE IMÁGENES RGB (SCIKIT-LEARN)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR IMAGEN RGB REAL PARA SIMULACIÓN AÉREA (DRON)")
print("=" * 80)

# Cargamos el dataset de imágenes de scikit-learn y seleccionamos una escena 
# abierta para emular una perspectiva cenital o de vuelo de dron.
dataset_imagenes = load_sample_images()
img_escena_np = dataset_imagenes.images[0]  # 'china.jpg' (Escena abierta con estructuras e infraestructuras)

print(f"\nDataset: scikit-learn Sample Images")
print(f"Imagen seleccionada: china.jpg (Simulación de perspectiva aérea / dron)")
print(f"Dimensiones de la imagen RGB: {img_escena_np.shape[1]}x{img_escena_np.shape[0]} píxeles con {img_escena_np.shape[2]} canales (RGB)")

# ============================================================================
# 2. INICIALIZAR EL MODELO SSD (SINGLE SHOT MULTIBOX DETECTOR)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: CONFIGURAR EL MODELO SSD300 (PRE-ENTRENADO COCO)")
print("=" * 80)

print("Cargando pesos pre-entrenados de SSD300 (VGG16 Backbone) para detección en una sola etapa...")
pesos_ssd = SSD300_VGG16_Weights.DEFAULT
modelo_ssd = ssd300_vgg16(weights=pesos_ssd)
modelo_ssd.eval()

# Obtener categorías COCO
categorias_coco = pesos_ssd.meta["categories"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelo_ssd = modelo_ssd.to(device)

print(f"✓ Modelo SSD desplegado exitosamente en dispositivo: {device}")
print(f"✓ Arquitectura optimizada para procesamiento en tiempo real (Edge/Drone AI)")

# ============================================================================
# 3. PROCESAMIENTO E INFERENCIA AÉREA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: EJECUTAR INFERENCIA AÉREA Y DETECCIÓN MULTIESCALA")
print("=" * 80)

# Preprocesar la imagen RGB a Tensor de PyTorch en escala [0, 1]
tensor_img = F.to_tensor(img_escena_np).unsqueeze(0).to(device)

print("Ejecutando pase directo (Single Shot) a través de las capas multiescala de SSD...")
t0 = time.time()
with torch.no_grad():
    predicciones = modelo_ssd(tensor_img)[0]
t_inferencia = (time.time() - t0) * 1000

print(f"✓ Inferencia completada en {t_inferencia:.2f} ms (Alta velocidad para embarcados en drones).")

# Extraer cajas, etiquetas y puntajes
boxes = predicciones['boxes'].cpu().numpy()
labels = predicciones['labels'].cpu().numpy()
scores = predicciones['scores'].cpu().numpy()

# Filtrar por umbral de confianza para drones (>= 35%)
UMBRAL_CONFIANZA = 0.35
indices_validos = np.where(scores >= UMBRAL_CONFIANZA)[0]

print(f"\nObjetos detectados desde la perspectiva del dron (Confianza >= {UMBRAL_CONFIANZA*100:.0f}%):")
conteo_dron = {}

for idx in indices_validos:
    nombre_objeto = categorias_coco[labels[idx]]
    confianza = scores[idx] * 100
    coord = boxes[idx].astype(int)
    
    conteo_dron[nombre_objeto] = conteo_dron.get(nombre_objeto, 0) + 1
    print(f"  - [{nombre_objeto}] Confianza: {confianza:.1f}% | BBox: {coord.tolist()}")

print("\n--- RESUMEN DE MONITOREO AÉREO ---")
for obj, cantidad in conteo_dron.items():
    print(f"  • {obj}: {cantidad} detectado(s)")
print("-----------------------------------")

# ============================================================================
# 4. VISUALIZACIÓN DE LA VISIÓN EN DRONES
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: GENERACIÓN DE GRÁFICOS Y ANÁLISIS DE DETECCIÓN")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Gráfico 1: Imagen RGB con Cajas Delimitadoras (Perspectiva de Dron)
axes[0].imshow(img_escena_np)
axes[0].set_title(f'Monitoreo Aéreo con SSD (Single Shot Detector)\n({len(indices_validos)} elementos detectados en vuelo)', 
                  fontsize=13, fontweight='bold')
axes[0].axis('off')

# Dibujar rectángulos en la imagen
colores_palette = ['#00FF66', '#FF3366', '#33CCFF', '#FFFF33', '#FF9933', '#CC33FF']
for i, idx in enumerate(indices_validos):
    box = boxes[idx]
    color = colores_palette[i % len(colores_palette)]
    ancho = box[2] - box[0]
    alto = box[3] - box[1]
    
    rect = patches.Rectangle((box[0], box[1]), ancho, alto, 
                             linewidth=2.5, edgecolor=color, facecolor='none')
    axes[0].add_patch(rect)
    
    etiqueta = f"{categorias_coco[labels[idx]]} ({scores[idx]*100:.0f}%)"
    axes[0].text(box[0], max(15, box[1] - 5), etiqueta, 
                 bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2),
                 fontsize=9, fontweight='bold', color='black')

# Gráfico 2: Distribución de Detecciones en el Monitoreo Aéreo
objetos_nombres = list(conteo_dron.keys())
cantidades = list(conteo_dron.values())

if len(objetos_nombres) > 0:
    sns.barplot(x=cantidades, y=objetos_nombres, palette='magma', ax=axes[1])
    axes[1].set_title('Conteo de Elementos en Zona de Estudio (Dron)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Cantidad')
    axes[1].set_ylabel('Clase Detectada')
    axes[1].grid(axis='x', alpha=0.3)
else:
    axes[1].text(0.5, 0.5, "No se detectaron objetos en el frame actual", 
                 ha='center', va='center', fontsize=12)
    axes[1].axis('off')

plt.tight_layout()
plt.savefig('/home/usuario/01_ssd_vision_drones.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_ssd_vision_drones.png")
plt.close()

# ============================================================================
# 5. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN TEÓRICO")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                SSD (SINGLE SHOT MULTIBOX DETECTOR) EN DRONES                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. POR QUÉ SSD ES IDEAL PARA COMPUTACIÓN EMBARCADA (DRONES):
   • Los vehículos aéreos no tripulados (UAVs) operan con baterías limitadas y 
     requieren tiempos de respuesta ultrarrápidos para evitar colisiones o rastrear objetivos.
   • A diferencia de las arquitecturas de dos etapas (como Faster R-CNN) que 
     consumen muchos recursos proponiendo regiones antes de clasificar, SSD evalúa 
     las cajas delimitadoras y las clases simultáneamente en un **solo pase directo** 
     (Single Shot), logrando tasas de cuadros (FPS) muy superiores.

2. DETECCIÓN MULTIESCALA:
   • En visiones cenitales o de dron, la escala de los objetos cambia drásticamente 
     según la altitud de vuelo. SSD utiliza mapas de características de diferentes 
     resoluciones de forma piramidal, permitiendo que las capas superiores detecten 
     estructuras grandes y las capas inferiores detecten objetos pequeños con gran precisión.

3. EL ROL DE LAS IMÁGENES RGB:
   • El procesamiento en color real (RGB) facilita la segmentación visual del terreno, 
     permitiendo al modelo distinguir vehículos, infraestructuras y obstáculos 
     independientemente de las condiciones lumínicas cenitales.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_ssd_vision_drones.png - Visualización de cajas delimitadoras sobre la 
                           perspectiva aérea y gráfico estadístico de elementos detectados.
═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Pipeline de Visión en Drones con SSD completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")