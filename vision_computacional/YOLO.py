"""
YOLO (You Only Look Once): Conteo Vehicular Automático y Control de Tráfico
Demostración con dataset proxy real de scikit-learn (LFW People) adaptado para 
simular un sistema inteligente de monitoreo e inventario vehicular vial.

Conceptos:
- YOLO (Ultralytics): Red de detección de objetos en tiempo real de una sola etapa (Single-stage detector).
  Divide la imagen en una cuadrícula y predice directamente las probabilidades de clase y
  las coordenadas de Bounding Boxes (Cajas Delimitadoras).
- Conteo Vehicular Automático: Sistema de visión artificial instalado en cámaras de tráfico 
  para contabilizar el número de automóviles, camiones o motocicletas en tiempo real.
- Formato YOLO: Coordenadas normalizadas [0, 1] compuestas por:
  [class_id, x_center, y_center, width, height]
"""

import os
import shutil
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Dependencias de Machine Learning y Deep Learning
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import torch
import cv2

# Librería Oficial de Ultralytics YOLO
from ultralytics import YOLO

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. CREACIÓN DEL DATASET VIAL Y GENERACIÓN DE ESTRUCTURA FORMATO YOLO
# ============================================================================
print("=" * 80)
print("PASO 1: GENERACIÓN DEL DATASET VIAL (SIMULACIÓN DE TRÁFICO VIAL)")
print("=" * 80)

print("Cargando dataset base real de scikit-learn (LFW People)...")
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=True)
imagenes_base = lfw.images  # Dimensiones: (N, H, W, C) [Valores 0-1]

# Definición de Clases de Vehículos para el Conteo
CLASES_VEHICULOS = {0: 'Auto_Sedán', 1: 'Camión_Carga', 2: 'Motocicleta'}

# Configurar directorios para estructura requerida por YOLO
dataset_dir = "dataset_trafico_yolo"
if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)

for split in ['train', 'val']:
    os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

np.random.seed(42)
num_imagenes_trafico = 80
CANVAS_H, CANVAS_W = 320, 320

print(f"Generando {num_imagenes_trafico} tomas viales sintéticas con Bounding Boxes estilo YOLO...")

indices = list(range(num_imagenes_trafico))
idx_train, idx_val = train_test_split(indices, test_size=0.25, random_state=42)

def generar_escena_vial(idx, split_name):
    # Crear asfalto de carretera con líneas divisorias de carril
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.float32) * 0.25  # Gris oscuro
    for x_carril in [100, 200]:
        for y_dash in range(0, CANVAS_H, 40):
            canvas[y_dash:y_dash+20, x_carril-2:x_carril+2, :] = 0.95  # Línea blanca
            
    num_vehiculos = np.random.randint(2, 6)
    yolo_labels = []
    
    for _ in range(num_vehiculos):
        idx_base = np.random.randint(0, len(imagenes_base))
        vehiculo_img = imagenes_base[idx_base]
        h_v, w_v, _ = vehiculo_img.shape
        
        # Coordenadas aleatorias dentro del canvas
        y1 = np.random.randint(10, CANVAS_H - h_v - 10)
        x1 = np.random.randint(10, CANVAS_W - w_v - 10)
        y2 = y1 + h_v
        x2 = x1 + w_v
        
        # Pegar vehículo en la calzada
        canvas[y1:y2, x1:x2, :] = vehiculo_img
        
        clase_id = np.random.choice([0, 1, 2])
        
        # Convertir Bounding Box a formato YOLO normalizado [x_center, y_center, width, height]
        x_center = ((x1 + x2) / 2.0) / CANVAS_W
        y_center = ((y1 + y2) / 2.0) / CANVAS_H
        width = (w_v) / CANVAS_W
        height = (h_v) / CANVAS_H
        
        yolo_labels.append(f"{clase_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
    # Guardar imagen (convertir BGR a escala uint8)
    img_uint8 = (canvas * 255).astype(np.uint8)
    img_path = os.path.join(dataset_dir, 'images', split_name, f"frame_{idx:03d}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    
    # Guardar archivo de etiquetas .txt
    label_path = os.path.join(dataset_dir, 'labels', split_name, f"frame_{idx:03d}.txt")
    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_labels))

for idx in idx_train:
    generar_escena_vial(idx, 'train')
for idx in idx_val:
    generar_escena_vial(idx, 'val')

# Generar archivo dataset.yaml para Ultralytics
yaml_content = f"""
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

names:
  0: Auto_Sedan
  1: Camion_Carga
  2: Motocicleta
"""
yaml_path = os.path.join(dataset_dir, "dataset_trafico.yaml")
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"✓ Dataset YOLO configurado en '{dataset_dir}'. Archivo YAML creado con éxito.")

# ============================================================================
# 2. CONFIGURACIÓN DEL MODELO DENSENET/YOLO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: CARGA DEL MODELO ULTRALYTICS YOLO")
print("=" * 80)

# Cargar la arquitectura nano de YOLO pre-entrenada
model = YOLO('yolo11n.pt')  # Utiliza pesos YOLO de arquitectura liviana

print("✓ Modelo YOLO cargado e inicializado correctamente.")

# ============================================================================
# 3. ENTRENAMIENTO DE YOLO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAMIENTO EN ESCENARIOS DE TRÁFICO VIAL")
print("=" * 80)

inicio = time.time()

# Proceso de Fine-Tuning de YOLO mediante la API de Ultralytics
results = model.train(
    data=yaml_path,
    epochs=5,
    imgsz=320,
    batch=8,
    verbose=False,
    device='cpu'  # Detecta automáticamete GPU ('cuda:0') si está disponible
)

duracion = time.time() - inicio
print(f"✓ Entrenamiento completado en {duracion:.2f} segundos.")

# ============================================================================
# 4. INFERENCIA Y CONTEO AUTOMÁTICO DE TRÁFICO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: EVALUACIÓN DE INFERENCIA Y CONTEO VEHICULAR DENSIDAD VIAL")
print("=" * 80)

# Seleccionar una toma de prueba
sample_val_img = os.path.join(dataset_dir, 'images', 'val', f"frame_{idx_val[0]:03d}.jpg")

# Inferencia con umbral de confianza ajustado
pred_results = model.predict(source=sample_val_img, conf=0.25, verbose=False)[0]

# Procesar cajas detectadas y realizar el conteo por categoría
boxes = pred_results.boxes
conteo_vehiculos = {0: 0, 1: 0, 2: 0}

if len(boxes) > 0:
    for box in boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in conteo_vehiculos:
            conteo_vehiculos[cls_id] += 1

total_vehiculos = sum(conteo_vehiculos.values())

print("\n--- REPORTE DE CONTEO Y DENSIDAD VEHICULAR EN TIEMPO REAL ---")
print(f"• Autos Sedán Detectados  : {conteo_vehiculos[0]} unidades")
print(f"• Camiones de Carga       : {conteo_vehiculos[1]} unidades")
print(f"• Motocicletas Detectadas : {conteo_vehiculos[2]} unidades")
print(f"• TOTAL VEHÍCULOS EN CARRETERA: {total_vehiculos} unidades")
print("-------------------------------------------------------------")

# Visualización gráfica de resultados
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfica 1: Histograma de Conteo Vehicular
categorias_nombres = [CLASES_VEHICULOS[i] for i in range(3)]
cantidades = [conteo_vehiculos[i] for i in range(3)]
colores = ['crimson', 'teal', 'darkorange']

axes[0].bar(categorias_nombres, cantidades, color=colores, alpha=0.85)
axes[0].set_title('Conteo Vehicular por Tipo de Transporte', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Unidades Detectadas')
axes[0].grid(True, axis='y', alpha=0.3)

# Gráfica 2: Renderizado de la Cámara de Tráfico con Bounding Boxes
img_bgr = cv2.imread(sample_val_img)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

axes[1].imshow(img_rgb)
axes[1].set_title(f'Toma de Control Vial (YOLO Detect)\nVehículos Detectados: {total_vehiculos}', 
                 fontsize=12, fontweight='bold', color='darkblue')

# Dibujar Bounding Boxes en Matplotlib
if len(boxes) > 0:
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, 
                                 edgecolor=colores[cls_id], facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, max(y1 - 5, 12), f"{CLASES_VEHICULOS[cls_id]}: {conf*100:.0f}%", 
                     color='white', fontsize=8, fontweight='bold', 
                     bbox=dict(boxstyle='round,pad=0.2', facecolor=colores[cls_id], alpha=0.8))

axes[1].axis('off')
plt.tight_layout()
plt.savefig('03_yolo_conteo_vehicular.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 03_yolo_conteo_vehicular.png")
plt.show()

# ============================================================================
# 5. RESUMEN TEÓRICO Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN TEÓRICO: YOLO EN SISTEMAS DE TRÁFICO INTELIGENTE")
print("=" * 80)

print("""
1. ARQUITECTURA SINGLE-STAGE (YOLO):
   • A diferencia de Fast/Faster R-CNN (que usan dos etapas con RPN), YOLO pasa la imagen
     una sola vez por la red neuronal, prediciendo simultáneamente las coordenadas de
     las cajas y sus clases.
   • Ideal para cámaras de tráfico y ciudades inteligentes (Smart Cities) gracias a su
     capacidad de procesar vídeo en tiempo real (>60 FPS).""")