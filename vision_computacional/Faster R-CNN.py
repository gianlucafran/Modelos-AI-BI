"""
Fast / Faster R-CNN: Inventario Visual (Detección de Objetos y Conteo en Almacén)
Demostración con dataset proxy de scikit-learn (LFW People / Labeled Faces in the Wild) 
adaptado para simular un sistema de inventario automatizado.

Conceptos:
- Fast R-CNN (2015) & Faster R-CNN (2017): Modelos de detección de objetos en 2 etapas.
  1. Extracción de características de la imagen completa con una red backbone (ResNet/FPN).
  2. Generación de propuestas de región (RPN - Region Proposal Network).
  3. RoI Pooling / RoI Align para extraer características fijas de cada región.
  4. Cabezales finales de Clasificación (¿Qué producto es?) y Regresión de Cajas (¿Dónde está?).
- Inventario Visual: Permite a las cámaras de un almacén o tienda física localizar 
  e identificar productos en estanterías, calculando automáticamente el conteo de stock.
"""

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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# API moderna de Torchvision para detección y transformaciones v2
import torchvision.transforms.v2 as transforms_v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. CREACIÓN DEL DATASET DE INVENTARIO VISUAL (ESTANTES Y PRODUCTOS)
# ============================================================================
print("=" * 80)
print("PASO 1: GENERACIÓN DEL DATASET DE INVENTARIO (SIMULACIÓN DE ESTANTERÍAS)")
print("=" * 80)

print("Cargando dataset base real de scikit-learn (LFW People)...")
# Cargamos imágenes en escala de grises / color de LFW People
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=True)
imagenes_base = lfw.images  # Forma: (N, H, W, C) [valores entre 0 y 1]

print("Generando imágenes sintéticas de estanterías con objetos de inventario...")

np.random.seed(42)
num_estantes = 120
CANVAS_SIZE = (256, 256)  # Alto x Ancho de las imágenes del almacén

dataset_imagenes = []
dataset_targets = []

# Definimos 2 clases de productos de inventario (además del Fondo = 0)
# Clase 1: "Producto Tipo A" (Cajas/Lotes)
# Clase 2: "Producto Tipo B" (Paquetes)
CATEGORIAS = {1: "Producto_A (Caja)", 2: "Producto_B (Paquete)"}

for i in range(num_estantes):
    # Crear fondo neutro para el estante
    canvas = np.ones((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), dtype=np.float32) * 0.85
    
    # Agregar líneas horizontales que simulan los estantes
    canvas[80:85, :, :] = 0.3
    canvas[170:175, :, :] = 0.3
    
    num_objetos = np.random.randint(1, 4)  # Entre 1 y 3 objetos por estante
    boxes = []
    labels = []
    
    for _ in range(num_objetos):
        idx_base = np.random.randint(0, len(imagenes_base))
        item_img = imagenes_base[idx_base]  # Imagen del producto
        h_obj, w_obj, _ = item_img.shape
        
        # Posición aleatoria en el canvas
        y1 = np.random.randint(10, CANVAS_SIZE[0] - h_obj - 10)
        x1 = np.random.randint(10, CANVAS_SIZE[1] - w_obj - 10)
        y2 = y1 + h_obj
        x2 = x1 + w_obj
        
        # Pegar objeto en el estante
        canvas[y1:y2, x1:x2, :] = item_img
        
        # Etiqueta aleatoria (1 o 2)
        clase_id = np.random.choice([1, 2])
        
        boxes.append([x1, y1, x2, y2])
        labels.append(clase_id)
        
    dataset_imagenes.append(canvas)
    dataset_targets.append({
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64)
    })

print(f"✓ Dataset generado: {num_estantes} imágenes de estanterías ({CANVAS_SIZE[0]}x{CANVAS_SIZE[1]} px).")
print(f"✓ Total de objetos etiquetados para conteo de inventario: {sum(len(t['labels']) for t in dataset_targets)}")

# Dividir en Entrenamiento y Prueba
indices = list(range(num_estantes))
idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

# Dataset personalizado para PyTorch / Torchvision Object Detection
class InventarioDataset(Dataset):
    def __init__(self, imagenes, targets, indices):
        self.imagenes = [imagenes[i] for i in indices]
        self.targets = [targets[i] for i in indices]

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, idx):
        img_np = self.imagenes[idx]
        # Convertir de (H, W, C) a Tensor de PyTorch (C, H, W)
        tensor_img = torch.from_numpy(img_np).permute(2, 0, 1)
        
        target = {
            'boxes': self.targets[idx]['boxes'],
            'labels': self.targets[idx]['labels']
        }
        return tensor_img, target

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = InventarioDataset(dataset_imagenes, dataset_targets, idx_train)
test_dataset = InventarioDataset(dataset_imagenes, dataset_targets, idx_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# ============================================================================
# 2. DEFINICIÓN DEL MODELO FASTER R-CNN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: CONFIGURACIÓN DE FASTER R-CNN (RESNET-50 FPN V2)")
print("=" * 80)

# Cargar Faster R-CNN con arquitectura optimizada V2
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
modelo_fastrcnn = fasterrcnn_resnet50_fpn_v2(weights=weights)

# Modificar el cabezal de clasificación para nuestras clases (Fondo + Producto_A + Producto_B = 3)
num_classes = 3  # 0 = Background, 1 = Producto_A, 2 = Producto_B
in_features = modelo_fastrcnn.roi_heads.box_predictor.cls_score.in_features

# Reemplazar el predictor final de cajas y clases
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
modelo_fastrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelo_fastrcnn = modelo_fastrcnn.to(device)

print(f"✓ Arquitectura Faster R-CNN cargada en dispositivo: {device}.")
print(f"✓ Cabezal RoI configurado para {num_classes} clases (Fondo + 2 tipos de productos).")

# ============================================================================
# 3. ENTRENAMIENTO DE LA RED PARA DETECCIÓN Y CONTEO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAMIENTO PARA INVENTARIO AUTOMATIZADO")
print("=" * 80)

# Seleccionar parámetros optimizables (cabezal de detección)
params = [p for p in modelo_fastrcnn.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)

epocas = 5
historial_pérdida = []

print("Iniciando entrenamiento del detector de objetos...")
inicio = time.time()

for e in range(epocas):
    modelo_fastrcnn.train()
    loss_acumulada = 0.0
    
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # En modo train, Faster R-CNN calcula automáticamente las pérdidas de RPN y RoI
        loss_dict = modelo_fastrcnn(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        loss_acumulada += losses.item()
        
    pérdida_media = loss_acumulada / len(train_loader)
    historial_pérdida.append(pérdida_media)
    print(f"  [Época {e+1:02d}/{epocas:02d}] Loss Total (RPN + RoI Box/Cls): {pérdida_media:.4f}")

duracion = time.time() - inicio
print(f"✓ Modelo entrenado correctamente en {duracion:.2f} segundos.")

# ============================================================================
# 4. VISUALIZACIÓN Y CONTEO DE INVENTARIO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: EVALUACIÓN Y CONTEO DE STOCK EN ESTANTERÍA")
print("=" * 80)

modelo_fastrcnn.eval()

# Evaluamos la primera muestra de validación
images_test, targets_test = next(iter(test_loader))
img_eval = images_test[0].to(device)

with torch.no_grad():
    prediction = modelo_fastrcnn([img_eval])[0]

# Filtrar por umbral de confianza (Score Threshold)
SCORE_THRESHOLD = 0.50
boxes_pred = prediction['boxes'].cpu().numpy()
labels_pred = prediction['labels'].cpu().numpy()
scores_pred = prediction['scores'].cpu().numpy()

indices_validos = scores_pred >= SCORE_THRESHOLD
boxes_filtradas = boxes_pred[indices_validos]
labels_filtradas = labels_pred[indices_validos]
scores_filtrados = scores_pred[indices_validos]

# Realizar Conteo Automático de Inventario
conteo_stock = {1: 0, 2: 0}
for lbl in labels_filtradas:
    if lbl in conteo_stock:
        conteo_stock[lbl] += 1

print("\n--- REPORTE DE INVENTARIO AUTOMÁTICO ---")
print(f"• Total de {CATEGORIAS[1]} detectados: {conteo_stock[1]} unidades")
print(f"• Total de {CATEGORIAS[2]} detectados: {conteo_stock[2]} unidades")
print(f"• Unidades Totales en Estante: {sum(conteo_stock.values())} unidades")
print("-----------------------------------------")

# Dibujar resultados
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfica 1: Pérdida durante el entrenamiento
axes[0].plot(range(1, epocas + 1), historial_pérdida, marker='o', color='crimson', linewidth=2)
axes[0].set_title('Convergencia del Modelo (Loss de Detección)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Pérdida (Loss)')
axes[0].grid(True, alpha=0.3)

# Gráfica 2: Detección e Identificación en Estantería
img_vis = img_eval.cpu().permute(1, 2, 0).numpy()
axes[1].imshow(img_vis)
axes[1].set_title(f'Inventario Visual Detectado\nConteo Total: {sum(conteo_stock.values())} items', 
                 fontsize=12, fontweight='bold', color='darkblue')

colors = {1: 'lime', 2: 'cyan'}

for box, label, score in zip(boxes_filtradas, labels_filtradas, scores_filtrados):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                             edgecolor=colors.get(label, 'red'), facecolor='none')
    axes[1].add_patch(rect)
    axes[1].text(x1, max(y1 - 5, 10), f"{CATEGORIAS.get(label, 'Item')}: {score*100:.0f}%", 
                 color='black', fontsize=9, fontweight='bold', 
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=colors.get(label, 'yellow'), alpha=0.8))

axes[1].axis('off')
plt.tight_layout()
plt.savefig('02_fast_rcnn_inventario.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 0")