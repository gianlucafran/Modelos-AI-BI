"""
Mask R-CNN (He et al., 2017): Segmentación de Instancias y Conteo Celular Automático
Aplicación: Detección, localización, segmentación a nivel de píxel y conteo individual de células / núcleos en imágenes microscópicas.

Conceptos Clave de Mask R-CNN para Conteo Celular:
1. Segmentación de Instancias vs. Segmentación Semántica:
   A diferencia de U-Net (que detecta la clase "célula" como una sola masa fusionada), 
   Mask R-CNN identifica y separa CADA objeto/célula de manera independiente (Instancia 1, Instancia 2...).
   Esto permite realizar un conteo celular exacto incluso cuando las células están adyacentes o solapadas.
2. RoIAlign (Region of Interest Align):
   Reemplaza el RoIPooling tradicional evitando el redondeo de coordenadas. Conserva la precisión 
   subpíxel indispensable para predecir máscaras exactas y finas.
3. Rama de Máscara (Fully Convolutional Network Branch):
   Añade una rama paralela a las cabezas de clasificación y regresión de Bounding Box. 
   Esta rama predice una máscara binaria $K \times m \times m$ para cada ROI seleccionada por la RPN.
4. Dataset Proxy (scikit-learn + Micrografía Sintética de Núcleos):
   Adaptamos el dataset 'Olivetti Faces' de scikit-learn como fondo de tejido celular y simulamos 
   múltiples núcleos elípticos/células individuales con Bounding Boxes, Máscaras de Instancia 
   y Etiquetas para alimentar el entrenamiento de PyTorch (torchvision Mask R-CNN).
"""

import time
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ML & Computer Vision dependencies
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# PyTorch & Torchvision framework para la arquitectura Mask R-CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ==========================================================
# 1. PREPARACIÓN DEL DATASET PROXY (scikit-learn + Instancias Celulares)
# ==========================================================
print("=== 1. Generando Dataset Proxy Microscópico con Instancias Celulares ===")
raw_data = fetch_olivetti_faces(shuffle=True, random_state=42)
raw_images = raw_data.images  # (400, 64, 64) en [0, 1]

images_dataset = []
targets_dataset = []

for img_idx, img in enumerate(raw_images):
    # Simular fondo de tejido/sustrato microscópico en escala de grises
    base_tissue = (img * 255).astype(np.uint8)
    base_tissue = cv2.GaussianBlur(base_tissue, (3, 3), 0)
    
    # Número aleatorio de células por imagen (entre 3 y 7 células)
    num_cells = np.random.randint(3, 8)
    
    boxes = []
    masks = []
    labels = []
    
    for _ in range(num_cells):
        cx, cy = np.random.randint(10, 54), np.random.randint(10, 54)
        rx, ry = np.random.randint(4, 8), np.random.randint(4, 8)
        
        # Generar máscara individual (instancia única)
        instance_mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.ellipse(instance_mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
        
        # Bounding box [xmin, ymin, xmax, ymax] a partir de la máscara
        pos = np.where(instance_mask)
        if len(pos[0]) == 0:
            continue
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        
        if xmax <= xmin or ymax <= ymin:
            continue
            
        # Inyectar la célula hiperintensa en la imagen base
        cell_patch = np.ones((64, 64), dtype=np.uint8) * 220
        cell_patch = cv2.GaussianBlur(cell_patch, (5, 5), 0)
        base_tissue = np.where(instance_mask == 1, cv2.addWeighted(base_tissue, 0.3, cell_patch, 0.7, 0), base_tissue)
        
        boxes.append([xmin, ymin, xmax, ymax])
        masks.append(instance_mask)
        labels.append(1)  # Clase 1: Célula/Núcleo
        
    if len(boxes) > 0:
        images_dataset.append(base_tissue)
        targets_dataset.append({
            'boxes': np.array(boxes, dtype=np.float32),
            'masks': np.array(masks, dtype=np.uint8),
            'labels': np.array(labels, dtype=np.int64)
        })

print(f"Dataset generado: {len(images_dataset)} imágenes microscópicas procesadas.")


# ==========================================================
# 2. CONSTRUCCIÓN DEL DATASET Y DATALOADER EN PYTORCH
# ==========================================================
class CellSegmentationDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convertir imagen a RGB 3 canales (requerido por backbone ResNet de torchvision)
        img_gray = self.images[idx]
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

        target = {
            'boxes': torch.tensor(self.targets[idx]['boxes'], dtype=torch.float32),
            'labels': torch.tensor(self.targets[idx]['labels'], dtype=torch.int64),
            'masks': torch.tensor(self.targets[idx]['masks'], dtype=torch.uint8)
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

# Divisón Train / Test
X_train, X_val, y_train, y_val = train_test_split(
    images_dataset, targets_dataset, test_size=0.20, random_state=42
)

train_dataset = CellSegmentationDataset(X_train, y_train)
val_dataset = CellSegmentationDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


# ==========================================================
# 3. CREACIÓN DE LA ARQUITECTURA MASK R-CNN (TORCHVISION)
# ==========================================================
def get_maskrcnn_model(num_classes=2):
    # Cargar modelo Mask R-CNN preentrenado con backbone ResNet-50-FPN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Reemplazar la cabeza de clasificación (FastRCNNPredictor)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Reemplazar la cabeza de predicción de máscaras (MaskRCNNPredictor)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model


# ==========================================================
# 4. ENTRENAMIENTO DEL MODELO MASK R-CNN
# ==========================================================
print("\n=== 2. Entrenando Mask R-CNN para Detección y Conteo Celular ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_maskrcnn_model(num_classes=2).to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

epochs = 8
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        
    avg_loss = running_loss / len(train_loader)
    print(f"Época [{epoch}/{epochs}] - Loss Total (Box + Mask + RPN): {avg_loss:.4f}")


# ==========================================================
# 5. INFERENCIA Y CONTEO AUTOMÁTICO DE CÉLULAS
# ==========================================================
print("\n=== 3. Inferencia de Mask R-CNN y Conteo de Instancias Celulares ===")
model.eval()

with torch.no_grad():
    val_img, val_target = val_dataset[0]
    prediction = model([val_img.to(device)])[0]

# Filtrar detecciones por umbral de confianza (Score Threshold > 0.60)
SCORE_THRESH = 0.60
pred_boxes = prediction['boxes'].cpu().numpy()
pred_scores = prediction['scores'].cpu().numpy()
pred_masks = prediction['masks'].cpu().numpy()

keep_indices = np.where(pred_scores >= SCORE_THRESH)[0]

final_boxes = pred_boxes[keep_indices]
final_masks = pred_masks[keep_indices]
final_scores = pred_scores[keep_indices]

cell_count = len(keep_indices)
gt_count = len(val_target['boxes'])

print(f"\n--- RESULTADO DEL CONTEO CELULAR ---")
print(f"Número de Células Real (Ground Truth): {gt_count}")
print(f"Número de Células Detectadas (Mask R-CNN): {cell_count}")


# ==========================================================
# 6. VISUALIZACIÓN DE RESULTADOS Y CONTEO AUTOMÁTICO
# ==========================================================
img_np = val_img.permute(1, 2, 0).numpy()
img_np = (img_np * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Imagen Microscópica Original
axes[0].imshow(img_np)
axes[0].set_title("1. Micrografía de Entrada", fontsize=11)
axes[0].axis('off')

# 2. Ground Truth con Bounding Boxes
img_gt = img_np.copy()
for box in val_target['boxes'].numpy():
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 1)

axes[1].imshow(img_gt)
axes[1].set_title(f"2. Ground Truth (Conteo Real: {gt_count})", fontsize=11)
axes[1].axis('off')

# 3. Predicción Mask R-CNN con Segmentación de Instancias y Conteo
img_pred = img_np.copy()
combined_mask = np.zeros_like(img_np)

colors = plt.cm.get_cmap('hsv', max(cell_count + 1, 10))

for idx in range(cell_count):
    box = final_boxes[idx]
    mask = final_masks[idx, 0] > 0.5
    
    x1, y1, x2, y2 = map(int, box)
    color = (np.array(colors(idx)[:3]) * 255).astype(int)
    
    # Dibujar Bounding Box
    cv2.rectangle(img_pred, (x1, y1), (x2, y2), color.tolist(), 1)
    
    # Aplicar color a la máscara de la instancia celular individual
    for c in range(3):
        combined_mask[:, :, c] = np.where(mask, color[c], combined_mask[:, :, c])

# Superposición de máscaras coloreadas sobre la imagen original
img_segmented = cv2.addWeighted(img_pred, 0.6, combined_mask, 0.4, 0)

axes[2].imshow(img_segmented)
axes[2].set_title(f"3. Mask R-CNN (Conteo Predicho: {cell_count} células)", fontsize=11)
axes[2].axis('off')

plt.tight_layout()
plt.show()