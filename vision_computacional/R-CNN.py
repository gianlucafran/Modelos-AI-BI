"""
R-CNN (Region-based Convolutional Neural Networks): Investigación Académica
Aplicación: Detección y localización de lesiones/anomalías sintéticas en radiografías craneales

Conceptos Clave de R-CNN (Girshick et al., 2014):
1. Búsqueda Selectiva (Selective Search): Genera ~2000 propuestas de región (Region Proposals)
   candidatas a ser objetos mediante la agrupación jerárquica de parches por color, textura y tamaño.
2. Extracción de Características (CNN): Cada propuesta de región se recorta, redimensiona
   a un tamaño fijo (warping) y se pasa por una CNN para extraer una representación vectorial.
3. Clasificación con SVMs / Capa Densa: Un clasificador evalúa los vectores para determinar
   la presencia de una clase específica.
4. Dataset Proxy: Adaptamos el dataset "Olivetti Faces" de scikit-learn (escala de grises)
   inyectando nódulos radiológicos sintéticos (lesiones/tumores) con Bounding Boxes
   ground truth para simular un pipeline de investigación académica en visión computacional médica.
"""

import time
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ML & Computer Vision dependencies
from sklearn.datasets import fetch_olivetti_faces
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)

# ==========================================================
# 1. PREPARACIÓN DEL DATASET PROXY (scikit-learn + Lesiones Sintéticas)
# ==========================================================
print("=== 1. Cargando y preparando dataset Olivetti Faces ===")
data = fetch_olivetti_faces(shuffle=True, random_state=42)
raw_images = data.images  # (400, 64, 64) en rango [0, 1]

images_dataset = []
annotations_dataset = []  # Guarda las bounding boxes [x, y, w, h] y etiqueta (1: Lesión, 0: Normal)

for img in raw_images:
    # Convertir a 0-255 uint8 para trabajar con OpenCV
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Decisión aleatoria: ¿Insertamos una lesión en esta imagen? (50% probabilidad)
    has_lesion = np.random.rand() > 0.5
    
    if has_lesion:
        # Generar coordenadas aleatorias dentro de la imagen de 64x64
        w, h = np.random.randint(10, 18), np.random.randint(10, 18)
        x = np.random.randint(5, 64 - w - 5)
        y = np.random.randint(5, 64 - h - 5)
        
        # Simular "lesión radiológica" (nódulo hiperdenso/brillante con desenfoque gaussiano)
        lesion_patch = np.ones((h, w), dtype=np.uint8) * 220
        lesion_patch = cv2.GaussianBlur(lesion_patch, (5, 5), 0)
        
        # Superponer la lesión mediante combinación ponderada
        img_uint8[y:y+h, x:x+w] = cv2.addWeighted(
            img_uint8[y:y+h, x:x+w], 0.3, lesion_patch, 0.7, 0
        )
        
        annotations_dataset.append({'bbox': [x, y, w, h], 'label': 1})
    else:
        annotations_dataset.append({'bbox': None, 'label': 0})
        
    images_dataset.append(img_uint8)

images_dataset = np.array(images_dataset)
print(f"Dataset generado: {len(images_dataset)} imágenes de {images_dataset[0].shape}")


# ==========================================================
# 2. SECTORES CLAVE DE R-CNN: BUSQUEDA SELECTIVA Y IoU
# ==========================================================
def calculate_iou(boxA, boxB):
    """
    Calcula la Intersección sobre Unión (Intersection over Union - IoU) entre dos Bounding Boxes.
    box = [x, y, w, h]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def generate_region_proposals(image, max_proposals=100):
    """
    Implementación de la Búsqueda Selectiva (Selective Search) usando OpenCV.
    """
    # OpenCV Selective Search requiere 3 canales (RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_rgb)
    ss.switchToSelectiveSearchFast()
    
    rects = ss.process()
    return rects[:max_proposals]


# ==========================================================
# 3. EXTRACTION DE CARACTERÍSTICAS (CNN proxy) Y CONSTRUCCIÓN DE PROCESO R-CNN
# ==========================================================
print("\n=== 2. Extrayendo Region Proposals y Muestreando Parches para Entrenar R-CNN ===")

X_patches = []
y_patches = []

# Muestreamos una subpoblación de imágenes para el pipeline demostrativo
SAMPLE_SIZE = 60
TARGET_PATCH_SIZE = (28, 28)

start_time = time.time()

for i in range(SAMPLE_SIZE):
    img = images_dataset[i]
    gt = annotations_dataset[i]
    gt_box = gt['bbox']
    
    proposals = generate_region_proposals(img, max_proposals=60)
    
    for (x, y, w, h) in proposals:
        if w < 6 or h < 6:
            continue
            
        # Extracción de la región de interés (ROI)
        roi = img[y:y+h, x:x+w]
        
        # Warp/Redimensionado estandarizado para la red de características
        roi_resized = cv2.resize(roi, TARGET_PATCH_SIZE)
        
        # Determinación de clase por IoU (Positive label if IoU >= 0.35 con Ground Truth)
        label = 0
        if gt_box is not None:
            iou = calculate_iou([x, y, w, h], gt_box)
            if iou >= 0.35:
                label = 1
                
        X_patches.append(roi_resized.flatten())  # Representación vectorial
        y_patches.append(label)

X_patches = np.array(X_patches) / 255.0  # Normalización [0, 1]
y_patches = np.array(y_patches)

print(f"Propuestas de región procesadas: {len(X_patches)}")
print(f"Parches Positivos (Lesión): {np.sum(y_patches == 1)} | Parches Negativos (Fondo): {np.sum(y_patches == 0)}")
print(f"Tiempo de extracción/segmentación: {time.time() - start_time:.2f} segundos")


# ==========================================================
# ==========================================================
# 4. ENTRENAMIENTO DEL CLASIFICADOR (Regresión Logística / SVM)
# ==========================================================
print("\n=== 3. Entrenando el Clasificador para las Regiones Propuestas ===")

# Dividir dataset de parches en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_patches, y_patches, test_size=0.2, random_state=42, 
    stratify=y_patches if len(np.unique(y_patches)) > 1 else None
)

# Inicializar y entrenar el clasificador proxy
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

# Evaluar el rendimiento del clasificador en los parches
y_pred = classifier.predict(X_test)
print("Reporte de Clasificación del Clasificador de Regiones:")
print(classification_report(y_test, y_pred))


# ==========================================================
# 5. INFERENCIA Y VISUALIZACIÓN R-CNN EN UNA IMAGEN DE PRUEBA
# ==========================================================
print("\n=== 4. Ejecutando Inferencia y Visualización R-CNN ===")

def non_max_suppression(boxes, scores, threshold=0.3):
    """
    Aplica Non-Max Suppression (NMS) para eliminar Bounding Boxes redundantes.
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
        
    return keep

# Seleccionar una imagen de prueba que contenga una lesión sintética
test_idx = -1
for idx in range(SAMPLE_SIZE, len(images_dataset)):
    if annotations_dataset[idx]['label'] == 1:
        test_idx = idx
        break

if test_idx == -1:
    test_idx = SAMPLE_SIZE  # Fallback si ninguna cumple

test_img = images_dataset[test_idx]
gt_info = annotations_dataset[test_idx]

# Generar propuestas de región para la imagen de prueba
test_proposals = generate_region_proposals(test_img, max_proposals=60)
box_predictions = []
box_scores = []

for (x, y, w, h) in test_proposals:
    if w < 6 or h < 6:
        continue
    
    roi = test_img[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, TARGET_PATCH_SIZE)
    feat = roi_resized.flatten().reshape(1, -1) / 255.0
    
    # Obtener la probabilidad de que la región contenga la lesión (clase 1)
    prob = classifier.predict_proba(feat)[0][1]
    
    if prob > 0.55:  # Umbral de confianza ajustable
        box_predictions.append([x, y, w, h])
        box_scores.append(prob)

# Aplicar Non-Max Suppression para filtrar duplicados
keep_indices = non_max_suppression(box_predictions, box_scores, threshold=0.3)
final_boxes = [box_predictions[i] for i in keep_indices]
final_scores = [box_scores[i] for i in keep_indices]

# ==========================================================
# 6. PLOT DE RESULTADOS FINALES
# ==========================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(test_img, cmap='gray')

# Dibujar Ground Truth (Verde)
if gt_info['bbox'] is not None:
    gx, gy, gw, gh = gt_info['bbox']
    rect_gt = plt.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='lime', 
                            facecolor='none', label='Ground Truth (Lesión)')
    ax.add_patch(rect_gt)

# Dibujar Predicciones del modelo R-CNN (Rojo Discontinuo)
for idx, (bx, by, bw, bh) in enumerate(final_boxes):
    score = final_scores[idx]
    rect_pred = plt.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', 
                              facecolor='none', linestyle='--', 
                              label='Predicción R-CNN' if idx == 0 else "")
    ax.add_patch(rect_pred)
    ax.text(bx, max(0, by - 4), f"{score:.2f}", color='red', fontsize=8, 
            fontweight='bold', backgroundcolor='white')

plt.title(f"Detección R-CNN de Lesión (Imagen de Prueba #{test_idx})", fontsize=11, fontweight='bold')
plt.legend(loc='upper right')
plt.axis('off')
plt.tight_layout()
# plt.savefig("rcnn_lesion_detection.png", dpi=300)
print("✓ Inferencia finalizada con éxito. Ventana de visualización lista.")
plt.show()