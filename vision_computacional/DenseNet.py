"""
DenseNet (Densely Connected Convolutional Networks): Diagnóstico de Rayos X
Demostración con dataset proxy (scikit-learn) adaptado a imágenes médicas

Conceptos:
- DenseNet (2017): Resuelve el problema del desvanecimiento del gradiente de una 
  forma radical. En lugar de usar conexiones residuales (sumas) como ResNet, 
  DenseNet CONCATENA los mapas de características.
- Conectividad Densa: Cada capa recibe como entrada los mapas de características de 
  TODAS las capas anteriores dentro de su bloque, y pasa sus propios mapas a TODAS 
  las capas siguientes.
- Tasa de Crecimiento (Growth Rate - k): Al concatenar, el número de canales crece. 
  La tasa de crecimiento (típicamente k=32) dicta cuántos canales nuevos aporta cada capa.
- Reutilización de Características (Feature Reuse): Ideal para imágenes médicas (Rayos X). 
  Las primeras capas detectan bordes sutiles (fisuras óseas, microcalcificaciones) 
  y esta información cruda se preserva y transmite directamente hasta la capa de 
  clasificación final sin alterarse.
- Dataset Simulado: Transformaremos el dataset "Olivetti Faces" de scikit-learn 
  (escala de grises) aplicándole filtros e inyectando anomalías sintéticas (tumores/nódulos) 
  para crear un dataset de "Radiografías Craneales" totalmente ejecutable.
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dependencias de Machine Learning y Deep Learning
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms_v2
from torchvision.models import densenet121, DenseNet121_Weights

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. CREACIÓN DEL DATASET (SIMULACIÓN DE RAYOS X MÉDICOS)
# ============================================================================
print("=" * 80)
print("PASO 1: GENERACIÓN DEL DATASET MÉDICO (RADIOGRAFÍAS SINTÉTICAS)")
print("=" * 80)

# Descargamos el dataset (imágenes en escala de grises de 64x64)
print("Cargando dataset base de scikit-learn (Olivetti Faces)...")
dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
imagenes_base = dataset.images

# Transformamos las imágenes para simular radiografías craneales (inversión de intensidades)
radiografias = 1.0 - imagenes_base

# Inyectamos patologías (Anomalías/Tumores sintéticos) en el 50% del dataset
np.random.seed(42)
n_muestras = len(radiografias)
etiquetas = np.zeros(n_muestras, dtype=np.int64)
indices_enfermos = np.random.choice(n_muestras, n_muestras // 2, replace=False)

etiquetas[indices_enfermos] = 1  # 1 = Patología (Tumor), 0 = Sano

print("Inyectando anomalías radiológicas (nódulos sintéticos)...")
for idx in indices_enfermos:
    img = radiografias[idx].copy()
    cy, cx = np.random.randint(15, 49, size=2)
    radio = np.random.randint(3, 7)
    
    # Brillo gaussiano/sférico para simular masa tumoral radiopaca
    y, x = np.ogrid[-cy:64-cy, -cx:64-cx]
    mascara = (x**2 + y**2) <= radio**2
    img[mascara] += 0.4
    radiografias[idx] = np.clip(img, 0.0, 1.0)

print(f"✓ Dataset Médico generado: {n_muestras} 'Radiografías' (64x64 px).")
print(f"✓ Clases: {np.sum(etiquetas==0)} Sanos | {np.sum(etiquetas==1)} Con Patología")

# Dividir en Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(
    radiografias, etiquetas, test_size=0.2, random_state=42, stratify=etiquetas
)

# Dataset personalizado en PyTorch
class XRayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Triplicar canales para emular entrada RGB requerida por DenseNet preentrenado
        img_1c = self.images[idx]
        img_3c = np.stack([img_1c] * 3, axis=-1).astype(np.float32)
        
        # Convertir a tensor (C, H, W)
        tensor_img = torch.from_numpy(img_3c).permute(2, 0, 1)
        
        if self.transform:
            tensor_img = self.transform(tensor_img)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_img, label

# Transformaciones con la API moderna torchvision.transforms.v2
transformacion_xray = transforms_v2.Compose([
    transforms_v2.ToDtype(torch.float32, scale=False),
    transforms_v2.Resize((224, 224), antialias=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = XRayDataset(X_train, y_train, transformacion_xray)
test_dataset = XRayDataset(X_test, y_test, transformacion_xray)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ============================================================================
# 2. DEFINICIÓN DEL MODELO DENSENET-121
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: TRANSFER LEARNING CON DENSENET-121")
print("=" * 80)

print("Cargando arquitectura DenseNet-121 con pesos IMAGENET1K_V1...")
modelo_densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

# Congelar parámetros base para extracción de características
for param in modelo_densenet.parameters():
    param.requires_grad = False

# Adaptar el clasificador final para 2 clases (Sano vs Patología)
num_features = modelo_densenet.classifier.in_features
modelo_densenet.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelo_densenet = modelo_densenet.to(device)

print(f"✓ Arquitectura DenseNet configurada en dispositivo: {device}.")
print(f"✓ Clasificador redefinido para 2 clases operativas.")

# ============================================================================
# 3. ENTRENAMIENTO REAL DE LA RED
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAMIENTO MÉDICO (FINE-TUNING REGLAMENTARIO)")
print("=" * 80)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo_densenet.classifier.parameters(), lr=1e-3)

epocas = 8
historial_loss = []
historial_acc = []

print("Iniciando ciclo de entrenamiento real...")
inicio = time.time()

for e in range(epocas):
    modelo_densenet.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = modelo_densenet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    historial_loss.append(epoch_loss)
    
    # Evaluación en Test
    modelo_densenet.eval()
    correctos = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = modelo_densenet(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correctos += (preds == labels).sum().item()
            
    acc = (correctos / total) * 100
    historial_acc.append(acc)
    
    print(f"  [Época {e+1:02d}/{epocas:02d}] Pérdida (Loss): {epoch_loss:.4f} | Precisión Validación: {acc:.2f}%")

duracion = time.time() - inicio
print(f"✓ Entrenamiento finalizado en {duracion:.2f} segundos.")

# ============================================================================
# 4. VISUALIZACIÓN: ARQUITECTURA DENSA Y DIAGNÓSTICOS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: VISUALIZACIÓN DE LA TEORÍA Y RESULTADOS CLÍNICOS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfica 1: Curva de Aprendizaje Real
axes[0, 0].plot(range(1, epocas+1), historial_acc, marker='o', color='teal', linewidth=3)
axes[0, 0].set_title('Convergencia del Diagnóstico (Precisión)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Época de Entrenamiento')
axes[0, 0].set_ylabel('Precisión (%)')
axes[0, 0].grid(True, alpha=0.3)

# Gráfica 2: El Concepto de Dense Block (Esquema Teórico)
ax_diag = axes[0, 1]
ax_diag.axis('off')
ax_diag.set_title('Arquitectura de un Bloque Denso (Dense Block)', fontsize=13, fontweight='bold')

ax_diag.add_patch(plt.Circle((0.2, 0.8), 0.05, color='gray', alpha=0.5))
ax_diag.text(0.2, 0.8, 'Input\n(x0)', ha='center', va='center', fontweight='bold')

ax_diag.add_patch(plt.Circle((0.5, 0.8), 0.05, color='lightblue', alpha=0.9))
ax_diag.text(0.5, 0.8, 'Conv 1\n(x1)', ha='center', va='center', fontweight='bold')

ax_diag.add_patch(plt.Circle((0.8, 0.8), 0.05, color='salmon', alpha=0.9))
ax_diag.text(0.8, 0.8, 'Conv 2\n(x2)', ha='center', va='center', fontweight='bold')

ax_diag.add_patch(plt.Circle((0.5, 0.4), 0.05, color='lightgreen', alpha=0.9))
ax_diag.text(0.5, 0.4, 'Concat\n[x0,x1,x2]', ha='center', va='center', fontweight='bold', fontsize=9)

prop_flecha = dict(arrowstyle="->", color="black", lw=1.5)
ax_diag.annotate("", xy=(0.45, 0.8), xytext=(0.25, 0.8), arrowprops=prop_flecha)
ax_diag.annotate("", xy=(0.45, 0.45), xytext=(0.22, 0.76), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
ax_diag.annotate("", xy=(0.75, 0.8), xytext=(0.55, 0.8), arrowprops=prop_flecha)
ax_diag.annotate("", xy=(0.5, 0.45), xytext=(0.5, 0.75), arrowprops=dict(arrowstyle="->", color="lightblue", lw=1.5, ls="--"))
ax_diag.annotate("", xy=(0.55, 0.45), xytext=(0.78, 0.76), arrowprops=dict(arrowstyle="->", color="salmon", lw=1.5, ls="--"))

ax_diag.text(0.5, 0.2, "En DenseNet, cada capa recibe los Feature Maps de\nTODAS las capas anteriores mediante CONCATENACIÓN,\npreservando los detalles finos de la radiografía.", 
             ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Gráficas 3 y 4: Inferencia en Muestras del Dataset
modelo_densenet.eval()
idx_sano = np.where(y_test == 0)[0][0]
idx_patologia = np.where(y_test == 1)[0][0]

def predecir_muestra(idx):
    img_tensor, _ = test_dataset[idx]
    img_input = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = modelo_densenet(img_input)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

prob_sano = predecir_muestra(idx_sano)
prob_patologia = predecir_muestra(idx_patologia)

axes[1, 0].imshow(X_test[idx_sano], cmap='bone')
axes[1, 0].set_title(f'Paciente A: Radiografía SANA\nPredicción DenseNet: Sano ({prob_sano[0]*100:.1f}%)', 
                     fontsize=12, fontweight='bold', color='forestgreen')
axes[1, 0].axis('off')

axes[1, 1].imshow(X_test[idx_patologia], cmap='bone')
circ = plt.Circle((32, 32), 18, color='red', fill=False, linewidth=2, linestyle='--')
axes[1, 1].add_patch(circ)
axes[1, 1].set_title(f'Paciente B: ANOMALÍA DETECTADA\nPredicción DenseNet: Tumor ({prob_patologia[1]*100:.1f}%)', 
                     fontsize=12, fontweight='bold', color='crimson')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('01_densenet_rayosx.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_densenet_rayosx.png")
plt.show()

# ============================================================================
# 5. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN TEÓRICO: DENSENET EN IMÁGENES MÉDICAS")
print("=" * 80)

print("""
1. CONCATENACIÓN VS SUMA:
   • ResNet utiliza sumas elementales $F(x) + x$, lo que degrada levemente la 
     información original de las primeras capas.
   • DenseNet realiza concatenación directa en la dimensión de canales. Esto permite 
     que patrones de baja escala (bordes de huesos, microcalcificaciones) lleguen 
     intactos a las capas de decisión clínica.

2. EFICIENCIA DE PARÁMETROS:
   • Gracias al "Feature Reuse", cada capa genera un número reducido de nuevos 
     mapas de características ($k \approx 32$), haciendo a DenseNet-121 mucho más ligera 
     y rápida de entrenar que arquitecturas tradicionales de profundidad similar.
""")