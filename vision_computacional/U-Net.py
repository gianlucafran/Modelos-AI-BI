"""
U-Net (Ronneberger et al., 2015): Segmentación Semántica Médica
Aplicación: Segmentación píxel a píxel de tumores/lesiones en imágenes de resonancia magnética (RM) craneal

Conceptos Clave de U-Net:
1. Arquitectura en Forma de 'U':
   - Contracción / Encoder: Extrae mapas de características de alto nivel reduciendo
     la resolución espacial mediante convoluciones y max pooling.
   - Expansión / Decoder: Recupera la resolución espacial original aplicando convoluciones
     transpuestas (Up-sampling / Deconvolution).
2. Conexiones de Salto (Skip Connections):
   Concatena directamente las capas de alta resolución del Encoder con las capas del Decoder.
   Esto conserva detalles espaciales finos y bordes de precisión (críticos para delimitar 
   el contorno exacto de masas tumorales).
3. Métrica de Evaluación y Función de Pérdida (Dice Coefficient / IoU Loss):
   La pérdida de Entropía Cruzada tradicional falla ante desbalances de clase severos 
   (donde el tumor ocupa < 5% de la imagen). Se utiliza el coeficiente Sørensen-Dice 
   para optimizar la superposición exacta de las máscaras.
4. Dataset Proxy: Adaptación del dataset 'Olivetti Faces' de scikit-learn con inyección de 
   tumores/nódulos y sus correspondientes máscaras binarias (Ground Truth Mask).
"""

import time
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ML & Computer Vision dependencies
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# PyTorch framework para la implementación de U-Net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ==========================================================
# 1. PREPARACIÓN DEL DATASET PROXY (scikit-learn + Máscaras de Tumores)
# ==========================================================
print("=== 1. Cargando Olivetti Faces y Generando Máscaras de Segmentación ===")
data = fetch_olivetti_faces(shuffle=True, random_state=42)
raw_images = data.images  # (400, 64, 64) en [0, 1]

images_dataset = []
masks_dataset = []

for img in raw_images:
    # Simular resonancia magnética ajustando contraste y brillo
    mri_img = (img * 255).astype(np.uint8)
    mri_img = cv2.equalizeHist(mri_img)
    
    mask = np.zeros((64, 64), dtype=np.uint8)
    
    # 60% de probabilidad de tener tumor
    if np.random.rand() > 0.4:
        # Forma elíptica del tumor sintético
        center_x = np.random.randint(15, 49)
        center_y = np.random.randint(15, 49)
        axes_x = np.random.randint(4, 10)
        axes_y = np.random.randint(4, 10)
        angle = np.random.randint(0, 180)
        
        # Dibujar máscara ground truth (1 en píxeles de tumor, 0 en fondo)
        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 1, -1)
        
        # Inyectar lesión hiperintensa en la imagen de resonancia
        tumor_region = np.ones((64, 64), dtype=np.uint8) * 230
        tumor_region = cv2.GaussianBlur(tumor_region, (5, 5), 0)
        
        mri_img = np.where(mask == 1, cv2.addWeighted(mri_img, 0.2, tumor_region, 0.8, 0), mri_img)
        
    images_dataset.append(mri_img)
    masks_dataset.append(mask)

images_dataset = np.array(images_dataset) / 255.0  # Normalizado [0, 1]
masks_dataset = np.array(masks_dataset)            # Binario {0, 1}

print(f"Dataset generado: {len(images_dataset)} imágenes de {images_dataset[0].shape}")


# ==========================================================
# 2. CONSTRUCCIÓN DEL DATASET Y DATALOADER EN PYTORCH
# ==========================================================
class MedicalSegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) # (N, 1, H, W)
        self.masks = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)   # (N, 1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

X_train, X_val, y_train, y_val = train_test_split(
    images_dataset, masks_dataset, test_size=0.20, random_state=42
)

train_dataset = MedicalSegmentationDataset(X_train, y_train)
val_dataset = MedicalSegmentationDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# ==========================================================
# 3. ARQUITECTURA U-NET COMPLETA
# ==========================================================
class DoubleConv(nn.Module):
    """Bloque de Convolución Doble: (Conv2d -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (Ruta de Contracción)
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck (Cuello de Botella)
        self.bottleneck = DoubleConv(64, 128)
        
        # Decoder (Ruta de Expansión) + Skip Connections
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)  # 64 (up) + 64 (skip enc2) = 128
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)   # 32 (up) + 32 (skip enc1) = 64
        
        # Clasificación por píxel final (Sigmoide)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder con Concatenación de Skip Connections
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1) # Concatenar características
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1) # Concatenar características
        d1 = self.dec1(d1)
        
        out = torch.sigmoid(self.final_conv(d1))
        return out


# ==========================================================
# 4. MÉTRICA DICE Y FUNCIÓN DE PÉRDIDA (DICE LOSS)
# ==========================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - torch.mean(dice)


# ==========================================================
# 5. ENTRENAMIENTO DE LA RED U-NET
# ==========================================================
print("\n=== 2. Entrenando Modelo U-Net ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1).to(device)

criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 15
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    if epoch % 3 == 0 or epoch == 1:
        print(f"Época [{epoch}/{epochs}] - Loss (Dice Loss): {epoch_loss:.4f}")


# ==========================================================
# 6. INFERENCIA Y EVALUACIÓN DE RESULTADOS
# ==========================================================
print("\n=== 3. Evaluando Segmentación U-Net en Dataset de Validación ===")
model.eval()

with torch.no_grad():
    val_imgs, val_masks = next(iter(val_loader))
    val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)
    preds = model(val_imgs)

# Convertir a NumPy para análisis visual
preds_np = preds.cpu().numpy().squeeze(1)
imgs_np = val_imgs.cpu().numpy().squeeze(1)
masks_np = val_masks.cpu().numpy().squeeze(1)

# Umbralizar predicción a máscara binaria
binary_preds = (preds_np > 0.5).astype(np.uint8)


# Visualización de 3 ejemplos
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for i in range(3):
    # Imagen RM original
    axes[i, 0].imshow(imgs_np[i], cmap='gray')
    axes[i, 0].set_title(f"Ejemplo {i+1}: RM Craneal")
    axes[i, 0].axis('off')
    
    # Máscara Ground Truth (Tumor Real)
    axes[i, 1].imshow(masks_np[i], cmap='magma')
    axes[i, 1].set_title("Máscara Ground Truth (Real)")
    axes[i, 1].axis('off')
    
    # Predicción U-Net
    axes[i, 2].imshow(binary_preds[i], cmap='magma')
    axes[i, 2].set_title("Segmentación Predicha (U-Net)")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()