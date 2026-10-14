"""
Vision Transformer (ViT) para Clasificación a Gran Escala
Aplicación: División de imágenes en parches, proyección lineal, atención multinúcleo y clasificación.
Dataset Real: LFW People (Labeled Faces in the Wild) de Scikit-Learn.

Este script implementa un módulo PatchEmbedding, Transformer Encoder de ViT y el 
cabezal de clasificación entrenado sobre el dataset real de rostros LFW.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set de semillas para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET REAL (Scikit-Learn LFW People)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET REAL DE SCIKIT-LEARN: LFW People (Clasificación)")
print("=" * 80)

# Cargar dataset de rostros etiquetados con un mínimo de imágenes por persona
lfw_people = fetch_lfw_people(min_faces_per_person=25, resize=0.5)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.images
y = lfw_people.target
target_names = lfw_people.target_names
num_classes = len(target_names)

print(f"✓ Dataset cargado exitosamente:")
print(f"   • Total de imágenes: {n_samples}")
print(f"   • Dimensiones por imagen: {h}x{w} píxeles")
print(f"   • Número de clases (personas): {num_classes}")

# Reescalar imágenes al tamaño compatible con parches (e.g., 60x40 píxeles -> 3 canales si fuera RGB)
# Agregamos dimensión de canal (1, H, W) y normalizamos entre 0 y 1
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1) / 255.0
y_tensor = torch.tensor(y, dtype=torch.long)

# División en Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# ============================================================================
# 2. IMPLEMENTACIÓN DE LA ARQUITECTURA VISION TRANSFORMER (ViT)
# ============================================================================
print("\n" + "=" * 80)
print("2. CONSTRUYENDO EL MODELO VISION TRANSFORMER (ViT)")
print("=" * 80)

class PatchEmbedding(nn.Module):
    """Convierte la imagen de entrada en parches y aplica una proyección lineal."""
    def __init__(self, in_channels=1, patch_size=10, embed_dim=128, img_size=(62, 47)):
        super().__init__()
        self.patch_size = patch_size
        # Usamos Conv2d con kernel y stride igual al tamaño del parche para proyectar linealmente
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Cálculo dinámico del número de parches
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    def forward(self, x):
        # x shape: [B, C, H, W] -> proj: [B, Embed_Dim, H_patches, W_patches]
        x = self.proj(x)
        # Flatten espacial: [B, Embed_Dim, Num_Patches] -> Permute: [B, Num_Patches, Embed_Dim]
        x = x.flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    """Ensamblado del Vision Transformer: Patch Embedding + Position Embedding + Transformer Encoder + MLP Head"""
    def __init__(self, img_size=(62, 47), patch_size=10, in_channels=1, num_classes=num_classes, 
                 embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.num_patches
        
        # Token especial [CLS] para la clasificación global
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional Embeddings aprendibles (num_patches + 1 para el [CLS] token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Bloques de Transformer Encoder estándar
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Cabezal de Clasificación MLP
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Inicialización de pesos
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        # Extracción de parches proyectados: [B, Num_Patches, Embed_Dim]
        x = self.patch_embed(x)
        
        # Expandir token [CLS] para cada elemento del batch
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenar [CLS] al inicio de la secuencia de parches
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Sumar información posicional
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        
        # Pasar por el Transformer Encoder
        x = self.blocks(x)
        x = self.norm(x)
        
        # Extraer únicamente la representación del token [CLS]
        cls_rep = x[:, 0]
        
        # Proyectar a las clases finales
        logits = self.head(cls_rep)
        return logits

# Instanciar el modelo con parámetros ligeros adaptados al tamaño de imagen
model = VisionTransformer(img_size=(h, w), patch_size=8, in_channels=1, num_classes=num_classes)
print(f"✓ Modelo ViT instanciado correctamente.")
print(f"   • Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN DEL VISION TRANSFORMER
# ============================================================================
print("\n" + "=" * 80)
print("3. ENTRENAMIENTO Y EVALUACIÓN DE CLASIFICACIÓN DE ROSTROS CON ViT")
print("=" * 80)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

epochs = 15
model.train()
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    acc = correct / total
    if epoch % 5 == 0 or epoch == 1:
        print(f"   • Época {epoch}/{epochs} | Pérdida: {total_loss/total:.4f} | Precisión Train: {acc*100:.2f}%")

# Evaluación en conjunto de prueba
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
print(f"\n✓ Precisión Final en el Conjunto de Test (Scikit-Learn LFW): {test_acc*100:.2f}%")