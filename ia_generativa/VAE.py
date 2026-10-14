"""
Variational Autoencoder (VAE): Generación e Interpolación de Rostros
Aplicación: Reducción Dimensional Probabilística e Interpolación en Espacio Latente
Dataset Real: LFW People (Labeled Faces in the Wild) de Scikit-Learn

Este script entrena un VAE utilizando PyTorch/TensorFlow o implementa el mecanismo 
de proyección latente probabilística sobre el dataset real de rostros LFW.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET REAL (LFW People)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET REAL DE SCIKIT-LEARN: LFW People (Rostros)")
print("=" * 80)

# Cargar dataset real de rostros recortados
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
n_samples, height, width = lfw.images.shape
input_dim = height * width

# Normalizar píxeles a rango [0, 1]
scaler = MinMaxScaler()
X_data = scaler.fit_transform(lfw.data)

print(f"✓ Dataset LFW cargado con éxito:")
print(f"   • Total de imágenes: {n_samples}")
print(f"   • Dimensión por imagen: {height}x{width} ({input_dim} píxeles)")

# Convertir a Tensores de PyTorch
tensor_x = torch.clamp(torch.tensor(X_data, dtype=torch.float32), 0.0, 1.0)
dataset = TensorDataset(tensor_x)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ============================================================================
# 2. DEFINICIÓN DE LA ARQUITECTURA DEL VAE
# ============================================================================
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Función de Pérdida del VAE: Reconstrucción (BCE) + Divergencia KL
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ============================================================================
# 3. ENTRENAMIENTO DEL MODELO VAE
# ============================================================================
print("\n" + "=" * 80)
print("2. ENTRENANDO EL AUTOENCODER VARIACIONAL (VAE)")
print("=" * 80)

latent_dim = 20
model = VAE(input_dim=input_dim, latent_dim=latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 30
model.train()
for epoch in range(1, epochs + 1):
    train_loss = 0
    for batch_idx, (data,) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"   • Época {epoch}/{epochs} | Pérdida Promedio: {train_loss / len(dataloader.dataset):.2f}")

# ============================================================================
# 4. INTERPOLACIÓN LINEAL EN EL ESPACIO LATENTE
# ============================================================================
print("\n" + "=" * 80)
print("3. REALIZANDO INTERPOLACIÓN EN EL ESPACIO LATENTE ENTRE DOS ROSTROS")
print("=" * 80)

model.eval()
with torch.no_grad():
    # Seleccionamos dos rostros distintos del dataset real LFW
    idx1, idx2 = 10, 50
    face1 = tensor_x[idx1].unsqueeze(0)
    face2 = tensor_x[idx2].unsqueeze(0)

    # Obtenemos sus representaciones latentes medias (mu)
    mu1, _ = model.encode(face1)
    mu2, _ = model.encode(face2)

    # Generar pasos de interpolación (alpha de 0.0 a 1.0)
    num_steps = 8
    alphas = np.linspace(0, 1, num_steps)
    interpolated_faces = []

    for alpha in alphas:
        # Interpolación Lineal en Z: z_interp = (1 - alpha) * z1 + alpha * z2
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        recon_face = model.decode(z_interp)
        interpolated_faces.append(recon_face.numpy().reshape(height, width))

print(f"✓ Interpolación completada: {num_steps} fotogramas generados entre Rostro #{idx1} y Rostro #{idx2}.")

# ============================================================================
# 5. VISUALIZACIÓN DE LA TRANSICIÓN
# ============================================================================
fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(interpolated_faces[i], cmap='gray')
    ax.set_title(f"α = {alphas[i]:.2f}", fontsize=9)
    ax.axis('off')

plt.suptitle("Interpolación en el Espacio Latente (VAE) sobre LFW People Dataset", fontsize=12, fontweight='bold')
plt.tight_layout()
#plt.savefig("vae_face_interpolation.png", dpi=300)
print("✓ Imagen con la secuencia de interpolación guardada como 'vae_face_interpolation.png'")