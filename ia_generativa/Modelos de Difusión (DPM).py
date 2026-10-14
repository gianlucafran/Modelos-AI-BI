import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Configurar dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ==========================================
# 1. Carga y Preparación del Dataset de Scikit-Learn
# ==========================================
print("Cargando dataset 'digits' de scikit-learn...")
digits = load_digits()
X = digits.images  # Matriz de 1797 imágenes de 8x8 píxeles

# Escalar imágenes al rango [-1, 1] habitual en modelos DPM
X_normalized = (X / 8.0) - 1.0  # Los valores originales van de 0 a 16
X_tensor = torch.tensor(X_normalized, dtype=torch.float32).unsqueeze(1)  # Forma: (N, 1, 8, 8)

# Crear DataLoader
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# ==========================================
# 2. Definición del Schedule de Ruido DDPM
# ==========================================
class DiffusionSchedule:
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def add_noise(self, x_0, t):
        """Agrega ruido gaussiano a la imagen x_0 en el paso t (Forward Process)"""
        noise = torch.randn_like(x_0).to(device)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Fórmula: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        noisy_image = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise
        return noisy_image, noise

schedule = DiffusionSchedule(timesteps=300)


# ==========================================
# 3. Red Neuronal para Estimación de Ruido (UNet / MLP ResNet)
# ==========================================
class SimpleDenoisingModel(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super(SimpleDenoisingModel, self).__init__()
        # Embedding temporal para el paso t
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Bloques convolucionales para procesar las imágenes
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # Normalizar t al rango [0, 1] y obtener embedding temporal
        t_normalized = (t.float() / 300.0).unsqueeze(-1)
        t_emb = self.time_mlp(t_normalized).unsqueeze(-1).unsqueeze(-1)
        
        h = self.relu(self.conv1(x)) + t_emb
        h = self.relu(self.conv2(h))
        output = self.conv3(h)
        return output

model = SimpleDenoisingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


# ==========================================
# 4. Entrenar el Modelo de Difusión (DPM)
# ==========================================
epochs = 20
print(f"Entrenando el Modelo de Difusión durante {epochs} épocas...")

model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for batch in dataloader:
        x_0 = batch[0].to(device)
        batch_size = x_0.shape[0]

        # Muestrear t aleatorio para cada imagen del batch
        t = torch.randint(0, schedule.timesteps, (batch_size,), device=device).long()

        # Difusión directa: agregar ruido a x_0
        x_t, noise = schedule.add_noise(x_0, t)

        # Predicción del ruido mediante la red neuronal
        predicted_noise = model(x_t, t)
        loss = criterion(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época {epoch+1}/{epochs} | Loss Promedio: {total_loss/len(dataloader):.4f}")


# ==========================================
# 5. Muestreo / Generación Sintética de Imágenes (Reverse Process)
# ==========================================
print("\nSintetizando nuevas imágenes a partir de ruido gaussiano...")

@torch.no_grad()
def sample_images(model, schedule, num_samples=8):
    model.eval()
    # Iniciar desde ruido puro N(0, I)
    x = torch.randn((num_samples, 1, 8, 8)).to(device)

    for t_step in reversed(range(schedule.timesteps)):
        t = torch.tensor([t_step] * num_samples, device=device).long()
        predicted_noise = model(x, t)

        beta_t = schedule.betas[t_step]
        alpha_t = schedule.alphas[t_step]
        alpha_cumprod_t = schedule.alphas_cumprod[t_step]

        # Calcular media según la fórmula DDPM
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)

        if t_step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

    # Reescalar de vuelta al rango de píxeles [0, 1]
    x = (x + 1.0) / 2.0
    return x.clamp(0, 1).cpu().numpy()

generated_samples = sample_images(model, schedule, num_samples=8)

# Visualizar resultados
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(generated_samples[i].squeeze(), cmap='gray')
    ax.axis('off')
    ax.set_title(f"Sintético {i+1}")

plt.suptitle("Imágenes de dígitos generadas sintéticamente mediante DPM")
plt.tight_layout()
plt.show()