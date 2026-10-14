import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

#!pip install torchvision
import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

!pip install torchmetrics
import torchmetrics

# ============================================================================
# TEORÍA DE DATOS: Mini-batch Gradient Descent
# Usamos lotes (batches) de 60 imágenes en lugar de todo el dataset a la vez.
# Esto acelera el cómputo, reduce el uso de memoria y ayuda al modelo a escapar
# de mínimos locales gracias al ruido estocástico introducido en cada lote.
# ============================================================================
batch_size = 60

train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
labels
# show images
imshow(torchvision.utils.make_grid(images))

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Building blocks of convolutional neural network.
        
        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from 0 to 9).
        """
        super(CNN, self).__init__()

        # ============================================================================
        # TEORÍA DE CONVOLUCIÓN (Feature Extraction)
        # La capa convolucional aplica filtros (kernels) que se deslizan sobre la imagen.
        # Operación matemática: Producto punto entre el filtro y un parche de la imagen.
        # Esto extrae características locales (bordes, curvas, texturas).
        # padding=1 asegura que la imagen resultante mantenga sus dimensiones espaciales.
        # ============================================================================
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        
        # ============================================================================
        # TEORÍA DE POOLING (Downsampling)
        # Max Pooling reduce la dimensionalidad espacial de la imagen a la mitad (stride=2).
        # Retiene solo el valor máximo de cada región 2x2.
        # Beneficios: Reduce parámetros, ahorra cómputo y aporta "invarianza a la traslación"
        # (reconoce el patrón sin importar si está un poco movido).
        # ============================================================================
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd convolutional layer (Extrae características más complejas a partir de las básicas)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        
        # ============================================================================
        # TEORÍA DE RED DENSA (Fully Connected Layer)
        # Actúa como el clasificador final. Toma los mapas de características abstractos
        # y realiza una combinación lineal para asignar una puntuación (score) a cada clase.
        # ============================================================================
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.
        """
        # ============================================================================
        # TEORÍA DE ACTIVACIÓN: ReLU (Rectified Linear Unit)
        # Función: f(x) = max(0, x). Convierte los valores negativos a 0.
        # Introduce "no linealidad" en la red, permitiendo que aprenda fronteras de
        # decisión complejas. Además, mitiga el problema del desvanecimiento del gradiente.
        # ============================================================================
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        
        # ============================================================================
        # TEORÍA FLATTEN (Aplanado)
        # Pasa de un tensor tridimensional (Canales x Alto x Ancho) a un vector 1D
        # necesario para alimentar la capa completamente conectada (Linear).
        # ============================================================================
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=1, num_classes=10).to(device)
print(model)

# ============================================================================
# TEORÍA DE FUNCIÓN DE PÉRDIDA (Cross Entropy Loss)
# Mide la divergencia entre la distribución de probabilidades predicha por el modelo
# y la distribución real (etiquetas verdaderas).
# Matemáticamente aplica un Softmax a las salidas para obtener probabilidades,
# seguido del Logaritmo de Verosimilitud Negativa (Negative Log-Likelihood).
# ============================================================================
criterion = nn.CrossEntropyLoss()

# ============================================================================
# TEORÍA DEL OPTIMIZADOR (Adam)
# Adam (Adaptive Moment Estimation) actualiza los pesos de la red.
# Combina los beneficios del momentum (dirección del gradiente) y 
# RMSProp (tasa de aprendizaje adaptativa para cada parámetro).
# ============================================================================
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=10
for epoch in range(num_epochs):
    # Iterate over training batches
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        
        # 1. Forward Pass (Predicción)
        scores = model(data)
        
        # 2. Cálculo de la pérdida (Error)
        loss = criterion(scores, targets)
        
        # ============================================================================
        # TEORÍA DE BACKPROPAGATION (Propagación hacia atrás)
        # ============================================================================
        # 3. zero_grad: PyTorch acumula gradientes por defecto, debemos reiniciarlos a 0.
        optimizer.zero_grad()
        
        # 4. backward: Calcula la derivada de la pérdida con respecto a cada peso
        # en la red aplicando la "Regla de la Cadena" del cálculo diferencial.
        loss.backward()
        
        # 5. step: Ajusta los pesos moviéndolos en la dirección opuesta al gradiente
        # para intentar minimizar la pérdida (Descenso del Gradiente).
        optimizer.step()

# Set up of multiclass accuracy, precision, and recall metrics
acc = torchmetrics.Accuracy(task="multiclass",num_classes=10).to(device)
precision_metric = torchmetrics.Precision(task="multiclass",num_classes=10).to(device)
recall_metric = torchmetrics.Recall(task="multiclass",num_classes=10).to(device)

# Iterate over the dataset batches
model.eval() # Modo evaluación (desactiva comportamientos de entrenamiento como Dropout)

# ============================================================================
# TEORÍA DE INFERENCIA (torch.no_grad)
# Al evaluar el modelo, no necesitamos actualizar los pesos.
# torch.no_grad() apaga el motor de cálculo de derivadas (Autograd),
# lo cual ahorra significativamente memoria y aceleración de hardware.
# ============================================================================
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Get predicted probabilities for test data batch
        outputs = model(images)
        
        # Extraemos la clase con la puntuación más alta (argmax lógico a lo largo del eje 1)
        _, preds = torch.max(outputs, 1)
        
        acc(preds, labels)
        precision_metric(preds, labels)
        recall_metric(preds, labels)

# Compute total test accuracy, precision, and recall
test_accuracy = acc.compute()
test_precision = precision_metric.compute()
test_recall = recall_metric.compute()

print(f"Test accuracy: {test_accuracy}")
print(f"Test precision: {test_precision}")
print(f"Test recall: {test_recall}")