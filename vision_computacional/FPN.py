import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork

# ==========================================
# 1. Definición del Modelo: Backbone + FPN
# ==========================================
class ResNetWithFPN(nn.Module):
    def __init__(self, out_channels=256):
        super(ResNetWithFPN, self).__init__()
        
        # Cargamos el backbone (ResNet18) con los pesos preentrenados
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Extraemos los mapas de características de diferentes etapas (capas C2, C3, C4, C5)
        return_nodes = {
            'layer1': 'feat0',
            'layer2': 'feat1',
            'layer3': 'feat2',
            'layer4': 'feat3'
        }
        self.body = create_feature_extractor(backbone, return_nodes=return_nodes)
        
        # Canales de entrada para el FPN según las capas seleccionadas de ResNet18
        in_channels_list = [64, 128, 256, 512]
        
        # FPN que unifica todos los niveles a 'out_channels' (p. ej. 256)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )

    def forward(self, x):
        # 1. Extraer características del backbone (ruta ascendente)
        features = self.body(x)
        
        # 2. Aplicar FPN (ruta descendente y conexiones laterales)
        fpn_features = self.fpn(features)
        
        return fpn_features

# ==========================================
# 2. Carga del Dataset Real (STL-10)
# ==========================================
print("Cargando dataset real (STL-10)...")
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset real de imágenes reales
dataset = STL10(root="./data", split="test", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ==========================================
# 3. Ejecución e Inspección de Salidas
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetWithFPN(out_channels=256).to(device)
model.eval()

# Tomar un batch del dataset real
images, labels = next(iter(dataloader))
images = images.to(device)

print(f"\n[Entrada] Tamaño del batch de imágenes: {images.shape}")

# Forward pass
with torch.no_grad():
    multiscale_outputs = model(images)

print("\n[Salida FPN] Mapas de características multiescala generados:")
for key, feature_map in multiscale_outputs.items():
    print(f" Nivel '{key}': Forma tensor = {list(feature_map.shape)} "
          f"(Canales homogenizados a 256, Resolución adaptada)")