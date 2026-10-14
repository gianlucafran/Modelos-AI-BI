"""
Fast R-CNN: Detección y conteo visual de objetos.

IMPORTANTE:
Este script implementa Fast R-CNN como arquitectura de detección usando
una red convolucional compartida, propuestas de regiones externas (Selective
Search), RoI Pooling, clasificación y regresión de cajas delimitadoras.

A diferencia de Faster R-CNN, Fast R-CNN NO incorpora una Region Proposal
Network (RPN). Por ello, este ejemplo genera propuestas con Selective Search
(OpenCV contrib), y después procesa esas regiones con Fast R-CNN.

Dependencias:
    pip install numpy matplotlib pillow torch torchvision opencv-contrib-python

Notas:
- Los pesos preentrenados de torchvision disponibles de forma estándar para
  detección corresponden a Faster R-CNN, no a Fast R-CNN clásico.
- Este script construye Fast R-CNN con un backbone ResNet-50 preentrenado en
  ImageNet, pero el detector (clasificador COCO + regresor de cajas) necesita
  entrenamiento con un dataset de detección para producir detecciones útiles.
- Por tanto, no se presenta como un detector COCO preentrenado listo para usar.
  El script es una demostración de la arquitectura/propuestas y muestra el
  flujo de regiones; para obtener detecciones reales hay que entrenar el modelo.
"""

from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_pool
from torchvision.transforms.functional import to_tensor


# ============================================================================
# CONFIGURACIÓN
# ============================================================================
NUM_CLASSES = 21  # 20 clases de Pascal VOC + fondo
MAX_PROPOSALS = 200
UMBRAL_CONFIANZA = 0.50
TAMANO_ROI = (7, 7)
NOMBRE_SALIDA = "fast_rcnn_detecciones.png"


class FastRCNN(nn.Module):
    """
    Implementación didáctica de Fast R-CNN.

    El backbone extrae un mapa de características compartido.
    RoI Pooling extrae características para cada propuesta.
    Dos cabezas predicen clase y ajustes de caja.

    NOTA: las cabezas se inicializan aleatoriamente; deben entrenarse con un
    dataset de detección antes de usarse para detecciones significativas.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        backbone_completo = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            backbone_completo.conv1,
            backbone_completo.bn1,
            backbone_completo.relu,
            backbone_completo.maxpool,
            backbone_completo.layer1,
            backbone_completo.layer2,
            backbone_completo.layer3,
            backbone_completo.layer4,
        )

        # ResNet-50 layer4 produce mapas de 2048 canales.
        self.roi_size = TAMANO_ROI
        self.fc1 = nn.Linear(2048 * self.roi_size[0] * self.roi_size[1], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self.box_regressor = nn.Linear(1024, num_classes * 4)

    def forward(self, images, proposals):
        """
        images: tensor [1, 3, H, W], valores en [0, 1].
        proposals: tensor [N, 4] con cajas x1,y1,x2,y2 en píxeles originales.
        """
        features = self.backbone(images)

        # Las coordenadas de RoI Pooling deben mapearse desde píxeles de
        # entrada al mapa de características.
        _, _, feature_h, feature_w = features.shape