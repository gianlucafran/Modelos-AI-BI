import cv2
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from ultralytics import YOLO

# ==========================================
# 1. Carga del Modelo de Segmentación
# ==========================================
print("Cargando modelo de segmentación en tiempo real...")
model = YOLO("yolov8n-seg.pt")  # Descarga automáticamente los pesos entrenados en COCO

# ==========================================
# 2. Efecto de Realidad Aumentada Interactiva
# ==========================================
def apply_ar_effect(frame, masks, classes, confidence_threshold=0.5):
    """
    Aplica un efecto de Realidad Aumentada Interactiva:
    Crea un mapa de textura holográfica/neón en las áreas segmentadas de los objetos.
    """
    ar_frame = frame.copy()
    
    if masks is None:
        return ar_frame

    # Generar un patrón virtual/holográfico para la AR
    h, w, _ = frame.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    pattern = (np.sin(x_coords / 10.0) + np.cos(y_coords / 10.0)) * 127 + 128
    
    # Textura neón/virtual (Cian/Azul)
    ar_texture = np.zeros_like(frame, dtype=np.uint8)
    ar_texture[:, :, 0] = pattern.astype(np.uint8)        # Canal Azul
    ar_texture[:, :, 1] = 255 - pattern.astype(np.uint8) # Canal Verde
    ar_texture[:, :, 2] = 200                            # Canal Rojo

    # Iterar sobre las máscaras de cada objeto detectado
    for mask, cls_id in zip(masks.data.cpu().numpy(), classes.cpu().numpy()):
        # Redimensionar la máscara al tamaño original de la imagen
        binary_mask = cv2.resize(mask, (w, h)) > 0.5
        
        if not np.any(binary_mask):
            continue

        # Corrección: Combinación alpha vectorizada con NumPy para subconjuntos enmascarados
        alpha = 0.4
        region_frame = frame[binary_mask].astype(np.float32)
        region_texture = ar_texture[binary_mask].astype(np.float32)
        blended = (1 - alpha) * region_frame + alpha * region_texture
        ar_frame[binary_mask] = blended.astype(np.uint8)
        
        # Dibujar un contorno neón alrededor de la máscara
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(ar_frame, contours, -1, (255, 255, 0), 2)

    return ar_frame


# ==========================================
# 3. Pipeline de Realidad Aumentada en Tiempo Real
# ==========================================
def run_interactive_ar_demo(source=0):
    """
    Ejecuta el pipeline de AR interactiva.
    source: 0 para Webcam o ruta a un archivo de video real.
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("No se pudo abrir la cámara o video. Usando una imagen de CIFAR-10 de prueba.")
        cifar = CIFAR10(root="./data", train=False, download=True)
        img_pil, _ = cifar[0]
        frame = cv2.cvtColor(np.array(img_pil.resize((640, 480))), cv2.COLOR_RGB2BGR)
        
        # Inferencia de segmentación
        results = model.predict(source=frame, conf=0.3)[0]
        ar_result = apply_ar_effect(frame, results.masks, results.boxes.cls)
        
        cv2.imwrite("ar_output_sample.jpg", ar_result)
        print("Resultado guardado en 'ar_output_sample.jpg'")
        return

    print("Iniciando Realidad Aumentada en tiempo real. Presiona 'q' para salir.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia rápida (Segmentación en tiempo real)
        results = model.predict(source=frame, conf=0.4, verbose=False)[0]
        
        # Aplicar superposición interactiva de AR
        if results.masks is not None:
            output_frame = apply_ar_effect(frame, results.masks, results.boxes.cls)
        else:
            output_frame = frame

        # Mostrar FPS e instrucciones
        cv2.putText(output_frame, "Realidad Aumentada YOLOv8 - Presiona 'q' para salir", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Interactive AR (Real-Time Instance Segmentation)", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Iniciar la demo usando la webcam principal (fuente 0)
    run_interactive_ar_demo(source=0)