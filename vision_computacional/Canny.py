#pip install opencv-python matplotlib
import cv2
import matplotlib.pyplot as plt

def detectar_bordes_canny(ruta_imagen, umbral_min=100, umbral_max=200):
    # 1. Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen en la ruta '{ruta_imagen}'")
        return

    # 2. Convertir a escala de grises (requerido para Canny)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # 3. Aplicar suavizado Gaussiano (reduce el ruido para no detectar bordes falsos)
    desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)

    # 4. Aplicar el algoritmo de Canny
    # Los píxeles con un gradiente superior a umbral_max se marcan como bordes.
    # Los inferiores a umbral_min se descartan.
    bordes = cv2.Canny(desenfoque, umbral_min, umbral_max)

    # 5. Visualizar los resultados
    plt.figure(figsize=(12, 6))
    
    # Mostrar imagen original (OpenCV carga en BGR, matplotlib usa RGB)
    plt.subplot(1, 2, 1)
    plt.title('Imagen Original')
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Mostrar imagen con bordes
    plt.subplot(1, 2, 2)
    plt.title('Detección de Bordes (Canny)')
    plt.imshow(bordes, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Opcional: Guardar la imagen resultante
    # cv2.imwrite('resultado_canny.jpg', bordes)

# --- EJECUCIÓN DEL SCRIPT ---
# Reemplaza 'ruta_a_tu_imagen.jpg' con la ruta real de tu archivo
ruta_archivo = r"C:\Users\Tecnologias\Downloads\DetectEdgesInImagesExample_01.png"
detectar_bordes_canny(ruta_archivo)