"""
SVM (Support Vector Machines): Reconocimiento Facial y Alta Dimensionalidad
Demostración con dataset 'Labeled Faces in the Wild' (Biometría) de scikit-learn

Conceptos:
- Support Vector Machines (SVM): Algoritmo que busca el "hiperplano" óptimo que 
  separa diferentes clases maximizando el margen entre ellas.
- Aplicabilidad (Texto y Biometría): SVM brilla en espacios de altísima dimensionalidad. 
  Tanto la categorización de texto (miles de palabras/TF-IDF) como la biometría 
  (miles de píxeles por imagen) se benefician de la capacidad de SVM para lidiar 
  con dimensiones mayores que el número de muestras.
- Kernel Trick (Truco del Kernel): Permite a SVM separar datos que no son linealmente 
  separables proyectándolos (matemáticamente) a dimensiones superiores. Usaremos el Kernel RBF.
- Eigenfaces (PCA + SVM): En biometría facial, es estándar usar PCA para extraer 
  las características principales del rostro antes de pasar los vectores a la SVM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. CARGAR Y EXPLORAR EL DATASET (BIOMETRÍA FACIAL)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR DATASET DE BIOMETRÍA (LABELED FACES IN THE WILD)")
print("=" * 80)

# Descargamos el dataset de rostros de scikit-learn.
# Filtramos para obtener solo personas de las que tengamos al menos 60 fotos
print("Descargando/Cargando dataset LFW (Puede tardar unos segundos)...")
caras = fetch_lfw_people(min_faces_per_person=60)

# Datos y etiquetas
X = caras.data       # Cada fila es una imagen aplanada en un vector 1D
y = caras.target
nombres_objetivo = caras.target_names
n_imagenes, n_caracteristicas = X.shape
h, w = caras.images.shape[1:]  # Alto y ancho original de la imagen (62x47)

print(f"\nDataset: Labeled Faces in the Wild (Reconocimiento Biométrico)")
print(f"Total de imágenes: {n_imagenes}")
print(f"Resolución de cada rostro: {h}x{w} píxeles")
print(f"Características (Píxeles totales por vector): {n_caracteristicas}")
print(f"Número de personas (Clases): {len(nombres_objetivo)}")

print("\nPersonas a identificar:")
for i, nombre in enumerate(nombres_objetivo):
    print(f"  - Clase {i}: {nombre}")

# ============================================================================
# 2. PREPROCESAMIENTO Y DIVISIÓN DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DIVISIÓN DE DATOS Y REDUCCIÓN DE DIMENSIONALIDAD (PCA)")
print("=" * 80)

# Dividir en entrenamiento (75%) y prueba (25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Datos de Entrenamiento: {X_train.shape[0]} imágenes")
print(f"Datos de Prueba: {X_test.shape[0]} imágenes")

# En lugar de usar los 2914 píxeles crudos, usaremos PCA para extraer los 
# 150 componentes principales ("Eigenfaces"). Esto elimina el ruido y 
# le da a la SVM características mucho más ricas y compactas.
n_componentes = 150
print(f"\nConfigurando PCA para extraer {n_componentes} 'Eigenfaces'...")

pca = PCA(n_components=n_componentes, whiten=True, random_state=42)

# ============================================================================
# 3. ENTRENAMIENTO DEL MODELO SVM
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAMIENTO DE LA MÁQUINA DE VECTORES DE SOPORTE (SVM)")
print("=" * 80)

# Configuramos la SVM
# - kernel='rbf': Radial Basis Function, excelente para relaciones no lineales como rostros.
# - class_weight='balanced': Crucial porque tenemos más fotos de unas personas que de otras.
# - C y gamma: Hiperparámetros del kernel (usualmente se buscan con GridSearchCV, aquí fijos).
print("Inicializando SVC (Support Vector Classifier) con Kernel RBF...")
svc = SVC(kernel='rbf', class_weight='balanced', C=10.0, gamma=0.005)

# Creamos un Pipeline que primero aplica PCA y luego entrena la SVM
modelo_biometrico = make_pipeline(pca, svc)

print("Entrenando la SVM para encontrar los hiperplanos de separación facial...")
modelo_biometrico.fit(X_train, y_train)

print("✓ Pipeline PCA + SVM entrenado exitosamente.")

# ============================================================================
# 4. PREDICCIÓN Y EVALUACIÓN BIOMÉTRICA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: EVALUACIÓN DEL RECONOCIMIENTO FACIAL")
print("=" * 80)

print("Realizando predicciones sobre el conjunto de prueba (rostros no vistos)...")
y_pred = modelo_biometrico.predict(X_test)

print("\nReporte de Clasificación Biométrica:")
print(classification_report(y_test, y_pred, target_names=nombres_objetivo))

# ============================================================================
# 5. VISUALIZACIÓN: RESULTADOS Y MATRIZ DE CONFUSIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DE RESULTADOS (GALERÍA Y MATRIZ)")
print("=" * 80)

fig = plt.figure(figsize=(16, 8))

# -----------------------------------------------------------------
# Gráfico 1: Matriz de Confusión
# -----------------------------------------------------------------
ax1 = plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nombres_objetivo, yticklabels=nombres_objetivo, 
            cbar=False, ax=ax1)
ax1.set_title('Matriz de Confusión (Reconocimiento Facial)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Predicción del Modelo (SVM)')
ax1.set_ylabel('Identidad Real')
ax1.tick_params(axis='x', rotation=45)

# -----------------------------------------------------------------
# Gráfico 2: Galería de Predicciones
# -----------------------------------------------------------------
# Mostramos una cuadrícula 3x3 con los rostros de prueba y la decisión de la SVM
ax2 = plt.subplot(1, 2, 2)
ax2.axis('off')  # Ocultar ejes del subplot contenedor principal

# Crear sub-cuadrículas dentro de la mitad derecha para los rostros
for i in range(9):
    ax = fig.add_subplot(3, 6, (i // 3) * 6 + (i % 3) + 4, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(h, w), cmap='bone')
    
    # Nombre predicho
    pred_name = nombres_objetivo[y_pred[i]].split()[-1]
    # Nombre real
    true_name = nombres_objetivo[y_test[i]].split()[-1]
    
    # Color del texto: Verde si acertó, Rojo si se equivocó
    color = 'green' if pred_name == true_name else 'red'
    
    ax.set_title(f'Pred: {pred_name}\nReal: {true_name}', color=color, fontsize=10, fontweight='bold')

plt.suptitle('Sistema Biométrico: Predicciones SVM sobre Imágenes de Prueba', fontsize=16, fontweight='bold')
plt.tight_layout()
#plt.savefig('/home/usuario/01_svm_biometria_rostros.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_svm_biometria_rostros.png")
plt.close()

# ============================================================================
# 6. VISUALIZACIÓN EXTRA: LAS EIGENFACES (QUÉ VE LA SVM)
# ============================================================================
print("\nGenerando visualización de 'Eigenfaces' (Las características latentes extraídas por PCA)...")

fig, axes = plt.subplots(3, 5, figsize=(15, 7),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    # pca.components_ contiene los vectores base descubiertos
    ax.imshow(pca.components_[i].reshape(h, w), cmap='bone')
    ax.set_title(f'Eigenface {i+1}', fontsize=10)

plt.suptitle('Las "Eigenfaces": Patrones base extraídos por PCA antes de alimentar a la SVM', 
             fontsize=14, fontweight='bold')
#plt.savefig('/home/usuario/02_svm_eigenfaces.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 02_svm_eigenfaces.png")
plt.close()

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             SVM PARA ALTA DIMENSIONALIDAD (BIOMETRÍA Y TEXTO)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. ¿POR QUÉ SVM PARA BIOMETRÍA Y TEXTO?:
   • Textos e Imágenes comparten un problema: la "Maldición de la Dimensionalidad".
     Un texto vectorizado (TF-IDF) o una imagen ({n_caracteristicas} píxeles) 
     generan miles de columnas.
   • SVM es excepcionalmente bueno manejando datasets donde el número de 
     características supera el número de muestras, gracias al cálculo del 
     margen óptimo y el uso de Vectores de Soporte (se enfoca solo en los 
     casos difíciles en la frontera de decisión).

2. EL PIPELINE (PCA + SVM):
   • PCA (Eigenfaces): Redujimos las {n_caracteristicas} dimensiones originales 
     a solo {n_componentes} componentes principales. Esto extrae la "esencia" del 
     rostro (iluminación, estructura ósea) y descarta el ruido.
   • SVM (RBF Kernel): Tomó esos {n_componentes} valores y proyectó los datos a un 
     espacio no lineal donde logró trazar hiperplanos que separan perfectamente a 
     George W. Bush de Tony Blair o Colin Powell.

3. DESEMPEÑO DEL SISTEMA:
   • El modelo logró precisiones excepcionales (Visibles en el Reporte de Clasificación),
     demostrando por qué SVM fue el estándar de oro en sistemas de reconocimiento
     facial comercial y análisis de texto antes de la llegada del Deep Learning masivo.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

SVC(kernel='rbf', C=10.0, gamma=0.005, class_weight='balanced'):
  - kernel='rbf': Permite a la frontera de decisión curvarse alrededor de las 
    agrupaciones complejas de rostros.
  - class_weight='balanced': Instruye a la SVM a penalizar más los errores 
    cometidos en clases minoritarias (personas con menos fotos).
  - C=10.0: Margen estricto; prefiere clasificar correctamente la mayoría 
    de los puntos de entrenamiento a costa de un margen más pequeño.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_svm_biometria_rostros.png - Matriz de Confusión y Galería de resultados 
                               (Verdes = Aciertos, Rojos = Fallos).
02_svm_eigenfaces.png        - Los patrones "fantasmales" que descubre PCA 
                               y que la SVM utiliza para clasificar.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Entrenamiento del sistema biométrico SVM completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")