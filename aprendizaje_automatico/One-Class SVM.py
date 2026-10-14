"""
One-Class SVM: Mantenimiento Predictivo Industrial
Demostración con datos numéricos normalizados generados mediante scikit-learn

Conceptos:
- One-Class SVM (Support Vector Machine de una sola clase): Algoritmo de aprendizaje 
  no supervisado diseñado para la Detección de Anomalías (Novelty/Anomaly Detection). 
  A diferencia de los SVM tradicionales, no requiere datos etiquetados de fallas; 
  se entrena exclusivamente con datos "normales" (operación nominal de maquinaria).
- Mantenimiento Predictivo: Estrategia industrial orientada a anticiparse a las fallas 
  mecánicas o eléctricas analizando vibraciones, temperatura, presión y consumo de corriente. 
  Cualquier desviación sutil del comportamiento nominal se clasifica como anomalía (-1).
- Hiperplano Óptimo: El modelo encuentra un hiperplano de separación de margen máximo 
  que aísla el 95% de los datos normales del origen en un espacio de alta dimensionalidad (Kernel RBF).
- CSV Numérico Normalizado: Los datos de sensores industriales tienen diferentes escalas 
  (ej. temperatura en °C vs vibración en mm/s). La normalización (StandardScaler) es 
  obligatoria para que el kernel SVM no sesgue su optimización hacia variables de mayor magnitud.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Scikit-Learn (Dataset sintético industrial, Normalización y One-Class SVM)
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# Configurar estilo visual profesional
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. GENERAR / CARGAR DATASET NUMÉRICO NORMALIZADO (MANTENIMIENTO PREDICTIVO)
# ============================================================================
print("=" * 80)
print("PASO 1: SIMULAR Y NORMALIZAR DATOS DE SENSORES INDUSTRIALES (CSV NORMALIZADO)")
print("=" * 80)

# Simulamos lecturas de sensores de maquinaria (ej. Temperatura del motor y Vibración RMS)
# Generamos 500 muestras de operación normal (Cluster principal)
np.random.seed(42)
n_normales = 500
n_anomalias = 35

X_normal, _ = make_blobs(n_samples=n_normales, n_features=2, centers=[[2.0, 2.0]], cluster_std=0.5, random_state=42)

# Generamos 35 anomalías reales (fallas incipientes: vibraciones y temperaturas anómalas)
X_anomalias = np.random.uniform(low=[4.0, 4.0], high=[6.5, 6.5], size=(n_anomalias, 2))

# Combinar dataset industrial completo
X_industrial = np.vstack([X_normal, X_anomalias])
# Etiquetas reales ocultas para evaluación: 1 = Normal, -1 = Anomalía (Falla)
y_real = np.ones(len(X_industrial), dtype=int)
y_real[n_normales:] = -1  # Las últimas 35 son fallas

# Guardar temporalmente como un archivo CSV simulando el entorno de producción
df_sensores = pd.DataFrame(X_industrial, columns=['Temperatura_Motor_Norm', 'Vibracion_RMS_Norm'])
df_sensores.to_csv('datos_sensores_industriales.csv', index=False)

print(f"✓ Dataset industrial generado y guardado en 'datos_sensores_industriales.csv'")
print(f"  - Total de registros de sensores: {len(df_sensores)}")
print(f"  - Muestras de operación nominal (Sanos): {n_normales}")
print(f"  - Muestras de fallas latentes (Anomalías): {n_anomalias}")

# ============================================================================
# 2. NORMALIZACIÓN Y PREPROCESAMIENTO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: NORMALIZACIÓN DE VARIABLES (STANDARD SCALER)")
print("=" * 80)

# Cargamos el CSV para replicar el pipeline de producción
df_cargado = pd.read_csv('datos_sensores_industriales.csv')
X_raw = df_cargado.values

# Estandarizar a media 0 y desviación estándar 1
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_raw)

print("✓ Datos normalizados correctamente (Media=0, Desviación Estándar=1).")
print("  - Esto asegura que el kernel RBF del SVM evalúe con equidad todas las métricas físicas.")

# ============================================================================
# 3. ENTRENAMIENTO DEL MODELO ONE-CLASS SVM
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAMIENTO DEL MODELO ONE-CLASS SVM (DETECCIÓN DE ANOMALÍAS)")
print("=" * 80)

# Configuración del One-Class SVM:
# - kernel='rbf': Permite modelar fronteras de decisión complejas y no lineales.
# - nu=0.07: Tasa esperada aproximada de anomalías en el dataset de entrenamiento (7%).
# - gamma='scale': Ajusta automáticamente la influencia del radio del kernel.
modelo_ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.07)

print("Entrenando One-Class SVM exclusivamente con el comportamiento nominal...")
# ¡CRÍTICO!: Entrenamos únicamente con datos limpios o con la mayor parte del dataset nominal
modelo_ocsvm.fit(X_norm[:450]) # Entrenamos con un subconjunto nominal puro

print("✓ Modelo entrenado. Generando predicciones sobre el conjunto total de sensores...")
y_pred = modelo_ocsvm.predict(X_norm) # Devuelve +1 (Normal) o -1 (Anomalía/Falla)

# Evaluación contra las etiquetas reales de falla
matriz_conf = confusion_matrix(y_real, y_pred)
print("\nMatriz de Confusión de Mantenimiento Predictivo:")
print(matriz_conf)

print("\nReporte de Clasificación:")
print(classification_report(y_real, y_pred, target_names=['Anomalía / Falla (-1)', 'Normal (+1)']))

# ============================================================================
# 4. VISUALIZACIÓN DE LA FRONTERA DE DECISIÓN Y ANOMALÍAS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: VISUALIZACIÓN DE LA FRONTERA DE DECISIÓN INDUSTRIAL")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: Dispersión de Sensores y Anomalías Detectadas por el OCSVM
colores = ['crimson' if pred == -1 else 'forestgreen' for pred in y_pred]
axes[0].scatter(X_norm[:, 0], X_norm[:, 1], c=colores, alpha=0.7, edgecolors='k', s=45)
axes[0].set_title('Monitoreo Predictivo OCSVM\n(Verde: Operación Nominal | Rojo: Anomalía Detectada)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Temperatura Normalizada (Z-Score)')
axes[0].set_ylabel('Vibración Normalizada (Z-Score)')
axes[0].grid(alpha=0.3)

# Gráfico 2: Frontera de Decisión (Meshgrid)
xx, yy = np.meshgrid(np.linspace(-3, 4, 200), np.linspace(-3, 4, 200))
Z = modelo_ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu, alpha=0.6)
axes[1].contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred') # Frontera de anomalía
axes[1].contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred', alpha=0.3) # Zona de anomalía

axes[1].scatter(X_norm[y_pred == 1, 0], X_norm[y_pred == 1, 1], c='forest