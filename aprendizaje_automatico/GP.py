"""
GP (Gaussian Processes): Optimización Bayesiana de Hiperparámetros
Demostración con dataset espacial/temporal sintético y scikit-learn

Conceptos:
- Procesos Gaussianos (Gaussian Processes - GP): Modelo no paramétrico basado en kernel 
  que define una distribución sobre funciones. No solo predice un valor esperado, 
  sino que proporciona una incertidud (desviación estándar) asociada a cada predicción.
- Optimización Bayesiana: Técnica avanzada para encontrar los hiperparámetros óptimos 
  de un modelo de Machine Learning sin evaluar todas las combinaciones posibles (Grid Search). 
  Utiliza un GP como "modelo sustituto" (surrogate model) para aproximar la función de coste.
- Función de Adquisición (Acquisition Function - Expected Improvement): Determina 
  inteligente dónde probar el siguiente hiperparámetro, balanceando la exploración 
  (zonas con alta incertidud) y la explotación (zonas con buen rendimiento conocido).
- Puntos Espaciales / Temporales: Los procesos gaussianos son nativamente modelos de 
  regresión espacial y temporal (Kriging en geoestadística).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')

# Configurar estilo visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. PREPARAR EL DATASET ESPACIAL / TEMPORAL (SCIKIT-LEARN)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR DATASET ESPACIAL/TEMPORAL (CALIFORNIA HOUSING)")
print("=" * 80)

# Cargamos el dataset de vivienda de California, que contiene coordenadas espaciales 
# reales (latitud y longitud) y características socioeconómicas.
california = fetch_california_housing(as_frame=True)
df = california.frame.sample(n=1000, random_state=42) # Submuestra para agilizar cómputo

# Extraemos coordenadas espaciales (Latitud y Longitud) como variables temporales/espaciales
X_espacial = df[['Latitude', 'Longitude']].values
y_valor = df['MedHouseVal'].values # Precio medio de la vivienda

print(f"\nDataset: California Housing (Datos Espaciales)")
print(f"Total de puntos espaciales analizados: {X_espacial.shape[0]}")
print(f"Dimensiones geográficas: Latitud [{X_espacial[:, 0].min():.2f}, {X_espacial[:, 0].max():.2f}], "
      f"Longitud [{X_espacial[:, 1].min():.2f}, {X_espacial[:, 1].max():.2f}]")

# ============================================================================
# 2. MODELADO CON PROCESOS GAUSSIANOS (GP REGRESSOR)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: ENTRENAMIENTO DEL PROCESO GAUSSIANO SOBRE PUNTOS ESPACIALES")
print("=" * 80)

# Definimos el Kernel (Matriz de Covarianza): 
# Usamos un kernel Matérn (v=1.5) combinado con un Kernel constante para modelar 
# la correlación espacial suave entre casas cercanas.
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)

# Instanciar el regresor de Procesos Gaussianos
gp = GaussianProcessRegressor(
    kernel=kernel, 
    alpha=1e-2,          # Ruido estimado en las mediciones espaciales
    n_restarts_optimizer=10, 
    random_state=42
)

print("Ajustando el Proceso Gaussiano a la distribución espacial...")
# Tomamos una pequeña muestra de entrenamiento espacial para interpolar el mapa
X_train_gp = X_espacial[:150]
y_train_gp = y_valor[:150]

gp.fit(X_train_gp, y_train_gp)

print(f"✓ GP ajustado exitosamente.")
print(f"  - Kernel optimizado: {gp.kernel_}")

# ============================================================================
# 3. OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS USANDO GP
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS")
print("=" * 80)

print("Problema: Queremos encontrar la profundidad óptima (max_depth) y el "
      "número mínimo de muestras por hoja (min_samples_split) para un modelo de "
      "Regresión de Árbol de Decisión sin probar todas las combinaciones a ciegas.")

# Definiremos una función de coste simulada (Error de validación cruzada en negativo)
def funcion_objetivo(hiperparametros):
    max_depth = int(np.round(hiperparametros[0]))
    min_samples_split = int(np.round(hiperparametros[1]))
    
    if max_depth < 1 or min_samples_split < 2:
        return 0.0 # Penalización por parámetros inválidos
        
    modelo = DecisionTreeRegressor(
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        random_state=42
    )
    
    # Validación cruzada (negativa porque buscamos maximizar el R^2 o minimizar MSE)
    scores = cross_val_score(modelo, X_espacial, y_valor, cv=3, scoring='r2')
    return np.mean(scores)

# Inicializamos puntos evaluados aleatoriamente
np.random.seed(42)
X_muestras = np.random.uniform(low=[1, 2], high=[15, 20], size=(5, 2))
y_evaluados = np.array([funcion_objetivo(x) for x in X_muestras])

# Iteraciones de Optimización Bayesiana (Modelo Sustituto con GP)
n_iteraciones = 10
print(f"\nEjecutando {n_iteraciones} iteraciones de Optimización Bayesiana...")

for i in range(n_iteraciones):
    # Entrenamos un GP sobre las evaluaciones anteriores de hiperparámetros
    gp_optimizer = GaussianProcessRegressor(kernel=RBF(), alpha=1e-4, random_state=42)
    gp_optimizer.fit(X_muestras, y_evaluados)
    
    # Generar candidatos aleatorios en el espacio de búsqueda para evaluar la función de adquisición
    candidatos = np.random.uniform(low=[1, 2], high=[15, 20], size=(100, 2))
    mu, sigma = gp_optimizer.predict(candidatos, return_std=True)
    
    # Función de Adquisición simple (Upper Confidence Bound - UCB)
    ucb = mu + 1.96 * sigma
    siguiente_punto = candidatos[np.argmax(ucb)]
    
    # Evaluar el nuevo punto en la función objetivo real
    resultado_real = funcion_objetivo(siguiente_punto)
    
    # Agregar al histórico
    X_muestras = np.vstack((X_muestras, siguiente_punto))
    y_evaluados = np.append(y_evaluados, resultado_real)
    
    print(f"  - Iteración {i+1:2d}: max_depth={siguiente_punto[0]:.1f}, min_samples={siguiente_punto[1]:.1f} | R² Evaluado: {resultado_real:.4f}")

# Encontrar el mejor hiperparámetro encontrado por el GP
mejor_idx = np.argmax(y_evaluados)
mejor_hiperparametro = X_muestras[mejor_idx]
print(f"\n✓ ¡Optimización Concluida!")
print(f"  - Mejores Hiperparámetros encontrados: max_depth={mejor_hiperparametro[0]:.0f}, min_samples_split={mejor_hiperparametro[1]:.0f}")
print(f"  - Mejor puntuación R² lograda: {y_evaluados[mejor_idx]:.4f}")

# ============================================================================
# 4. VISUALIZACIÓN: PREDICCIÓN ESPACIAL Y MODELO SUSTITUTO GP
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: VISUALIZACIÓN DEL PROCESO GAUSSIANO Y LA OPTIMIZACIÓN")
print("=" * 80)

# Generar una malla espacial para graficar la interpolación del GP
lat_min, lat_max = X_espacial[:, 0].min(), X_espacial[:, 0].max()
lon_min, lon_max = X_espacial[:, 1].min(), X_espacial[:, 1].max()

xx, yy = np.meshgrid(np.linspace(lat_min, lat_max, 50), np.linspace(lon_min, lon_max, 50))
malla_puntos = np.c_[xx.ravel(), yy.ravel()]

# Predicción de la media y la incertidud (desviación estándar) en el mapa
pred_media, pred_std = gp.predict(malla_puntos, return_std=True)
pred_media_malla = pred_media.reshape(xx.shape)
pred_std_malla = pred_std.reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: Predicción Espacial (Kriging / Media del GP)
im1 = axes[0].contourf(yy, xx, pred_media_malla, levels=20, cmap='viridis', alpha=0.8)
axes[0].scatter(X_train_gp[:, 1], X_train_gp[:, 0], c=y_train_gp, cmap='viridis', edgecolor='white', s=50, label='Muestras Espaciales')
axes[0].set_title('Modelo Espacial GP: Predicción de Precios en el Mapa', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Longitud')
axes[0].set_ylabel('Latitud')
fig.colorbar(im1, ax=axes[0], label='Valor Estimado de la Vivienda')
axes[0].legend(loc='upper right')

# Gráfico 2: Incertidud del Proceso Gaussiano (Desviación Estándar)
im2 = axes[1].contourf(yy, xx, pred_std_malla, levels=20, cmap='magma', alpha=0.8)
axes[1].scatter(X_train_gp[:, 1], X_train_gp[:, 0], color='white', edgecolor='black', s=30, label='Puntos Observados')
axes[1].set_title('Incertidud del GP (Desviación Estándar $\sigma$)\n(Mayor incertidumbre lejos de las observaciones)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Longitud')
axes[1].set_ylabel('Latitud')
fig.colorbar(im2, ax=axes[1], label='Incertidud ($\sigma$)')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('/home/usuario/01_gaussian_process_spatial.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_gaussian_process_spatial.png")
plt.close()

# ============================================================================
# 5. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN TEÓRICO")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               PROCESOS GAUSSIANOS (GP) Y OPTIMIZACIÓN BAYESIANA              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. NATURALEZA ESPACIAL Y TEMPORAL:
   • En geoestadística y series temporales, los Procesos Gaussianos (Kriging) 
     modelan la correlación entre puntos cercanos. 
   • Como se observa en el Gráfico 2, la incertidud ($\sigma$) es baja cerca de 
     las muestras reales observadas y aumenta de forma controlada en las regiones 
     donde el GP no tiene información, reflejando matemáticamente nuestro nivel de duda.

2. OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS:
   • Grid Search prueba a ciegas una cuadrícula fija. Random Search prueba al azar.
   • La Optimización Bayesiana utiliza un GP como "modelo sustituto". 
     El GP aprende la forma matemática de la función de coste y predice qué 
     combinación de hiperparámetros tiene la mayor probabilidad de mejorar el rendimiento, 
     ahorrando cientos de horas de entrenamiento en modelos pesados de Deep Learning o Machine Learning.

3. FUNCIÓN DE ADQUISICIÓN (EXPLORACIÓN VS EXPLOTACIÓN):
   • El algoritmo calcula dónde hay mejores resultados conocidos (explotación) y 
     dónde hay mayor incertidumbre (exploración), eligiendo inteligentemente el 
     siguiente punto a evaluar en cada iteración.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_gaussian_process_spatial.png - Visualización bivariada de la media de predicción 
                                  espacial y el mapa de incertidumbre ($\sigma$) 
                                  del Proceso Gaussiano.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de Procesos Gaussianos y Optimización Bayesiana completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")