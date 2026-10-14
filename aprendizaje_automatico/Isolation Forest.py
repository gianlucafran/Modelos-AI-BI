"""
Isolation Forest: Detección de Anomalías y Prevención de Fraudes Financieros
Demostración con dataset sintético altamente desbalanceado de scikit-learn

Conceptos:
- Isolation Forest: Algoritmo de detección de anomalías basado en árboles. En lugar 
  de modelar el comportamiento normal, aísla las anomalías explícitamente.
- Principio de Aislamiento: Las anomalías (fraudes) son "Pocas y Diferentes", por lo que 
  requieren menos particiones aleatorias para ser aisladas en un nodo de un árbol.
- Contamination (Contaminación): Hiperparámetro que indica la proporción esperada 
  de anomalías en el dataset.
- Fraude Financiero: Problema clásico de clases altamente desbalanceadas (ej. 99% 
  transacciones legítimas, 1% fraudulentas).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. GENERAR Y EXPLORAR EL DATASET (SIMULACIÓN DE FRAUDE FINANCIERO)
# ============================================================================
print("=" * 80)
print("PASO 1: SIMULAR DATASET DE TRANSACCIONES FINANCIERAS")
print("=" * 80)

# Scikit-learn no incluye datasets financieros reales por privacidad, 
# usaremos make_classification para simular uno altamente desbalanceado.
# Simularemos 10,000 transacciones donde solo el 1% es fraude.

n_samples = 10000
fraud_ratio = 0.01

X, y = make_classification(
    n_samples=n_samples, 
    n_features=5,           # 5 características simuladas de la transacción
    n_informative=3,        # 3 de ellas son realmente útiles para detectar fraude
    n_redundant=2, 
    n_classes=2, 
    weights=[1 - fraud_ratio, fraud_ratio], # 99% Normal, 1% Fraude
    class_sep=1.2,          # Separación moderada (el fraude suele camuflarse)
    random_state=42
)

# Nombres de columnas simulando variables bancarias
feature_names = [
    'Monto_ZScore', 
    'Distancia_Ubicacion_IP', 
    'Tasa_Frecuencia_Diaria', 
    'Riesgo_Comercio', 
    'Desviacion_Horario'
]

df = pd.DataFrame(X, columns=feature_names)
df['Es_Fraude_Real'] = y

print(f"\nDataset: Transacciones Financieras Simuladas")
print(f"Total de transacciones: {df.shape[0]}")
print(f"Número de características (variables): {df.shape[1] - 1}")

# Desbalanceo de clases
counts = df['Es_Fraude_Real'].value_counts()
print("\nDistribución de las Clases (Desbalanceo Extremo):")
print(f"  Transacciones Legítimas (0): {counts[0]} ({(counts[0]/n_samples)*100:.1f}%)")
print(f"  Fraudes Reales (1):          {counts[1]} ({(counts[1]/n_samples)*100:.1f}%)")

print(f"\nPrimeras 5 transacciones:")
print(df.head().round(3))

# ============================================================================
# 2. ENTRENAMIENTO DEL ISOLATION FOREST
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: ENTRENAR ISOLATION FOREST")
print("=" * 80)

# Isolation Forest es No Supervisado. No le daremos las etiquetas 'y' para entrenar.
# El modelo intentará encontrar las anomalías basándose puramente en la estructura de X.

print("\nConfigurando Isolation Forest...")
# El parámetro 'contamination' es crucial. Le dice al modelo qué porcentaje 
# de los datos asumimos que son anomalías.
modelo_if = IsolationForest(
    n_estimators=100,        # Número de árboles en el bosque
    max_samples='auto',      # Número de muestras para entrenar cada árbol
    contamination=fraud_ratio, # Estimación inicial de fraude (1%)
    random_state=42,
    n_jobs=-1
)

print("Entrenando el modelo (identificando patrones de aislamiento)...")
# Entrenamos SOLO con las características (X), ignorando si es fraude o no (y)
modelo_if.fit(X)

print("✓ Modelo entrenado exitosamente.")

# ============================================================================
# 3. PREDICCIÓN Y TRANSFORMACIÓN DE ETIQUETAS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: DETECCIÓN DE FRAUDES (ANOMALÍAS)")
print("=" * 80)

# El modelo devuelve:
#   1 para inliers (transacciones normales)
#  -1 para outliers (anomalías/fraude)
predicciones_if = modelo_if.predict(X)

# Transformamos estas predicciones al formato de nuestro dataset (0 y 1)
# Si es -1 (Anomalía), lo marcamos como 1 (Fraude). Si es 1 (Normal), lo marcamos como 0.
y_pred_fraude = np.where(predicciones_if == -1, 1, 0)
df['Fraude_Predicho'] = y_pred_fraude

# Anomalys score: Valores negativos menores indican mayor anormalidad
scores_anomalia = modelo_if.decision_function(X)
df['Score_Anomalia'] = scores_anomalia

anomalias_detectadas = np.sum(y_pred_fraude == 1)
print(f"\nEl modelo ha etiquetado {anomalias_detectadas} transacciones como posibles FRAUDES.")
print("Recuerda: El Isolation Forest aisló estos puntos más rápido en sus ramas de decisión.")

# ============================================================================
# 4. EVALUACIÓN DEL RENDIMIENTO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: EVALUACIÓN DE LA DETECCIÓN")
print("=" * 80)

print("\nMatriz de Confusión:")
cm = confusion_matrix(y, y_pred_fraude)
print(pd.DataFrame(cm, 
                   index=['Real_Legítima (0)', 'Real_Fraude (1)'], 
                   columns=['Pred_Legítima (0)', 'Pred_Fraude (1)']))

print("\nFalsos Positivos (Bloqueos de tarjetas por error):", cm[0, 1])
print("Falsos Negativos (Fraudes que se escaparon):", cm[1, 0])
print("Verdaderos Positivos (Fraudes detectados exitosamente):", cm[1, 1])

print("\nReporte de Clasificación:")
print(classification_report(y, y_pred_fraude, target_names=['Legítima', 'Fraude']))

# ============================================================================
# 5. VISUALIZACIÓN: ESPACIO DE TRANSACCIONES
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN MEDIANTE PCA")
print("=" * 80)

# Reducimos las 5 variables bancarias a 2 para poder graficarlas
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Fraudes Reales
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', 
                           alpha=0.6, s=20, edgecolor='k', linewidth=0.5)
axes[0].set_title('Realidad: Transacciones Legítimas vs Fraudes Reales', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Componente Principal 1')
axes[0].set_ylabel('Componente Principal 2')
legend1 = axes[0].legend(handles=scatter1.legend_elements()[0], labels=['Legítima (0)', 'Fraude (1)'])
axes[0].grid(alpha=0.3)

# Gráfico 2: Fraudes Predichos por Isolation Forest
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_fraude, cmap='coolwarm', 
                           alpha=0.6, s=20, edgecolor='k', linewidth=0.5)
axes[1].set_title('Detección Isolation Forest (Algoritmo No Supervisado)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Componente Principal 1')
axes[1].set_ylabel('Componente Principal 2')
legend2 = axes[1].legend(handles=scatter2.legend_elements()[0], labels=['Normal (Inlier)', 'Anomalía (Outlier)'])
axes[1].grid(alpha=0.3)

# Destacar las zonas donde el algoritmo detectó las anomalías (suelen estar en la periferia)
axes[1].text(0.05, 0.95, "Nota: El algoritmo detecta\npuntos en la 'periferia'\ndel espacio multidimensional.", 
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/usuario/01_fraude_isolation_forest.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_fraude_isolation_forest.png")
plt.close()

# ============================================================================
# 6. VISUALIZACIÓN: DISTRIBUCIÓN DEL SCORE DE ANOMALÍA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: ANÁLISIS DEL SCORE DE ANOMALÍA")
print("=" * 80)

fig, ax = plt.subplots(figsize=(10, 5))

# Graficamos el score de anomalía devuelto por Isolation Forest
sns.histplot(data=df, x='Score_Anomalia', hue='Es_Fraude_Real', bins=50, 
             palette={0: 'blue', 1: 'red'}, alpha=0.6, ax=ax)

# Línea divisoria que el algoritmo eligió (basado en contamination=0.01)
umbral = df[df['Fraude_Predicho'] == 1]['Score_Anomalia'].max()
ax.axvline(umbral, color='black', linestyle='--', linewidth=2, 
           label=f'Umbral de decisión (Contamination={fraud_ratio})')

ax.set_title('Distribución del Score de Anomalía (Valores menores = Más anómalos)', 
             fontsize=13, fontweight='bold')
ax.set_xlabel('Score de Anomalía (Decision Function)')
ax.set_ylabel('Frecuencia')
ax.legend()

plt.tight_layout()
plt.savefig('/home/usuario/02_score_anomalia.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 02_score_anomalia.png")
plt.close()

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ISOLATION FOREST PARA PREVENCIÓN DE FRAUDES                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. EL PROBLEMA DEL FRAUDE BANCARIO:
   • El desbalanceo es extremo: De {n_samples} transacciones, solo el {fraud_ratio*100}% son fraude.
   • Los algoritmos supervisados tradicionales (ej. Regresión Logística) suelen fallar 
     porque simplemente aprenden a decir "todo es normal" y obtienen 99% de precisión.

2. ¿CÓMO FUNCIONA ISOLATION FOREST?
   • Selecciona una variable al azar y un valor de corte al azar.
   • Repite este proceso dividiendo los datos en ramas (como un árbol).
   • LÓGICA CLAVE: Las anomalías son diferentes, por lo que se quedan aisladas en ramas 
     muy cortas (cerca de la raíz). Las transacciones normales requieren muchas divisiones 
     para ser separadas de otras normales.

3. RESULTADOS OBTENIDOS (NO SUPERVISADO):
   • Verdaderos Fraudes Detectados: {cm[1, 1]} de {counts[1]} (Recall de Fraude).
   • Falsas Alarmas (Tarjetas bloqueadas por error): {cm[0, 1]}.
   • A diferencia de técnicas supervisadas, el Isolation Forest detectó estos fraudes 
     sin que le dijéramos explícitamente cuáles eran durante el entrenamiento.

4. EL PAPEL DEL PARÁMETRO "CONTAMINATION":
   • Se fijó en {fraud_ratio}. Esto instruyó al algoritmo a buscar el {fraud_ratio*100}% 
     más anómalo de los datos.
   • En la vida real (bancos), si el equipo de prevención de fraudes quiere atrapar más 
     fraudes, pueden subir este valor (ej. 0.05), a costa de generar más falsas alarmas 
     (requiere intervención humana/llamadas al cliente).

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_fraude_isolation_forest.png - Proyección PCA (Realidad vs Detección IF)
02_score_anomalia.png          - Histograma que muestra cómo los fraudes tienen 
                                 los scores más bajos (negativos) en el modelo.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de prevención de fraudes completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")