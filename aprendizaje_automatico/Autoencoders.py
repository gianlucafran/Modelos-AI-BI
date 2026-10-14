"""
Autoencoders: Detección de Anomalías Complejas
Demostración con dataset Breast Cancer de scikit-learn (adaptado a Detección de Novedades)

Conceptos:
- Autoencoder (Codificador Automático): Red neuronal no supervisada diseñada para copiar 
  su entrada a su salida. Compuesta por un Encoder (compresión) y un Decoder (reconstrucción).
- Cuello de Botella (Bottleneck): Capa oculta central con menor dimensionalidad que fuerza 
  a la red a aprender los patrones esenciales (espacio latente) ignorando el ruido.
- Error de Reconstrucción: Diferencia (MSE) entre la entrada original y la salida reconstruida.
- Detección de Anomalías: Se entrena el Autoencoder SOLO con datos normales. Cuando se le 
  presenta una anomalía, no sabe cómo reconstruirla adecuadamente, generando un error alto.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. CARGAR Y ADAPTAR EL DATASET PARA DETECCIÓN DE ANOMALÍAS
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR DATASET Y ADAPTARLO PARA DETECCIÓN DE NOVEDADES")
print("=" * 80)

# Usamos el dataset Breast Cancer (30 características de células).
# Asumiremos que las células Benignas son nuestro comportamiento "Normal" 
# y las Malignas son "Anomalías Complejas" a detectar.
data = load_breast_cancer()
X = data.data
y = data.target  # 1 = Benigno (Normal), 0 = Maligno (Anomalía)
feature_names = data.feature_names

# Invertimos las etiquetas para estandarizar: 0 = Normal, 1 = Anomalía (Fraude/Falla/Enfermedad)
y_anomaly = np.where(y == 1, 0, 1)

df = pd.DataFrame(X, columns=feature_names)
df['Es_Anomalia'] = y_anomaly

print(f"\nDataset: Breast Cancer (Adaptado)")
print(f"Número total de muestras: {df.shape[0]}")
print(f"Dimensiones (características complejas): {df.shape[1] - 1}")

# Separamos los datos. El modelo solo debe ver datos NORMALES durante el entrenamiento.
X_normal = df[df['Es_Anomalia'] == 0].drop('Es_Anomalia', axis=1).values
X_anomalo = df[df['Es_Anomalia'] == 1].drop('Es_Anomalia', axis=1).values

# Dividimos los normales en Entrenamiento (para el autoencoder) y Prueba
X_train_normal, X_test_normal = train_test_split(X_normal, test_size=0.2, random_state=42)

# El conjunto de prueba final contendrá tanto los normales no vistos como TODAS las anomalías
X_test = np.vstack((X_test_normal, X_anomalo))
y_test = np.hstack((np.zeros(len(X_test_normal)), np.ones(len(X_anomalo))))

print("\nDistribución de Datos:")
print(f"  Datos Normales para ENTRENAMIENTO: {X_train_normal.shape[0]}")
print(f"  Datos Normales para PRUEBA:        {X_test_normal.shape[0]}")
print(f"  Datos Anómalos para PRUEBA:        {X_anomalo.shape[0]}")

# ============================================================================
# 2. ESTANDARIZACIÓN (VITAL PARA REDES NEURONALES)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: ESTANDARIZACIÓN (PREPROCESAMIENTO)")
print("=" * 80)

scaler = StandardScaler()

# Se ajusta el scaler SOLO con los datos de entrenamiento normales
X_train_scaled = scaler.fit_transform(X_train_normal)

# Se transforman los datos de prueba usando el mismo scaler
X_test_scaled = scaler.transform(X_test)

print("\n✓ Datos escalados utilizando StandardScaler.")
print("  El escalado evita que características con valores altos dominen la función")
print("  de pérdida (MSE) durante el aprendizaje de la red neuronal.")

# ============================================================================
# 3. CONSTRUCCIÓN Y ENTRENAMIENTO DEL AUTOENCODER
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: DISEÑO Y ENTRENAMIENTO DEL AUTOENCODER")
print("=" * 80)

# Scikit-learn no tiene una clase "Autoencoder" explícita, pero podemos construir 
# uno usando MLPRegressor (Perceptrón Multicapa para Regresión), entrenándolo para 
# que prediga su propia entrada (X_train_scaled -> X_train_scaled).

# Arquitectura: 30 (Entrada) -> 16 -> 8 (Cuello de Botella) -> 16 -> 30 (Salida)
arquitectura = (16, 8, 16)

print(f"\nArquitectura de la Red (Dimensiones): 30 -> {arquitectura[0]} -> {arquitectura[1]} -> {arquitectura[2]} -> 30")
print("El 'Cuello de Botella' fuerza a la red a comprimir la información a solo 8 dimensiones.")

autoencoder = MLPRegressor(
    hidden_layer_sizes=arquitectura,
    activation='relu',       # Función de activación no lineal
    solver='adam',           # Optimizador
    learning_rate_init=0.001,
    max_iter=500,            # Épocas máximas
    random_state=42,
    early_stopping=True      # Detiene el entrenamiento si no mejora
)

print("\nEntrenando la red neuronal (Aprendiendo el patrón normal)...")
# La variable objetivo 'y' es la misma matriz de características 'X'
autoencoder.fit(X_train_scaled, X_train_scaled)

print(f"✓ Entrenamiento completado en {autoencoder.n_iter_} iteraciones.")
print(f"  Pérdida final de entrenamiento (MSE): {autoencoder.loss_:.4f}")

# ============================================================================
# 4. CÁLCULO DEL ERROR DE RECONSTRUCCIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: CÁLCULO DEL ERROR DE RECONSTRUCCIÓN")
print("=" * 80)

# Pasamos los datos de prueba (que contienen normales y anomalías) por la red
X_test_reconstructed = autoencoder.predict(X_test_scaled)

# Calculamos el Error Cuadrático Medio (MSE) por cada muestra (fila)
mse_reconstruction = np.mean(np.power(X_test_scaled - X_test_reconstructed, 2), axis=1)

# Agregamos los resultados a un DataFrame para análisis
df_results = pd.DataFrame({
    'Error_Reconstruccion': mse_reconstruction,
    'Clase_Real': y_test
})

print("\nEstadísticas del Error de Reconstrucción:")
print("  Clase NORMAL (0):")
print(f"    Media: {df_results[df_results['Clase_Real']==0]['Error_Reconstruccion'].mean():.4f}")
print(f"    Max:   {df_results[df_results['Clase_Real']==0]['Error_Reconstruccion'].max():.4f}")
print("  Clase ANOMALÍA (1):")
print(f"    Media: {df_results[df_results['Clase_Real']==1]['Error_Reconstruccion'].mean():.4f}")
print(f"    Min:   {df_results[df_results['Clase_Real']==1]['Error_Reconstruccion'].min():.4f}")

# Definimos el umbral (Threshold) para clasificar como anomalía.
# Usamos el percentil 95 del error en los datos normales de entrenamiento para fijarlo.
X_train_reconstructed = autoencoder.predict(X_train_scaled)
mse_train = np.mean(np.power(X_train_scaled - X_train_reconstructed, 2), axis=1)
umbral_anomalia = np.percentile(mse_train, 98) # Permitimos un 2% de falsos positivos en entreno

print(f"\nUmbral de Detección Automático fijado en: {umbral_anomalia:.4f}")

df_results['Anomalia_Predicha'] = (df_results['Error_Reconstruccion'] > umbral_anomalia).astype(int)

# ============================================================================
# 5. VISUALIZACIÓN: DISTRIBUCIÓN DEL ERROR
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DE RESULTADOS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Histograma de Errores de Reconstrucción
sns.histplot(data=df_results, x='Error_Reconstruccion', hue='Clase_Real', 
             bins=50, kde=True, palette={0: 'blue', 1: 'red'}, alpha=0.6, ax=axes[0])
axes[0].axvline(umbral_anomalia, color='black', linestyle='--', linewidth=2, 
                label=f'Umbral Detección ({umbral_anomalia:.2f})')
axes[0].set_title('Distribución del Error de Reconstrucción\nAutoencoder', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Error Cuadrático Medio (MSE)')
axes[0].set_ylabel('Frecuencia')
axes[0].legend(labels=['Umbral', 'Anomalía (1)', 'Normal (0)'])
axes[0].set_xlim(0, df_results['Error_Reconstruccion'].quantile(0.95) * 2) # Acotar para mejor vista

# Gráfico 2: Curva ROC (Evaluar capacidad general del modelo sin depender del umbral)
fpr, tpr, thresholds = roc_curve(df_results['Clase_Real'], df_results['Error_Reconstruccion'])
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
axes[1].set_title('Curva ROC del Autoencoder', fontsize=13, fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/usuario/01_autoencoder_errores.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_autoencoder_errores.png")
plt.close()

# ============================================================================
# 6. EVALUACIÓN Y RENDIMIENTO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: EVALUACIÓN DEL RENDIMIENTO")
print("=" * 80)

print("\nMatriz de Confusión (Datos de Prueba):")
cm = confusion_matrix(df_results['Clase_Real'], df_results['Anomalia_Predicha'])
print(pd.DataFrame(cm, 
                   index=['Real_Normal (0)', 'Real_Anomalía (1)'], 
                   columns=['Pred_Normal (0)', 'Pred_Anomalía (1)']))

print("\nReporte de Clasificación:")
print(classification_report(df_results['Clase_Real'], df_results['Anomalia_Predicha'], 
                            target_names=['Normal (0)', 'Anomalía (1)']))

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              AUTOENCODERS PARA DETECCIÓN DE ANOMALÍAS COMPLEJAS              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. INTUICIÓN DEL MODELO:
   • El Autoencoder se vio forzado a comprimir 30 variables complejas a solo 8.
   • Durante el entrenamiento SOLO observó células benignas (Normales). Aprendió
     perfectamente cómo se relacionan los valores normales para reconstruirlos.

2. MECANISMO DE DETECCIÓN:
   • Al pasar una célula maligna (Anomalía) por la red, las correlaciones entre sus 
     variables eran diferentes.
   • El decoder intentó reconstruirla asumiendo que era una célula normal, fallando 
     catastróficamente.
   • Este fallo se mide mediante el Error Cuadrático Medio (MSE). Un MSE alto 
     significa: "No reconozco este patrón", disparando la alerta de anomalía.

3. DESEMPEÑO DEL MODELO:
   • AUC-ROC: {roc_auc:.4f} (Excelente capacidad de separación).
   • Verdaderas Anomalías Detectadas: {cm[1, 1]} de {cm[1, 0] + cm[1, 1]}.
   • La ventaja sobre Isolation Forest o One-Class SVM es que las redes neuronales 
     (Autoencoders) pueden capturar interacciones no lineales mucho más profundas 
     y complejas en alta dimensionalidad.

4. CONSIDERACIONES TÉCNICAS (USO DE SCIKIT-LEARN):
   • Se implementó de manera nativa utilizando `MLPRegressor` fijando la entrada `X` 
     como su propio objetivo `y` (`fit(X, X)`).
   • Para implementaciones en producción a gran escala o con imágenes/audio, se 
     recomienda migrar esta misma arquitectura a frameworks como Keras/TensorFlow 
     o PyTorch.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation='relu'):
  - Arquitectura tipo "reloj de arena" indispensable en los autoencoders.
  - El cuello de botella de tamaño 8 actúa como filtro de características.
  - activation='relu': Permite mapear relaciones no lineales entre variables.
  - early_stopping=True: Detiene el entrenamiento cuando la pérdida se estabiliza.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_autoencoder_errores.png - Histograma bimodal del error de reconstrucción y
                             evaluación métrica mediante Curva ROC.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de anomalías complejas mediante Autoencoders completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")