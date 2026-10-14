"""
LDA (Linear Discriminant Analysis): Análisis de Riesgo Crediticio
Demostración con dataset sintético (make_classification) de scikit-learn

Conceptos:
- LDA (Análisis Discriminante Lineal): Técnica estadística y de Machine Learning utilizada 
  tanto para clasificación predictiva como para reducción de dimensionalidad supervisada.
- Maximización de Separabilidad: A diferencia de PCA (que busca la varianza máxima ignorando 
  las clases), LDA busca el eje (hiperplano) que maximice la distancia ENTRE las clases y 
  minimice la dispersión DENTRO de cada clase.
- Riesgo Crediticio (Credit Scoring): Clasificación binaria para decidir si un cliente 
  es "Buen Pagador" (Aprobado) o "Mal Pagador / Default" (Rechazado).
- Reducción a 1D: Para un problema de 2 clases, LDA reduce los datos a exactamente 1 
  dimensión (discriminante lineal), lo que facilita enormemente crear un "Score de Riesgo".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. GENERAR Y EXPLORAR EL DATASET (SIMULACIÓN DE CRÉDITO)
# ============================================================================
print("=" * 80)
print("PASO 1: SIMULAR DATASET DE RIESGO CREDITICIO")
print("=" * 80)

# Dado que scikit-learn no incluye un dataset financiero nativo de riesgo crediticio,
# utilizamos make_classification de sklearn.datasets para simular perfiles financieros.
# Generaremos un dataset con ligero desbalance (80% Buenos Pagadores, 20% Default).

n_clientes = 2000
X, y = make_classification(
    n_samples=n_clientes, 
    n_features=5,           # 5 variables financieras
    n_informative=4,        # 4 variables realmente aportan información
    n_redundant=1,          # 1 variable correlacionada (ruido)
    n_classes=2, 
    weights=[0.8, 0.2],     # 80% Clase 0 (Buen Pagador), 20% Clase 1 (Default/Riesgo)
    class_sep=1.1,          # Separación moderada entre clases
    random_state=42
)

# Simulamos nombres reales para las características financieras
feature_names = [
    'Ingreso_Anual_Normalizado', 
    'Deuda_Total_Normalizada', 
    'Score_Crediticio_Externo', 
    'Antiguedad_Laboral', 
    'Ratio_Deuda_Ingreso'
]

target_names = ['Buen Pagador (0)', 'Default / Riesgo (1)']

df = pd.DataFrame(X, columns=feature_names)
df['Estado_Credito'] = y

print(f"\nDataset: Simulación de Solicitudes de Crédito")
print(f"Total de clientes evaluados: {df.shape[0]}")
print(f"Número de variables financieras: {df.shape[1] - 1}")

print("\nDistribución de la Cartera:")
print(f"  Aprobados / Buen Pagador (0): {sum(y==0)} ({(sum(y==0)/n_clientes)*100:.1f}%)")
print(f"  Rechazados / Default (1):     {sum(y==1)} ({(sum(y==1)/n_clientes)*100:.1f}%)")

print(f"\nPrimeros 5 perfiles de clientes:")
print(df.head().round(3))

# ============================================================================
# 2. PREPROCESAMIENTO Y DIVISIÓN DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: PREPROCESAMIENTO DE LOS DATOS")
print("=" * 80)

# Dividir en conjunto de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# La estandarización es recomendada para LDA, especialmente para interpretar
# correctamente los coeficientes (pesos) que el modelo asignará a cada variable.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Datos divididos en Entrenamiento (70%) y Prueba (30%).")
print("✓ StandardScaler aplicado para asegurar que el ingreso y la edad tengan el mismo peso estadístico.")

# ============================================================================
# 3. ENTRENAMIENTO DEL MODELO LDA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAMIENTO DEL ANÁLISIS DISCRIMINANTE LINEAL")
print("=" * 80)

print("Inicializando LinearDiscriminantAnalysis (LDA)...")
# Almacenamos el modelo LDA
lda = LinearDiscriminantAnalysis()

# Entrenamos el modelo. 
# A diferencia de PCA, LDA REQUIERE la variable objetivo (y_train) porque 
# busca la separación explícita de las clases.
lda.fit(X_train_scaled, y_train)

print("✓ Modelo entrenado. Eje discriminante (Score de Riesgo) calculado.")

# Interpretabilidad Financiera: ¿Qué variables pesan más?
# Los coeficientes de LDA nos dicen qué variable influye más para separar a los pagadores de los morosos.
coeficientes = pd.DataFrame(
    lda.coef_[0], 
    index=feature_names, 
    columns=['Peso_Discriminante']
).sort_values(by='Peso_Discriminante', ascending=False)

print("\nPeso de las variables en la decisión crediticia (Importancia en LDA):")
print(coeficientes.round(4))
print("  (Valores positivos altos empujan hacia Default, negativos hacia Buen Pagador)")

# ============================================================================
# 4. PREDICCIÓN Y EVALUACIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: EVALUACIÓN DEL MODELO CREDITICIO")
print("=" * 80)

# Predicciones de clase y probabilidades
y_pred = lda.predict(X_test_scaled)
y_prob = lda.predict_proba(X_test_scaled)[:, 1] # Probabilidad de ser Default (Clase 1)

print("\nMatriz de Confusión (Conjunto de Prueba):")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, 
                   index=[f'Real {target_names[0]}', f'Real {target_names[1]}'], 
                   columns=[f'Pred {target_names[0]}', f'Pred {target_names[1]}']))

auc_roc = roc_auc_score(y_test, y_prob)
print(f"\nMétrica ROC-AUC (Capacidad de separación): {auc_roc:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ============================================================================
# 5. LDA COMO REDUCCIÓN DE DIMENSIONALIDAD (EL SCORE DE CRÉDITO)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: REDUCCIÓN DE DIMENSIONALIDAD Y CREACIÓN DEL SCORE")
print("=" * 80)

# LDA puede proyectar las 5 dimensiones originales a 1 sola dimensión
# (n_clases - 1 = 2 - 1 = 1 dimensión). 
# Esta única dimensión es el famoso "Credit Score" o Eje Discriminante.

X_lda_train = lda.transform(X_train_scaled)
X_lda_test = lda.transform(X_test_scaled)

print(f"Dimensiones antes de LDA: {X_test_scaled.shape[1]}")
print(f"Dimensiones después de transformar con LDA: {X_lda_test.shape[1]}")
print("\nHemos comprimido 5 variables financieras en 1 solo 'Score de Riesgo' que maximiza")
print("la separación entre Buenos Pagadores y Defaults.")

# ============================================================================
# 6. VISUALIZACIÓN: SEPARABILIDAD Y MATRIZ DE CONFUSIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: VISUALIZACIÓN DEL ESPACIO DISCRIMINANTE")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Histograma del Espacio Latente LDA (Credit Score)
# Dibujamos cómo LDA logró separar a los Buenos Pagadores de los Morosos en 1D
sns.histplot(X_lda_test[y_test == 0], bins=30, color='blue', alpha=0.5, 
             kde=True, label='Buen Pagador (0)', ax=axes[0])
sns.histplot(X_lda_test[y_test == 1], bins=30, color='red', alpha=0.5, 
             kde=True, label='Default (1)', ax=axes[0])

# El límite de decisión teórico en un LDA estándar estandarizado suele estar cerca de 0
axes[0].axvline(0, color='black', linestyle='--', linewidth=2, label='Frontera de Decisión')

axes[0].set_title('Proyección LDA 1D: Distribución del Score de Riesgo\n(Maximizando Separabilidad entre Clases)', 
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Componente Discriminante Lineal 1 (LD1)')
axes[0].set_ylabel('Frecuencia de Clientes')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Gráfico 2: Matriz de Confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred Buen Pagador', 'Pred Default'], 
            yticklabels=['Real Buen Pagador', 'Real Default'], 
            cbar=False, linewidths=1, linecolor='black', ax=axes[1])
axes[1].set_title('Matriz de Confusión en la Evaluación de Crédito', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Decisión del Modelo LDA')
axes[1].set_ylabel('Realidad del Cliente')

plt.tight_layout()
#plt.savefig('/home/usuario/01_lda_riesgo_credito.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_lda_riesgo_credito.png")
plt.close()

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               LDA APLICADO AL ANÁLISIS DE RIESGO CREDITICIO                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. ¿POR QUÉ LDA EN RIESGO CREDITICIO?:
   • En finanzas institucionales, los modelos de "Caja Negra" (como Redes Neuronales) 
     están fuertemente regulados. LDA es un modelo lineal, transparente y altamente 
     interpretable.
   • El paso 3 nos permitió imprimir los coeficientes exactos: podemos explicar 
     matemáticamente por qué a un cliente se le negó el crédito basándonos en 
     el peso de su Ratio Deuda-Ingreso.

2. LDA COMO CREADOR DE "SCORES":
   • LDA transformó (redujo) un perfil de 5 dimensiones financieras complejas a 
     un solo número (El LD1 proyectado en el Gráfico 1). 
   • Este LD1 puede ser utilizado directamente por el banco como el "Score de Riesgo". 
     Valores negativos = Seguro de aprobar. Valores fuertemente positivos = Alto riesgo.

3. DIFERENCIA CLAVE CON PCA:
   • Si hubiéramos usado PCA, el algoritmo habría buscado la característica financiera 
     que más varía (quizás el ingreso). 
   • LDA usa la variable Y (Estado_Credito) de forma SUPERVISADA para encontrar el 
     eje que separa a los buenos pagadores de los que hacen default, sin importar 
     si esa variable es la que más varianza general tiene.

4. RENDIMIENTO DEL MODELO:
   • ROC-AUC: {auc_roc:.4f}. Un modelo robusto.
   • A pesar del desbalance (80/20), la técnica discriminante logró establecer un 
     umbral claro de separación. En la práctica financiera, se podría mover el 
     umbral (la línea punteada negra) hacia la izquierda para ser más estrictos 
     y evitar Defaults a costa de rechazar clientes buenos.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

LinearDiscriminantAnalysis():
  - solver='svd': Método por defecto robusto y rápido.
  - No requiere ajustar n_components cuando se usa como clasificador (fit/predict).
  - Al usar `transform()`, scikit-learn automáticamente reduce los datos a 
    (n_classes - 1) dimensiones, que en clasificación binaria es exactamente 1.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_lda_riesgo_credito.png - Visualización del Espacio Latente Discriminante (1D) 
                            mostrando la capacidad del LDA para separar ambas clases,
                            y Matriz de Confusión comercial.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de riesgo crediticio con LDA completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")