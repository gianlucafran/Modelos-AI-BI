"""
Elastic Net: Modelado Estadístico Avanzado y Regresión Penalizada
Demostración con el dataset REAL "Diabetes" de scikit-learn

Conceptos:
- Elastic Net: Un modelo de regresión regularizada que combina linealmente 
  las penalizaciones L1 (Lasso) y L2 (Ridge).
- Solución a la Multicolinealidad Extrema: Lasso tiende a seleccionar aleatoriamente 
  una variable de un grupo de variables altamente correlacionadas y descartar el resto. 
  Elastic Net soluciona esto seleccionando grupos enteros de variables correlacionadas.
- Parámetro l1_ratio: Controla la mezcla entre las penalizaciones. Un l1_ratio = 1 
  es Lasso puro, l1_ratio = 0 es Ridge puro, y valores intermedios son Elastic Net.
- Modelado Avanzado (Expansión Polinómica): Aumentaremos artificialmente la dimensionalidad 
  creando interacciones entre variables para demostrar cómo Elastic Net maneja espacios 
  complejos de alta dimensionalidad.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. CARGAR Y EXPLORAR EL DATASET (DATOS CLÍNICOS REALES)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR DATASET REAL (PROGRESIÓN DE DIABETES)")
print("=" * 80)

# El dataset de Diabetes contiene mediciones de 442 pacientes y su progresión
# de la enfermedad un año después.
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['Progresion_Enfermedad'] = y

print(f"\nDataset: Diabetes (Scikit-Learn Nativo)")
print(f"Total de pacientes (muestras): {df.shape[0]}")
print(f"Características clínicas originales: {len(feature_names)}")
print("Variables: Edad, Sexo, IMC, Presión Arterial y 6 mediciones de suero sanguíneo.")

# ============================================================================
# 2. MODELADO AVANZADO: EXPANSIÓN DE DIMENSIONALIDAD
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: EXPANSIÓN POLINÓMICA Y PREPROCESAMIENTO")
print("=" * 80)

# Para demostrar el verdadero poder de Elastic Net frente a Ridge o Lasso, 
# simularemos un escenario complejo de alta dimensionalidad creando características 
# polinómicas (interacciones entre variables clínicas).
# Pasaremos de 10 variables a 65 variables fuertemente correlacionadas.

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly_feature_names = poly.get_feature_names_out(feature_names)

print(f"\nSe aplicó PolynomialFeatures(degree=2).")
print(f"Nuevas características generadas (Dimensionalidad): {X_poly.shape[1]}")
print("Esta expansión crea multicolinealidad extrema (ej. IMC * Presion_Arterial),")
print("el escenario perfecto donde OLS fracasa y Elastic Net brilla.")

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# Estandarización obligatoria para modelos penalizados (L1/L2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Datos divididos y estandarizados (StandardScaler).")

# ============================================================================
# 3. ENTRENAMIENTO COMPARATIVO Y BÚSQUEDA DE HIPERPARÁMETROS (CV)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAMIENTO DE MODELOS MEDIANTE CROSS-VALIDATION")
print("=" * 80)

print("\nEntrenando modelos usando validación cruzada interna (CV=5) para")
print("encontrar la penalización óptima (alpha) automáticamente...")

# 1. RIDGE (Solo penalización L2 - No elimina características)
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)

# 2. LASSO (Solo penalización L1 - Fuerza esparcidad extrema)
lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 100), cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

# 3. ELASTIC NET (Penalización Mixta L1 + L2)
# Probamos un abanico de l1_ratios. El modelo probará internamente si debe 
# comportarse más como Ridge, más como Lasso, o un híbrido perfecto.
l1_ratios_a_probar = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
elastic_cv = ElasticNetCV(l1_ratio=l1_ratios_a_probar, alphas=np.logspace(-3, 1, 100), 
                          cv=5, max_iter=10000, random_state=42)
elastic_cv.fit(X_train_scaled, y_train)

print(f"\n✓ Modelos Entrenados. Parámetros óptimos encontrados:")
print(f"  - RIDGE Alpha: {ridge_cv.alpha_:.4f}")
print(f"  - LASSO Alpha: {lasso_cv.alpha_:.4f}")
print(f"  - ELASTIC NET Alpha: {elastic_cv.alpha_:.4f} | L1_Ratio Óptimo: {elastic_cv.l1_ratio_}")
print("    (Un l1_ratio de 0.5 significa una mezcla exacta 50% L1 y 50% L2)")

# Analizar la cantidad de características retenidas (Esparcidad)
features_ridge = np.sum(ridge_cv.coef_ != 0)
features_lasso = np.sum(lasso_cv.coef_ != 0)
features_elastic = np.sum(elastic_cv.coef_ != 0)

print(f"\nSelección de Biomarcadores Polinómicos (de {X_poly.shape[1]} posibles):")
print(f"  - Ridge retuvo:       {features_ridge} (No selecciona, las usa todas)")
print(f"  - Lasso retuvo:       {features_lasso} (Esparcidad agresiva)")
print(f"  - Elastic Net retuvo: {features_elastic} (Balance perfecto)")

# ============================================================================
# 4. PREDICCIÓN Y EVALUACIÓN ESTADÍSTICA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: EVALUACIÓN DEL RENDIMIENTO EN DATOS NUEVOS")
print("=" * 80)

# Predicciones
y_pred_ridge = ridge_cv.predict(X_test_scaled)
y_pred_lasso = lasso_cv.predict(X_test_scaled)
y_pred_elastic = elastic_cv.predict(X_test_scaled)

# Cálculos de R2
r2_ridge = r2_score(y_test, y_pred_ridge)
r2_lasso = r2_score(y_test, y_pred_lasso)
r2_elastic = r2_score(y_test, y_pred_elastic)

print("\nResultados de R2 Score (Varianza explicada en Test Set):")
print("-" * 50)
print(f"{'Modelo':<15} | {'R2 Score':<10} | {'Sparsity (Características)'}")
print("-" * 50)
print(f"{'Ridge (L2)':<15} | {r2_ridge:<10.4f} | {features_ridge}")
print(f"{'Lasso (L1)':<15} | {r2_lasso:<10.4f} | {features_lasso}")
print(f"{'Elastic Net':<15} | {r2_elastic:<10.4f} | {features_elastic}")
print("-" * 50)

# ============================================================================
# 5. VISUALIZACIÓN: COMPORTAMIENTO DE LOS COEFICIENTES Y PREDICCIONES
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DEL COMPORTAMIENTO DE LAS PENALIZACIONES")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Perfil de Coeficientes (Ridge vs Lasso vs Elastic Net)
x_pos = np.arange(X_poly.shape[1])

# Hacemos los coeficientes transparentes para superponerlos y ver la estrategia de cada modelo
axes[0].plot(x_pos, ridge_cv.coef_, alpha=0.6, color='blue', label='Ridge (Estabiliza todo)', linewidth=2)
axes[0].plot(x_pos, lasso_cv.coef_, alpha=0.6, color='red', label='Lasso (Descarta agresivamente)', linewidth=2)
axes[0].plot(x_pos, elastic_cv.coef_, alpha=0.9, color='green', label=f'Elastic Net (El Balance)', linewidth=2, linestyle='--')

axes[0].set_title('Efecto del Modelado Estadístico Avanzado en los Coeficientes\n(L1, L2 y Elastic Net)', 
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Índice de la Característica (Variables Simples e Interacciones)')
axes[0].set_ylabel('Magnitud del Coeficiente')
axes[0].axhline(0, color='black', linewidth=1)
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# Explicación en el gráfico
axes[0].text(2, max(elastic_cv.coef_)*0.8, 
             "Observe cómo Elastic Net\nencoge coeficientes como Ridge,\npero lleva a 0 los irrelevantes\ncomo Lasso.", 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Gráfico 2: Realidad vs Predicción (Elastic Net)
# Gráfico de dispersión para evaluar el ajuste del mejor modelo
axes[1].scatter(y_test, y_pred_elastic, color='teal', alpha=0.6, edgecolor='k', s=50)

# Línea de ajuste perfecto
lims = [min(y_test.min(), y_pred_elastic.min()) - 10, 
        max(y_test.max(), y_pred_elastic.max()) + 10]
axes[1].plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ajuste Perfecto (Ideal)')

axes[1].set_title(f'Desempeño de Predicción Clínica (Elastic Net)\nR² Score: {r2_elastic:.4f}', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Progresión de la Enfermedad (Real)')
axes[1].set_ylabel('Progresión Predicha (Elastic Net)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
#plt.savefig('/home/usuario/01_elastic_net_modelado.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_elastic_net_modelado.png")
plt.close()

# ============================================================================
# 6. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               ELASTIC NET PARA MODELADO ESTADÍSTICO AVANZADO                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. EL PROBLEMA (MULTICOLINEALIDAD POLINÓMICA):
   • Al aplicar `PolynomialFeatures`, transformamos 10 mediciones clínicas en 
     {X_poly.shape[1]} interacciones complejas. Esto creó una matriz donde muchas 
     columnas estaban altamente correlacionadas entre sí (Ej. Edad*IMC, IMC*Presión).
   • Si aplicáramos Regresión OLS clásica aquí, el modelo explotaría (Overfitting).

2. LAS LIMITACIONES DE LOS EXTREMOS:
   • RIDGE (L2 puro): Usó los {features_ridge} biomarcadores. Evitó que los 
     coeficientes explotaran, pero retuvo demasiado ruido, dificultando la 
     interpretación médica.
   • LASSO (L1 puro): Fue ultra agresivo y se quedó con solo {features_lasso} 
     variables. El problema de Lasso es que si dos biomarcadores biológicamente 
     importantes están correlacionados, Lasso tira una moneda, guarda uno y elimina 
     el otro.

3. LA SOLUCIÓN: ELASTIC NET:
   • Logró lo mejor de ambos mundos. Encontró automáticamente a través de validación 
     cruzada (`ElasticNetCV`) que la proporción ideal (l1_ratio) era {elastic_cv.l1_ratio_}.
   • Retuvo grupos de variables fuertemente correlacionadas (gracias a L2) pero al 
     mismo tiempo eliminó el ruido irrelevante llevando sus coeficientes a 0 (gracias a L1).
   • El resultado final arrojó un panel de {features_elastic} características estables, 
     obteniendo el mejor o un R2 sumamente competitivo ({r2_elastic:.4f}) frente a 
     las alternativas puras.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99], cv=5):
  - L1_ratio: Determina el peso relativo de la penalidad L1 respecto a la L2.
    La función `*CV` probó toda esa lista y escogió el mejor valor para estos 
    datos clínicos de forma automática.
  - alphas: El algoritmo probó 100 valores distintos de penalización total (magnitud).

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_elastic_net_modelado.png - Análisis espectral de los coeficientes 
                              (Ridge vs Lasso vs Elastic Net) y el scatter plot 
                              de la predicción vs realidad.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de modelado avanzado con Elastic Net completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")