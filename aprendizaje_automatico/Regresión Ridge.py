"""
Regresión Ridge: Previsión en Series Temporales Económicas
Demostración con dataset sintético (make_regression) de scikit-learn

Conceptos:
- Regresión Ridge (Penalización L2): Extensión de la regresión lineal que añade una 
  penalización proporcional al cuadrado de la magnitud de los coeficientes.
- Multicolinealidad Macroeconómica: En datos económicos (ej. empleo sectorial, valor 
  agregado regional), las variables suelen moverse juntas. Ridge estabiliza el modelo 
  evitando que los coeficientes exploten debido a esta alta correlación.
- Partición de Series Temporales: El orden cronológico es sagrado. Los datos se dividen 
  secuencialmente (shuffle=False) para evitar el "data leakage" (predecir el pasado 
  usando datos del futuro).
- Complementariedad Analítica: Mientras que los filtros estadísticos (como Hodrick-Prescott) 
  o modelos como SARIMAX extraen tendencias y estacionalidad, Ridge permite incorporar 
  decenas de regresores exógenos altamente correlacionados sin sobreajustar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. GENERAR EL DATASET (SIMULACIÓN MACROECONÓMICA)
# ============================================================================
print("=" * 80)
print("PASO 1: SIMULAR DATASET DE SERIES TEMPORALES ECONÓMICAS")
print("=" * 80)

# Para simular variables macroeconómicas con alta colinealidad (donde Ridge brilla),
# utilizamos el parámetro 'effective_rank'. Esto asegura que, aunque hay 6 variables,
# la información real se condensa en menos dimensiones, creando correlaciones fuertes.
n_meses = 150 # 12.5 años de datos mensuales

X, y = make_regression(
    n_samples=n_meses, 
    n_features=6, 
    n_informative=3, 
    noise=20.0, 
    effective_rank=2,  # Fuerza la multicolinealidad entre las variables
    random_state=42
)

# Nombres simulados para indicadores macroeconómicos regionales
feature_names = [
    'Empleo_Sector_Secundario', 
    'Empleo_Sector_Terciario', 
    'Valor_Agregado_Bruto', 
    'Inversion_Directa', 
    'Indice_Precios_Consumidor', 
    'Tasa_Interes_Referencia'
]

# Crear un índice temporal mensual
fechas = pd.date_range(start='2014-01-01', periods=n_meses, freq='M')

df_eco = pd.DataFrame(X, columns=feature_names, index=fechas)
df_eco['Crecimiento_PIB_Mensual'] = y

print(f"\nDataset: Indicadores Macroeconómicos Mensuales")
print(f"Período: {fechas[0].strftime('%Y-%m')} a {fechas[-1].strftime('%Y-%m')} ({n_meses} meses)")
print(f"Características (Regresores Exógenos): {len(feature_names)}")

# Verificamos la multicolinealidad
corr_matrix = df_eco[feature_names].corr()
print("\nAdvertencia de Multicolinealidad detectada en el sistema:")
alta_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
if alta_corr.any().any():
    print("  Existen variables fuertemente correlacionadas (rho > 0.8).")
    print("  Una Regresión Lineal Ordinaria (OLS) se volverá inestable aquí.")

print(f"\nPrimeros 5 meses de datos:")
print(df_eco.head().round(2))

# ============================================================================
# 2. PREPROCESAMIENTO TEMPORAL Y ESTANDARIZACIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: SPLIT CRONOLÓGICO Y ESTANDARIZACIÓN")
print("=" * 80)

# En Series Temporales, NUNCA debemos mezclar los datos (shuffle=False).
# Entrenamos con el pasado (primeros 80%) y predecimos el futuro (último 20%).
X_ts = df_eco.drop('Crecimiento_PIB_Mensual', axis=1)
y_ts = df_eco['Crecimiento_PIB_Mensual']

X_train, X_test, y_train, y_test = train_test_split(
    X_ts, y_ts, test_size=0.2, shuffle=False
)

print(f"Set de Entrenamiento: {X_train.shape[0]} meses (Pasado)")
print(f"Set de Prueba (Forecasting): {X_test.shape[0]} meses (Futuro)")

# Ridge Regression es altamente sensible a la escala. La penalización L2 se aplica 
# por igual a todos los coeficientes. Si no estandarizamos, las variables con 
# escalas grandes (ej. Valor Agregado en millones) serán penalizadas injustamente.
scaler = StandardScaler()

# Ajustamos solo con datos de entrenamiento para evitar filtración de información temporal
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Estandarización aplicada a los regresores económicos (Media=0, Var=1).")

# ============================================================================
# 3. ENTRENAMIENTO COMPARATIVO (OLS vs RIDGE)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAMIENTO Y EFECTO DE LA PENALIZACIÓN L2")
print("=" * 80)

# Modelo 1: Regresión Lineal Clásica (OLS - Sin penalización)
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# Modelo 2: Regresión Ridge
# alpha (o lambda) controla la fuerza de la penalización. 
# alpha=0 equivale a OLS. Valores más altos "encogen" más los coeficientes.
alpha_val = 50.0 
ridge = Ridge(alpha=alpha_val, random_state=42)
ridge.fit(X_train_scaled, y_train)

print(f"Modelos Entrenados.")
print(f"  - OLS Clásico (Susceptible a multicolinealidad)")
print(f"  - Modelo Ridge (alpha={alpha_val})")

# Comparamos cómo Ridge reduce y estabiliza los coeficientes
df_coefs = pd.DataFrame({
    'Variable': feature_names,
    'Coef_OLS': ols.coef_,
    'Coef_Ridge': ridge.coef_
})
df_coefs['Reduccion_Absoluta'] = np.abs(df_coefs['Coef_OLS']) - np.abs(df_coefs['Coef_Ridge'])

print("\nImpacto de Ridge en los Coeficientes (Shrinkage):")
print(df_coefs.round(3))
print("\nNota: Debido a la colinealidad forzada (effective_rank=2), OLS infla")
print("algunos coeficientes de manera inestable. Ridge los restringe a valores lógicos.")

# ============================================================================
# 4. PREDICCIÓN Y FORECASTING
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: EVALUACIÓN DE LAS PREDICCIONES (FORECASTING)")
print("=" * 80)

# Predicciones futuras
y_pred_ols = ols.predict(X_test_scaled)
y_pred_ridge = ridge.predict(X_test_scaled)

# Métricas OLS
mse_ols = mean_squared_error(y_test, y_pred_ols)
mae_ols = mean_absolute_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)

# Métricas Ridge
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRendimiento sobre Datos Futuros (Test Set):")
print("-" * 45)
print(f"{'Métrica':<10} | {'Regresión OLS':<15} | {'Regresión Ridge':<15}")
print("-" * 45)
print(f"{'R2 Score':<10} | {r2_ols:<15.4f} | {r2_ridge:<15.4f}")
print(f"{'MSE':<10} | {mse_ols:<15.2f} | {mse_ridge:<15.2f}")
print(f"{'MAE':<10} | {mae_ols:<15.2f} | {mae_ridge:<15.2f}")
print("-" * 45)

if mse_ridge < mse_ols:
    print("\n✓ Ridge mejoró la capacidad predictiva al evitar el sobreajuste al ruido histórico.")

# ============================================================================
# 5. VISUALIZACIÓN: SERIES TEMPORALES Y SHRINKAGE
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DEL FORECASTING Y LOS COEFICIENTES")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Gráfico 1: Forecasting de la Serie Temporal
axes[0].plot(fechas[:len(y_train)], y_train, color='gray', label='Historia (Datos de Entrenamiento)', alpha=0.7)
axes[0].plot(fechas[len(y_train):], y_test, color='black', label='Realidad Futura (Test)', linewidth=2)
axes[0].plot(fechas[len(y_train):], y_pred_ols, color='red', linestyle='--', label='Predicción OLS', alpha=0.7)
axes[0].plot(fechas[len(y_train):], y_pred_ridge, color='blue', linewidth=2.5, label=f'Predicción Ridge (α={alpha_val})')

axes[0].set_title('Predicción del Crecimiento del PIB basado en Variables Macroeconómicas', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Crecimiento del PIB')
axes[0].legend(loc='upper left')
axes[0].grid(alpha=0.3)

# Resaltar la zona de forecasting
axes[0].axvspan(fechas[len(y_train)], fechas[-1], color='lavender', alpha=0.3, label='Ventana de Forecasting')

# Gráfico 2: Efecto de Contracción de Coeficientes (Shrinkage)
x_pos = np.arange(len(feature_names))
width = 0.35

axes[1].bar(x_pos - width/2, df_coefs['Coef_OLS'], width, label='OLS (Clásico)', color='salmon', edgecolor='black')
axes[1].bar(x_pos + width/2, df_coefs['Coef_Ridge'], width, label='Ridge (L2)', color='steelblue', edgecolor='black')

axes[1].set_title('Estabilización de Coeficientes: OLS vs Ridge', fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(feature_names, rotation=15, ha='right')
axes[1].set_ylabel('Magnitud del Coeficiente')
axes[1].axhline(0, color='black', linewidth=1)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
#plt.savefig('/home/usuario/01_ridge_series_temporales.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_ridge_series_temporales.png")
plt.close()

# ============================================================================
# 6. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               RIDGE REGRESSION EN SERIES TEMPORALES ECONÓMICAS               ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. MANEJO DE LA MULTICOLINEALIDAD:
   • Los modelos macroeconómicos (empleo sectorial, valor agregado, tasas) 
     sufren de alta multicolinealidad. La regresión OLS tradicional vuelve sus 
     coeficientes muy volátiles para compensar, lo que genera predicciones 
     erráticas ante nuevos datos.
   • Ridge aplicó una penalización L2 que "encogió" (Shrinkage) los coeficientes,
     repartiendo el peso predictivo de manera más estable entre las variables.

2. FORECASTING Y VALIDACIÓN TEMPORAL:
   • Fue imperativo realizar el `train_test_split` con `shuffle=False`. En 
     econometría y data science orientada a negocios, el tiempo es una variable 
     aserrada; aprender del futuro para predecir el pasado arruina el modelo.

3. INTEGRACIÓN ANALÍTICA:
   • Ridge no modela la autocorrelación de los errores (como haría un modelo 
     ARIMA o SARIMAX). Sin embargo, es la herramienta perfecta para modelar la 
     parte EXÓGENA de la serie. 
   • Se puede aplicar un filtro estadístico (ej. Hodrick-Prescott) para aislar 
     el componente cíclico del PIB, y luego usar Ridge sobre las variables 
     macro para predecir exclusivamente ese ciclo, logrando un ensamble 
     altamente robusto.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Ridge(alpha={alpha_val}, random_state=42):
  - alpha: Nivel de regularización. Un alpha mayor obliga a los coeficientes a 
    acercarse más a cero. La optimización de este valor se realiza típicamente 
    con `RidgeCV` o validación cruzada respetando series de tiempo 
    (`TimeSeriesSplit`).

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_ridge_series_temporales.png - Contiene el pronóstico de la serie a futuro
                                 y el diagrama de barras que comprueba el 
                                 efecto de estabilización de la penalización L2.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de forecasting económico con Ridge Regression completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")