"""
Publicidad Digital y AdTech: Modelo Nano / Micro-Especializado para Predicción de CTR y Conversiones
Aplicación: Predicción de Clics/Conversión y Optimización de Ofertas (Real-Time Bidding - RTB)

Este script utiliza un dataset de scikit-learn configurado para reflejar el comportamiento
de usuarios en campañas de publicidad digital (impresiones, tasa de clics, segmentación geográfica y demográfica).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix

# Estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. GENERACIÓN / CARGA DEL DATASET DE PUBLICIDAD DIGITAL (SCIKIT-LEARN)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET DE PUBLICIDAD DIGITAL (Predecir Conversión / CTR)")
print("=" * 80)

# Generamos un dataset sintético altamente representativo de datos de publicidad (impresiones vs conversiones)
# 10,000 impresiones publicitarias con características como: Edad, Horario, Dispositivo, Historial de Clics
X_raw, y_raw = make_classification(
    n_samples=12000,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    weights=[0.88, 0.12], # Tasa de conversión real (12% CTR/CR - típico en AdTech)
    random_state=42
)

nombres_caracteristicas = [
    'edad_usuario', 'frecuencia_impresion', 'hora_del_dia', 
    'categoria_anuncio', 'dispositivo_movil', 'historial_ctr_usuario',
    'score_interes_marca', 'tiempo_en_sitio_previo', 'tamano_banner', 'geolocalizacion_code'
]

df_ads = pd.DataFrame(X_raw, columns=nombres_caracteristicas)
df_ads['conversion'] = y_raw

print(f"✓ Total de impresiones analizadas: {len(df_ads):,} filas.")
print(f"✓ Tasa global de conversión/clic (CTR): {(df_ads['conversion'].mean() * 100):.2f}%")
print(f"✓ Variables publicitarias disponibles: {len(nombres_caracteristicas)}")

# ============================================================================
# 2. PREPARACIÓN DE DATOS Y DIVISIÓN TRAIN / TEST
# ============================================================================
X = df_ads.drop(columns=['conversion'])
y = df_ads['conversion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. ENTRENAMIENTO DEL MODELO DE LATENCIA ULTRABAJA (NANO-ADS PREDICTOR)
# ============================================================================
print("\n" + "=" * 80)
print("2. ENTRENANDO EL MODELO COMPACTO Y RÁPIDO (PRO PREDICCIÓN RTB)")
print("=" * 80)

# HistGradientBoostingClassifier es ultra-rápido para decisiones en tiempo real (< 5ms)
model_ads = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

model_ads.fit(X_train_scaled, y_train)

# Predicción de probabilidades de conversión (pCTR)
y_pred_proba = model_ads.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"✓ Modelo entrenado exitosamente.")
print(f"✓ Métrica ROC-AUC en AdTech (Discriminación de Conversión): {auc_score:.4f}")

# ============================================================================
# 4. OPTIMIZACIÓN DE PUJAS EN TIEMPO REAL (REAL-TIME BIDDING - RTB)
# ============================================================================
print("\n" + "=" * 80)
print("3. SIMULACIÓN DE PUJAS Y CÁLCULO DE RETORNO DE INVERSIÓN (ROAS / eCPM)")
print("=" * 80)

# Estrategia de puja basada en valor esperado: Bid = pCTR * Valor de la Conversión ($USD)
valor_conversion_usd = 25.0  # Ganancia por cada compra/conversión
costo_base_cpm = 2.50         # Costo por mil impresiones base ($)

df_test_results = X_test.copy()
df_test_results['pCTR'] = y_pred_proba
df_test_results['conversion_real'] = y_test.values

# Asignar oferta recomendada
df_test_results['oferta_puja_usd'] = df_test_results['pCTR'] * valor_conversion_usd

# Filtrar subastas ganadas (solo pujar si la probabilidad supera el umbral rentable)
umbral_pCTR = 0.15
impresiones_ganadas = df_test_results[df_test_results['pCTR'] >= umbral_pCTR]

conversiones_logradas = impresiones_ganadas['conversion_real'].sum()
inversion_total = impresiones_ganadas['oferta_puja_usd'].sum()
ingreso_total = conversiones_logradas * valor_conversion_usd
roas = (ingreso_total / inversion_total) if inversion_total > 0 else 0

print(f"• Impresiones evaluadas en subasta: {len(X_test):,}")
print(f"• Impresiones ganadas (Filtro Inteligente): {len(impresiones_ganadas):,}")
print(f"• Conversiones reales obtenidas: {conversiones_logradas}")
print(f"• Inversión publicitaria total: ${inversion_total:,.2f} USD")
print(f"• Ingresos generados: ${ingreso_total:,.2f} USD")
print(f"• ROAS (Return on Ad Spend): {roas:.2f}x (Por cada $1 invertido se recuperan ${roas:.2f})")

# ============================================================================
# 5. VISUALIZACIÓN DE RESULTADOS Y RENDIMIENTO DE CAMPAÑA
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Distribución de pCTR (Probabilidad de Conversión)
sns.histplot(df_test_results['pCTR'], bins=30, kde=True, ax=axes[0], color='teal')
axes[0].axvline(umbral_pCTR, color='red', linestyle='--', label=f'Umbral de Puja (pCTR >= {umbral_pCTR})')
axes[0].set_title('Distribución de pCTR (Probabilidad de Clic/Conversión)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Probabilidad de Conversión Estimada')
axes[0].set_ylabel('Cantidad de Impresiones')
axes[0].legend()

# Gráfico 2: Matriz de Confusión de Predicción Publicitaria
y_pred_binary = (y_pred_proba >= umbral_pCTR).astype(int)
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
axes[1].set_title('Matriz de Decisión de Subasta Publicitaria', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicción de Subasta (1: Pujar / 0: Ignorar)')
axes[1].set_ylabel('Conversión Real del Usuario')

plt.tight_layout()
#plt.savefig('06_nano_banana_publicidad_digital.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("✓ Gráfico de análisis publicitario guardado: '06_nano_banana_publicidad_digital.png'")
print("=" * 80)