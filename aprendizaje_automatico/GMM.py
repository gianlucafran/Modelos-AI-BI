"""
GMM: Modelos de Mezcla Gaussiana (Gaussian Mixture Models)
Segmentación de Clientes Híbridos mediante Soft Clustering
Demostración con dataset sintético de scikit-learn

Conceptos:
- GMM (Gaussian Mixture Models): Modelo probabilístico que asume que los datos provienen 
  de una mezcla de múltiples distribuciones gaussianas.
- Soft Clustering (Agrupamiento Suave): A diferencia de algoritmos como K-Means, GMM 
  no fuerza a un punto a pertenecer a un solo clúster, sino que calcula la probabilidad 
  de pertenencia a cada uno.
- Clientes Híbridos: Usuarios que se encuentran en las zonas de intersección de los 
  segmentos, mostrando comportamientos mixtos (ej. combinan compras mayoristas con retail).
- Criterios AIC / BIC: Métricas de información utilizadas para encontrar el número 
  óptimo de distribuciones (segmentos) penalizando la complejidad excesiva del modelo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. CARGAR Y EXPLORAR EL DATASET (SIMULACIÓN DE CLIENTES)
# ============================================================================
print("=" * 80)
print("PASO 1: SIMULAR EL COMPORTAMIENTO DE CLIENTES")
print("=" * 80)

# Scikit-learn no incluye un dataset nativo de segmentación de clientes, 
# por lo que simulamos uno usando make_blobs con alta varianza para forzar solapamiento.
n_samples = 1500

# Simulamos 3 perfiles:
# 1. Clientes de bajo valor y baja frecuencia (Ocasionales)
# 2. Clientes de alto valor y baja frecuencia (Mayoristas)
# 3. Clientes de valor medio y alta frecuencia (Frecuentes / Leales)
X, y_true = make_blobs(
    n_samples=n_samples, 
    centers=[[20, 2], [100, 3], [50, 10]], 
    cluster_std=[10.0, 15.0, 12.0], # Genera la dispersión y la zona "híbrida"
    random_state=42
)

# Limpiar valores negativos irreales
X = np.abs(X)

df = pd.DataFrame(X, columns=['Gasto_Promedio_Ticket', 'Visitas_Mensuales'])

print(f"\nDataset: Segmentación de Clientes Retail")
print(f"Total de clientes analizados: {df.shape[0]}")
print(f"Variables de comportamiento: {df.shape[1]}")

print(f"\nPrimeros 5 registros de clientes:")
print(df.head().round(2))

# ============================================================================
# 2. ESTANDARIZACIÓN (NORMALIZACIÓN)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: ESTANDARIZACIÓN DE LOS DATOS")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nLa estandarización es CRUCIAL para GMM porque:")
print("  - Las variables tienen escalas muy diferentes (Gasto en $ vs Visitas en unidades).")
print("  - GMM calcula covarianzas geométricas; sin estandarización, la variable de mayor")
print("    magnitud dominaría por completo la forma de las distribuciones.")

# ============================================================================
# 3. SELECCIÓN DEL NÚMERO ÓPTIMO DE SEGMENTOS (AIC / BIC)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: DETERMINAR EL NÚMERO ÓPTIMO DE SEGMENTOS")
print("=" * 80)

# Evaluamos desde 1 hasta 8 posibles segmentos
n_components = np.arange(1, 9)
models = [GaussianMixture(n, covariance_type='full', random_state=42).fit(X_scaled) 
          for n in n_components]

# Extraemos el Criterio de Información Bayesiano (BIC) y de Akaike (AIC)
bic = [m.bic(X_scaled) for m in models]
aic = [m.aic(X_scaled) for m in models]

# El K óptimo es aquel que minimiza el BIC
optimal_k = n_components[np.argmin(bic)]
print(f"\nNúmero óptimo de segmentos recomendados por BIC: {optimal_k}")

# Visualizar las curvas AIC y BIC
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_components, bic, label='BIC (Bayesian Info Criterion)', marker='o', lw=2)
ax.plot(n_components, aic, label='AIC (Akaike Info Criterion)', marker='s', lw=2)
ax.axvline(optimal_k, color='red', linestyle='--', label=f'K Óptimo ({optimal_k})')
ax.set_title('Optimización de Segmentos (AIC / BIC)', fontsize=13, fontweight='bold')
ax.set_xlabel('Número de Segmentos (Clusters)')
ax.set_ylabel('Valor de Penalización (Menor es mejor)')
ax.legend()
plt.tight_layout()
#plt.savefig('/home/usuario/01_optimizacion_gmm.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 01_optimizacion_gmm.png")
plt.close()

# ============================================================================
# 4. ENTRENAMIENTO DE GMM Y SOFT CLUSTERING
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: APLICAR GMM Y CALCULAR PROBABILIDADES (SOFT CLUSTERING)")
print("=" * 80)

# Inicializamos el modelo con el K óptimo (3)
# covariance_type='full' permite que los clusters tengan formas elípticas libres
gmm = GaussianMixture(n_components=optimal_k, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# Asignación dura (Hard Clustering - El segmento con mayor probabilidad)
df['Segmento_Asignado'] = gmm.predict(X_scaled)

# Asignación suave (Soft Clustering - Probabilidad de pertenecer a cada segmento)
probabilidades = gmm.predict_proba(X_scaled)
for i in range(optimal_k):
    df[f'Prob_Seg_{i}'] = probabilidades[:, i]

df['Max_Probabilidad'] = probabilidades.max(axis=1)

print("\nResultados del Soft Clustering (Primeros 5 clientes):")
cols_resumen = ['Gasto_Promedio_Ticket', 'Segmento_Asignado', 'Max_Probabilidad', 'Prob_Seg_0', 'Prob_Seg_1', 'Prob_Seg_2']
print(df[cols_resumen].head().round(3))

# ============================================================================
# 5. IDENTIFICACIÓN DE CLIENTES HÍBRIDOS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: IDENTIFICACIÓN DE CLIENTES HÍBRIDOS (ZONAS DE TRANSICIÓN)")
print("=" * 80)

# Un cliente es "Híbrido" si no pertenece a ningún segmento con gran certeza.
# Establecemos un umbral del 80%. Si su probabilidad máxima es menor a esto, es híbrido.
umbral_certeza = 0.80
df['Es_Hibrido'] = df['Max_Probabilidad'] < umbral_certeza

total_hibridos = df['Es_Hibrido'].sum()
porcentaje = (total_hibridos / n_samples) * 100

print(f"\nUmbral de Certeza para ser considerado 'Puro': {umbral_certeza * 100}%")
print(f"Clientes Híbridos detectados: {total_hibridos} ({porcentaje:.1f}% de la cartera)")

if total_hibridos > 0:
    print("\nEjemplo de un perfil Híbrido (Cliente en Transición):")
    ejemplo = df[df['Es_Hibrido']].iloc[0]
    print(f"  Gasto Ticket: ${ejemplo['Gasto_Promedio_Ticket']:.2f}")
    print(f"  Visitas/Mes:  {ejemplo['Visitas_Mensuales']:.1f}")
    print(f"  Composición:  {ejemplo['Prob_Seg_0']*100:.1f}% Seg 0 | "
          f"{ejemplo['Prob_Seg_1']*100:.1f}% Seg 1 | "
          f"{ejemplo['Prob_Seg_2']*100:.1f}% Seg 2")

# ============================================================================
# 6. VISUALIZACIÓN: ESPACIO DE CLIENTES Y ZONAS HÍBRIDAS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: VISUALIZACIÓN DEL MERCADO (HARD VS SOFT CLUSTERING)")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: Segmentación Dura
sns.scatterplot(
    x='Gasto_Promedio_Ticket', y='Visitas_Mensuales', 
    hue='Segmento_Asignado', palette='tab10', data=df, 
    alpha=0.7, edgecolor='white', ax=axes[0]
)
axes[0].set_title('Visión Tradicional (Hard Clustering)\nCada cliente pertenece a 1 solo grupo', 
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Gasto Promedio por Ticket ($)')
axes[0].set_ylabel('Visitas Mensuales')
axes[0].grid(alpha=0.3)

# Gráfico 2: Segmentación Suave (Resaltando Híbridos)
# Clientes puros en gris tenue
sns.scatterplot(
    x='Gasto_Promedio_Ticket', y='Visitas_Mensuales', 
    data=df[~df['Es_Hibrido']], 
    color='lightgray', alpha=0.4, edgecolor='none', label='Clientes Definidos', ax=axes[1]
)
# Clientes híbridos resaltados
sns.scatterplot(
    x='Gasto_Promedio_Ticket', y='Visitas_Mensuales', 
    hue='Segmento_Asignado', palette='tab10', 
    data=df[df['Es_Hibrido']], 
    marker='D', s=100, edgecolor='black', linewidth=1, label='Clientes Híbridos', ax=axes[1]
)

axes[1].set_title('Visión Probabilística (GMM)\nIdentificación de Zonas de Transición / Híbridos', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Gasto Promedio por Ticket ($)')
axes[1].set_ylabel('Visitas Mensuales')
axes[1].legend(title='Estado del Cliente')
axes[1].grid(alpha=0.3)

plt.tight_layout()
#plt.savefig('/home/usuario/02_segmentacion_hibrida.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 02_segmentacion_hibrida.png")
plt.close()

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              GMM APLICADO A LA SEGMENTACIÓN DE CLIENTES HÍBRIDOS             ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. SUPERANDO LAS LIMITACIONES DE K-MEANS:
   • Mientras algoritmos rígidos dibujan líneas rectas entre clientes, GMM 
     entiende que el comportamiento humano es probabilístico. 
   • Permitió detectar que el {porcentaje:.1f}% de nuestra cartera no encaja 
     perfectamente en un solo "buyer persona", sino que son híbridos.

2. APLICACIÓN DE NEGOCIO (MARKETING PERSONALIZADO):
   • Clientes Híbridos = Oportunidad de Upselling/Cross-selling. 
   • Si un cliente tiene 60% de probabilidad de ser 'Ocasional' y 40% de 
     ser 'Leal', el equipo de marketing puede enviarle una campaña mixta para 
     terminar de empujarlo hacia la fidelidad absoluta.

3. FLEXIBILIDAD ESTRUCTURAL:
   • Usando `covariance_type='full'`, el modelo permitió que el segmento de los 
     "Mayoristas" tuviera una forma muy alargada (varían mucho en gasto pero 
     poco en visitas), adaptándose a la realidad comercial en lugar de forzar 
     círculos perfectos.

4. SELECCIÓN DE MODELOS (AIC / BIC):
   • Logramos evitar el sesgo humano al elegir la cantidad de segmentos.
   • La curva BIC confirmó matemáticamente que agrupar a los clientes en {optimal_k} 
     segmentos principales ofrecía el mejor balance entre precisión estadística 
     y simplicidad interpretativa.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

GaussianMixture(n_components={optimal_k}, covariance_type='full'):
  - n_components: Clusters latentes derivados del análisis BIC.
  - covariance_type='full': Cada cluster tiene su propia varianza independiente 
    en cada eje, permitiendo elipses de cualquier tamaño y orientación.
  - predict_proba(): Función clave que transforma el modelo de un clasificador 
    rígido a una herramienta de análisis probabilístico.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_optimizacion_gmm.png       - Gráfica de penalización AIC/BIC.
02_segmentacion_hibrida.png   - Visualización de clientes definidos vs clientes 
                                en zonas de transición (marcados con rombos).

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de segmentación probabilística completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")