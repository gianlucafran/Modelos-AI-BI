"""
t-SNE: Exploración de Espacios Latentes y Visualización de Manifolds
Demostración con dataset Digits (imágenes de números) de scikit-learn

Conceptos:
- t-SNE (t-Distributed Stochastic Neighbor Embedding): Técnica no lineal de reducción 
  de dimensionalidad ideal para visualización.
- Espacio Latente: Representación comprimida donde datos semánticamente similares 
  (ej. el mismo número escrito por distintas personas) se agrupan en regiones cercanas.
- Perplejidad (Perplexity): Hiperparámetro crucial que define el tamaño del vecindario 
  local a considerar. Balancea la estructura local vs global.
- No linealidad: A diferencia de PCA, t-SNE puede desenredar estructuras complejas 
  (variedades o manifolds) en alta dimensionalidad.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import time

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("tab10")  # tab10 es ideal para 10 clases distintas

# ============================================================================
# 1. CARGAR Y EXPLORAR EL DATASET (IMÁGENES DE DÍGITOS)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR Y EXPLORAR EL DATASET")
print("=" * 80)

# Cargamos el dataset de dígitos (8x8 píxeles = 64 dimensiones)
digits = load_digits()
X = digits.data
y = digits.target
target_names = digits.target_names

print(f"\nDataset: Optical Recognition of Handwritten Digits")
print(f"Número de muestras (imágenes): {X.shape[0]}")
print(f"Número de características (píxeles por imagen): {X.shape[1]} (8x8 px)")
print(f"Clases reales (dígitos): {target_names}")

print(f"\nEjemplo de características de la primera imagen (vector de 64 px):")
print(X[0].reshape(8, 8).round(1))

# ============================================================================
# 2. ESTANDARIZACIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: PREPROCESAMIENTO DE LOS DATOS")
print("=" * 80)

# Aunque los píxeles están en escala 0-16, es buena práctica estandarizar
# para algoritmos basados en distancias como t-SNE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nSe ha aplicado StandardScaler.")
print("  - t-SNE depende de la medición de distancias euclidianas.")
print("  - Escalar previene que características con rangos mayores dominen el cálculo.")

# ============================================================================
# 3. PCA vs t-SNE: CREACIÓN DEL ESPACIO LATENTE
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: CREACIÓN DEL ESPACIO LATENTE (PCA vs t-SNE)")
print("=" * 80)

# 3.1 PCA (Línea Base)
print("\nEjecutando PCA (Línea base lineal)...")
t0 = time.time()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
t_pca = time.time() - t0
print(f"✓ PCA completado en {t_pca:.2f} segundos.")
print(f"  Varianza explicada por 2D: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
print("  (Nota: PCA captura muy poca varianza en 2D porque las imágenes son complejas)")

# 3.2 t-SNE (Exploración no lineal)
print("\nEjecutando t-SNE (Exploración de manifold no lineal)...")
print("Configuración recomendada de sklearn: init='pca' para mayor estabilidad.")
t0 = time.time()
tsne = TSNE(
    n_components=2, 
    perplexity=30.0,      # Balance estándar entre estructura local/global
    init='pca',           # Inicialización basada en PCA (recomendado)
    learning_rate='auto', 
    random_state=42,
    n_jobs=-1             # Usar todos los núcleos
)
X_tsne = tsne.fit_transform(X_scaled)
t_tsne = time.time() - t0
print(f"✓ t-SNE completado en {t_tsne:.2f} segundos.")
print("  t-SNE es computacionalmente más intensivo que PCA, pero el espacio latente")
print("  resultante agrupa características de manera semántica mucho más pura.")

# ============================================================================
# 4. VISUALIZACIÓN: COMPARATIVA DEL ESPACIO LATENTE
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: VISUALIZACIÓN COMPARATIVA")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico PCA
scatter_pca = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', 
                              alpha=0.6, s=15, edgecolor='none')
axes[0].set_title(f'Espacio Latente Lineal: PCA\n(Tiempo: {t_pca:.2f}s)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Componente Principal 1')
axes[0].set_ylabel('Componente Principal 2')
axes[0].grid(alpha=0.3)

# Gráfico t-SNE
scatter_tsne = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', 
                               alpha=0.6, s=15, edgecolor='none')
axes[1].set_title(f'Espacio Latente No Lineal: t-SNE\n(Tiempo: {t_tsne:.2f}s)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Dimensión t-SNE 1')
axes[1].set_ylabel('Dimensión t-SNE 2')
axes[1].grid(alpha=0.3)

# Leyenda compartida
legend1 = axes[1].legend(*scatter_tsne.legend_elements(),
                         loc="best", title="Dígitos", markerscale=2)
axes[1].add_artist(legend1)

plt.tight_layout()
plt.savefig('/home/usuario/01_pca_vs_tsne.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_pca_vs_tsne.png")
print("  Observación: t-SNE separa perfectamente los dígitos en el espacio latente,")
print("  mientras que PCA los mezcla en el centro.")
plt.close()

# ============================================================================
# 5. EXPLORACIÓN DE LA PERPLEJIDAD (PERPLEXITY)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: IMPACTO DE LA PERPLEJIDAD EN EL ESPACIO LATENTE")
print("=" * 80)

perplexities = [5, 30, 100]
print(f"Probando t-SNE con diferentes valores de perplejidad: {perplexities}")
print("  - Valores bajos (ej. 5): Se enfoca mucho en vecindarios locales (clusters fragmentados).")
print("  - Valores medios (ej. 30): Buen balance local/global (clusters bien definidos).")
print("  - Valores altos (ej. 100): Se enfoca en estructura global (clusters más aglomerados).")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, p in enumerate(perplexities):
    print(f"  Calculando t-SNE con perplexity={p}...")
    tsne_p = TSNE(n_components=2, perplexity=p, init='pca', 
                  learning_rate='auto', random_state=42, n_jobs=-1)
    X_p = tsne_p.fit_transform(X_scaled)
    
    axes[i].scatter(X_p[:, 0], X_p[:, 1], c=y, cmap='tab10', alpha=0.7, s=10)
    axes[i].set_title(f't-SNE (Perplexity = {p})', fontsize=12, fontweight='bold')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].grid(alpha=0.2)

plt.tight_layout()
plt.savefig('/home/usuario/02_tsne_perplejidades.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 02_tsne_perplejidades.png")
plt.close()

# ============================================================================
# 6. ANÁLISIS DE DISTANCIAS EN EL ESPACIO LATENTE (INTERPRETACIÓN)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: INTERPRETACIÓN DEL ESPACIO LATENTE")
print("=" * 80)

print("\nCaracterísticas del Espacio Latente Generado por t-SNE:")
print("1. Proximidad = Similitud Visual:")
print("   Si miramos el gráfico, el cluster del dígito '1' suele estar cerca del '7',")
print("   ya que visualmente comparten trazos similares (líneas verticales/diagonales).")
print("2. Preservación Local vs Global:")
print("   t-SNE garantiza que las imágenes que son muy parecidas estén juntas (distancia local).")
print("   Sin embargo, la distancia entre un cluster entero y otro (distancia global) NO")
print("   tiene un significado estricto en t-SNE. ¡No se deben medir distancias entre clusters distantes!")
print("3. Densidad:")
print("   El tamaño o densidad de un cluster en t-SNE no representa la dispersión real de los datos,")
print("   ya que el algoritmo adapta el tamaño basándose en la densidad local.")

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               t-SNE PARA EXPLORACIÓN DE ESPACIOS LATENTES                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN DEL ESPACIO LATENTE:
   • Se transformaron {X.shape[1]} dimensiones (píxeles crudos) en 2 dimensiones.
   • En este nuevo espacio 2D (latente), el modelo logró "entender" la semántica
     de los dígitos sin haber sido entrenado explícitamente para clasificarlos 
     (t-SNE es no supervisado).

2. COMPARATIVA PCA vs t-SNE:
   • PCA falló en separar las clases porque los píxeles de imágenes no tienen 
     una relación lineal simple.
   • t-SNE logró separar casi perfectamente los 10 dígitos en "islas" distintas.

3. EL PAPEL DE LA PERPLEJIDAD (PERPLEXITY):
   • Es el parámetro más importante de afinar. Representa el número estimado
     de vecinos cercanos que cada punto tiene.
   • Sklearn recomienda valores entre 5 y 50. Para el dataset Digits (1797 muestras),
     una perplejidad de 30 generó los mejores resultados visuales.

4. LIMITACIONES IMPORTANTES (ADVERTENCIAS):
   ❌ t-SNE NO debe usarse como preprocesamiento para modelos predictivos (como SVM).
   ❌ No existe la función `transform()` en t-SNE de sklearn para nuevos datos; 
      solo se usa para visualización estática (`fit_transform`).
   ❌ Las distancias globales (distancia entre clusters) no son fiables.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto'):
  - n_components=2: Para proyectar en un plano 2D visualizable.
  - init='pca': Recomendación actual de scikit-learn. Usar PCA para inicializar
    los puntos en lugar de 'random' hace que el proceso sea más estable y global.
  - learning_rate='auto': Ajusta la tasa de aprendizaje en función del tamaño (N).

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_pca_vs_tsne.png        - Comparativa directa entre el fallo de PCA y el éxito de t-SNE
02_tsne_perplejidades.png - Efecto de variar la perplejidad en la formación del manifold

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis y exploración del espacio latente completados exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")