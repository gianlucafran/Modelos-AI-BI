"""
UMAP: Uniform Manifold Approximation and Projection
Visualización de Embeddings en Alta Dimensionalidad
Demostración con dataset Digits (imágenes vectorizadas) de scikit-learn

Conceptos:
- Embeddings: Representaciones vectoriales densas en un espacio de alta dimensionalidad 
  (en este caso, los vectores de 64 píxeles que representan cada imagen).
- UMAP: Técnica moderna de reducción de dimensionalidad basada en geometría Riemanniana 
  y topología algebraica.
- Preservación Global: A diferencia de t-SNE, UMAP logra preservar tanto la estructura 
  local (vecindarios) como la estructura global (distancias entre diferentes clusters).
- Escalabilidad: Es significativamente más rápido y eficiente que t-SNE, siendo el 
  estándar actual para visualizar embeddings de Deep Learning (textos, imágenes, grafos).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import time

# Nota: Requiere instalación previa de umap-learn (pip install umap-learn)
try:
    import umap
except ImportError:
    print("Por favor, instala umap-learn usando: pip install umap-learn")
    import sys
    sys.exit()

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("tab10")

# ============================================================================
# 1. CARGAR Y EXPLORAR EL DATASET (COMO EMBEDDINGS)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR EL DATASET (EMBEDDINGS DE ENTRADA)")
print("=" * 80)

# Usamos el dataset de dígitos, donde cada imagen de 8x8 es aplanada a un 
# vector de 64 dimensiones. Trataremos estos vectores como nuestros "embeddings".
digits = load_digits()
X = digits.data
y = digits.target
target_names = digits.target_names

print(f"\nDataset: Digits (Tratado como embeddings vectoriales)")
print(f"Número de embeddings (muestras): {X.shape[0]}")
print(f"Dimensionalidad original del embedding: {X.shape[1]}")
print(f"Clases reales (dígitos 0-9): {target_names}")

# ============================================================================
# 2. ESTANDARIZACIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: ESTANDARIZACIÓN DE LOS EMBEDDINGS")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nLos embeddings han sido estandarizados.")
print("  - UMAP utiliza el cálculo de distancias (por defecto Euclidiana o Angular) ")
print("    en el espacio de alta dimensionalidad para construir un grafo topológico.")
print("  - Estandarizar asegura que ninguna dimensión domine métricamente a las demás.")

# ============================================================================
# 3. APLICACIÓN DE UMAP PARA VISUALIZACIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: PROYECCIÓN DE EMBEDDINGS (PCA vs UMAP)")
print("=" * 80)

# 3.1 PCA (Para comparación)
print("\nEjecutando PCA (Proyección Lineal)...")
t0 = time.time()
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
t_pca = time.time() - t0
print(f"✓ PCA completado en {t_pca:.2f} segundos.")

# 3.2 UMAP (Proyección Topológica)
print("\nEjecutando UMAP (Preservación Local y Global)...")
t0 = time.time()
# Parámetros por defecto de UMAP (muy robustos para la mayoría de casos)
reducer = umap.UMAP(
    n_neighbors=15,      # Balance local vs global
    min_dist=0.1,        # Distancia mínima entre puntos (qué tan apretados)
    n_components=2,      # Dimensiones de salida (2D para visualización)
    metric='euclidean',  # Métrica de distancia ('cosine' es ideal para embeddings de texto)
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)
t_umap = time.time() - t0
print(f"✓ UMAP completado en {t_umap:.2f} segundos.")
print("  (A pesar de ser no lineal y complejo, UMAP está altamente optimizado y es muy veloz).")

# ============================================================================
# 4. VISUALIZACIÓN COMPARATIVA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: VISUALIZACIÓN DE EMBEDDINGS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico PCA
scatter_pca = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', 
                              alpha=0.7, s=15, edgecolor='none')
axes[0].set_title(f'PCA: Proyección Lineal\nSolapamiento masivo de clases', fontsize=13, fontweight='bold')
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')
axes[0].grid(alpha=0.3)

# Gráfico UMAP
scatter_umap = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', 
                               alpha=0.7, s=15, edgecolor='none')
axes[1].set_title(f'UMAP: Proyección de Embeddings\nClusters puros y topología preservada', fontsize=13, fontweight='bold')
axes[1].set_xlabel('UMAP Dim 1')
axes[1].set_ylabel('UMAP Dim 2')
axes[1].grid(alpha=0.3)

legend = axes[1].legend(*scatter_umap.legend_elements(),
                        loc="best", title="Clases (0-9)", markerscale=2)
axes[1].add_artist(legend)

plt.tight_layout()
plt.savefig('/home/usuario/01_pca_vs_umap.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_pca_vs_umap.png")
print("  Observación: UMAP no solo separa los números de forma nítida (como t-SNE),")
print("  sino que las distancias entre clústeres son significativas (ej. números ")
print("  visualmente similares como 1 y 7, o 3 y 8 pueden estar más cercanos).")
plt.close()

# ============================================================================
# 5. EXPLORACIÓN DE HIPERPARÁMETROS (N_NEIGHBORS y MIN_DIST)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: IMPACTO DE LOS HIPERPARÁMETROS DE UMAP")
print("=" * 80)

print("\nUMAP tiene dos hiperparámetros fundamentales que moldean la visualización:")
print("1. n_neighbors: Controla el balance entre estructura local (bajo) y global (alto).")
print("2. min_dist: Controla qué tan apretado se empacan los puntos en la proyección final.")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Fila 1: Variando n_neighbors (con min_dist fijo = 0.1)
neighbors = [5, 15, 50]
for i, n in enumerate(neighbors):
    print(f"  Calculando UMAP (n_neighbors={n}, min_dist=0.1)...")
    u = umap.UMAP(n_neighbors=n, min_dist=0.1, random_state=42)
    embedding = u.fit_transform(X_scaled)
    axes[0, i].scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='tab10', alpha=0.7, s=10)
    axes[0, i].set_title(f'Variando Estructura\nn_neighbors = {n}', fontsize=12, fontweight='bold')
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    
# Fila 2: Variando min_dist (con n_neighbors fijo = 15)
dists = [0.0, 0.25, 0.8]
for i, d in enumerate(dists):
    print(f"  Calculando UMAP (n_neighbors=15, min_dist={d})...")
    u = umap.UMAP(n_neighbors=15, min_dist=d, random_state=42)
    embedding = u.fit_transform(X_scaled)
    axes[1, i].scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='tab10', alpha=0.7, s=10)
    axes[1, i].set_title(f'Variando Empaquetado\nmin_dist = {d}', fontsize=12, fontweight='bold')
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

plt.tight_layout()
plt.savefig('/home/usuario/02_umap_hiperparametros.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 02_umap_hiperparametros.png")
plt.close()

# ============================================================================
# 6. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               UMAP PARA VISUALIZACIÓN DE EMBEDDINGS                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. MANEJO DE ALTA DIMENSIONALIDAD (EMBEDDINGS):
   • UMAP es excepcional para tomar representaciones vectoriales densas 
     (embeddings de texto como BERT, imágenes como CNNs, etc.) y proyectarlas 
     en 2D o 3D para entender cómo el modelo agrupó semánticamente los datos.

2. ¿POR QUÉ UMAP Y NO t-SNE?:
   ✓ Mayor velocidad: Escala infinitamente mejor para millones de puntos.
   ✓ Estructura Global: Las distancias ENTRE los clústeres tienen significado. 
     (Si el clúster A está lejos del clúster B, realmente son conceptualmente 
     diferentes en el espacio de embedding).
   ✓ Transformación de nuevos datos: A diferencia de t-SNE, UMAP permite usar 
     `transform()` para proyectar nuevos datos en un espacio previamente ajustado.

3. HIPERPARÁMETROS CLAVE:
   • n_neighbors (2-100): 
     - Valores bajos focalizan en estructura muy fina (vecindarios pequeños).
     - Valores altos fuerzan a ver la "foto completa" (visión macro).
   • min_dist (0.0-0.99):
     - 0.0: Aglomeración pura (útil para clustering puro).
     - > 0.5: Dispersión suave (útil para ver la topología general sin amontonar).

4. MÉTRICA DE DISTANCIA (metric):
   • Para este caso (píxeles) usamos 'euclidean'.
   • PRO TIP: Cuando visualices embeddings de texto (ej. Word2Vec, TF-IDF), 
     cambiar a metric='cosine' mejora dramáticamente los resultados.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_pca_vs_umap.png          - Comparativa lineal vs topológica
02_umap_hiperparametros.png - Efecto de n_neighbors y min_dist

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis y proyección de embeddings completados exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")