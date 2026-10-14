"""
PCA: Análisis Multivariado Previo al Clustering
Demostración con dataset Iris de scikit-learn

Conceptos:
- PCA (Principal Component Analysis): Reduce dimensionalidad manteniendo varianza
- Análisis Multivariado: Analiza múltiples variables simultáneamente
- Preparación para Clustering: Mejora la calidad del clustering en espacios reducidos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. CARGAR Y EXPLORAR EL DATASET
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR Y EXPLORAR EL DATASET")
print("=" * 80)

iris = load_iris()
X = iris.data  # Características: [sepal_length, sepal_width, petal_length, petal_width]
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"\nDataset: Iris")
print(f"Número de muestras: {X.shape[0]}")
print(f"Número de características (dimensiones): {X.shape[1]}")
print(f"Características: {feature_names}")
print(f"\nClases reales: {target_names}")
print(f"\nPrimeras 5 muestras:\n{pd.DataFrame(X[:5], columns=feature_names)}")

# Estadísticas básicas
print(f"\nEstadísticas descriptivas:")
print(pd.DataFrame(X, columns=feature_names).describe())

# ============================================================================
# 2. ESTANDARIZACIÓN (NORMALIZACIÓN)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: ESTANDARIZACIÓN DE LOS DATOS")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nLa estandarización es CRUCIAL para PCA porque:")
print("  - PCA es sensible a la escala de los datos")
print("  - Centra los datos (media=0) y escala por desv. est. (std=1)")
print("  - Asegura que cada característica contribuya equitativamente")

print(f"\nMedia de datos estandarizados (debe ser ~0):")
print(f"  {np.mean(X_scaled, axis=0)}")
print(f"\nDesviación estándar (debe ser ~1):")
print(f"  {np.std(X_scaled, axis=0)}")

# ============================================================================
# 3. ANÁLISIS MULTIVARIADO: MATRIZ DE CORRELACIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ANÁLISIS MULTIVARIADO - MATRIZ DE CORRELACIÓN")
print("=" * 80)

correlation_matrix = np.corrcoef(X_scaled.T)
print("\nMatriz de Correlación (datos estandarizados):")
print(pd.DataFrame(correlation_matrix, 
                   index=feature_names, 
                   columns=feature_names).round(3))

print("\nObservaciones:")
print("  - Correlaciones altas indican redundancia entre variables")
print("  - PCA captura esta redundancia en componentes principales")

# Visualizar matriz de correlación
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            xticklabels=feature_names,
            yticklabels=feature_names,
            ax=ax,
            cbar_kws={'label': 'Correlación'})
plt.title('Matriz de Correlación - Dataset Iris', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/claude/01_matriz_correlacion.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 01_matriz_correlacion.png")
plt.close()

# ============================================================================
# 4. DETERMINAR NÚMERO ÓPTIMO DE COMPONENTES
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: DETERMINAR NÚMERO ÓPTIMO DE COMPONENTES")
print("=" * 80)

# PCA con todos los componentes
pca_full = PCA()
pca_full.fit(X_scaled)

# Varianza explicada
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nVarianza explicada por cada componente principal:")
for i, var in enumerate(explained_variance, 1):
    print(f"  PC{i}: {var:.4f} ({var*100:.2f}%)")

print(f"\nVarianza acumulada:")
for i, cum_var in enumerate(cumulative_variance, 1):
    print(f"  Hasta PC{i}: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Elegir componentes que expliquen ~95% de varianza
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponentes para explicar 95% de varianza: {n_components_95}")

# Visualizar varianza explicada
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Varianza individual
axes[0].bar(range(1, len(explained_variance) + 1), 
            explained_variance, 
            alpha=0.7, 
            color='steelblue',
            edgecolor='black')
axes[0].set_xlabel('Componente Principal', fontsize=12)
axes[0].set_ylabel('Varianza Explicada', fontsize=12)
axes[0].set_title('Varianza Explicada por Componente', fontsize=13, fontweight='bold')
axes[0].set_xticks(range(1, len(explained_variance) + 1))
axes[0].grid(axis='y', alpha=0.3)

# Gráfico 2: Varianza acumulada
axes[1].plot(range(1, len(cumulative_variance) + 1), 
             cumulative_variance, 
             marker='o', 
             linestyle='-', 
             linewidth=2,
             markersize=8,
             color='darkgreen')
axes[1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Varianza')
axes[1].axvline(x=n_components_95, color='red', linestyle='--', linewidth=2)
axes[1].fill_between(range(1, len(cumulative_variance) + 1), 
                      cumulative_variance, 
                      alpha=0.3, 
                      color='green')
axes[1].set_xlabel('Número de Componentes', fontsize=12)
axes[1].set_ylabel('Varianza Acumulada', fontsize=12)
axes[1].set_title('Varianza Acumulada', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(1, len(cumulative_variance) + 1))
axes[1].set_ylim([0, 1.05])
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/02_varianza_explicada.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 02_varianza_explicada.png")
plt.close()

# ============================================================================
# 5. APLICAR PCA (2 y 3 COMPONENTES)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: APLICAR PCA CON 2 Y 3 COMPONENTES")
print("=" * 80)

# PCA con 2 componentes (para visualización 2D)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# PCA con 3 componentes (para visualización 3D y análisis)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print(f"\nComponentes principales (2D):")
print(f"  Forma de datos originales: {X_scaled.shape}")
print(f"  Forma de datos PCA 2D: {X_pca_2d.shape}")
print(f"  Varianza explicada: {pca_2d.explained_variance_ratio_.sum():.4f}")

print(f"\nComponentes principales (3D):")
print(f"  Forma de datos PCA 3D: {X_pca_3d.shape}")
print(f"  Varianza explicada: {pca_3d.explained_variance_ratio_.sum():.4f}")

# Cargas de las características (loadings)
print(f"\nCargas del PCA 2D (contribución de características a componentes):")
loadings_df = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
print(loadings_df.round(4))

# ============================================================================
# 6. VISUALIZACIÓN: DATOS EN ESPACIO PCA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: VISUALIZACIÓN EN ESPACIO PCA")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gráfico 2D
for i, target in enumerate(np.unique(y)):
    mask = y == target
    axes[0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                   label=target_names[target],
                   alpha=0.7,
                   s=100,
                   edgecolors='black',
                   linewidth=0.5)

axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
axes[0].set_title('Datos en Espacio PCA (2D)\nClases Reales', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Gráfico con loadings
axes[1].set_xlim(-1, 1)
axes[1].set_ylim(-1, 1)
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].axvline(x=0, color='k', linewidth=0.5)

# Dibujar vectores de features
for i, feature in enumerate(feature_names):
    axes[1].arrow(0, 0, 
                 pca_2d.components_[0, i]*3,
                 pca_2d.components_[1, i]*3,
                 head_width=0.1, head_length=0.1,
                 fc='red', ec='red', alpha=0.7)
    axes[1].text(pca_2d.components_[0, i]*3.2,
                pca_2d.components_[1, i]*3.2,
                feature.replace(' (cm)', ''),
                fontsize=10, ha='center', fontweight='bold')

circle = plt.Circle((0, 0), 1, fill=False, edgecolor='blue', linestyle='--', linewidth=2)
axes[1].add_patch(circle)
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
axes[1].set_title('Cargas de Características (Loadings)\nContribución al PCA', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('/home/claude/03_espacio_pca.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 03_espacio_pca.png")
plt.close()

# ============================================================================
# 7. CLUSTERING K-MEANS EN ESPACIO ORIGINAL vs PCA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 7: K-MEANS CLUSTERING - COMPARACIÓN ORIGINAL vs PCA")
print("=" * 80)

# Encontrar K óptimo usando el método del codo
inertias_original = []
inertias_pca = []
silhouette_scores_original = []
silhouette_scores_pca = []

K_range = range(2, 11)

print("\nCalculando K-Means para K = 2 a 10...")
for k in K_range:
    # En espacio original
    kmeans_orig = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_orig.fit(X_scaled)
    inertias_original.append(kmeans_orig.inertia_)
    silhouette_scores_original.append(silhouette_score(X_scaled, kmeans_orig.labels_))
    
    # En espacio PCA
    kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_pca.fit(X_pca_3d)
    inertias_pca.append(kmeans_pca.inertia_)
    silhouette_scores_pca.append(silhouette_score(X_pca_3d, kmeans_pca.labels_))

# K óptimo (silhouette score más alto)
k_optimal_silhouette = K_range[np.argmax(silhouette_scores_pca)]
print(f"\nK óptimo (según Silhouette Score): {k_optimal_silhouette}")

# Usar K=3 (sabemos que hay 3 clases de Iris)
k_optimal = 3
print(f"Usar K = {k_optimal} (número real de clases)")

kmeans_final_orig = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
labels_orig = kmeans_final_orig.fit_predict(X_scaled)

kmeans_final_pca = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
labels_pca = kmeans_final_pca.fit_predict(X_pca_3d)

# Métricas de calidad
print(f"\nMétricas de Clustering (K={k_optimal}):")
print(f"\nEspacio Original:")
print(f"  Silhouette Score: {silhouette_score(X_scaled, labels_orig):.4f}")
print(f"  Davies-Bouldin Index: {davies_bouldin_score(X_scaled, labels_orig):.4f}")

print(f"\nEspacio PCA:")
print(f"  Silhouette Score: {silhouette_score(X_pca_3d, labels_pca):.4f}")
print(f"  Davies-Bouldin Index: {davies_bouldin_score(X_pca_3d, labels_pca):.4f}")

# Visualizar métodos del codo y silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Método del codo
axes[0].plot(K_range, inertias_original, 'o-', linewidth=2, markersize=8, label='Original', alpha=0.7)
axes[0].plot(K_range, inertias_pca, 's-', linewidth=2, markersize=8, label='PCA', alpha=0.7)
axes[0].axvline(x=k_optimal, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Número de Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inercia', fontsize=12)
axes[0].set_title('Método del Codo', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Silhouette Score
axes[1].plot(K_range, silhouette_scores_original, 'o-', linewidth=2, markersize=8, label='Original', alpha=0.7)
axes[1].plot(K_range, silhouette_scores_pca, 's-', linewidth=2, markersize=8, label='PCA', alpha=0.7)
axes[1].axvline(x=k_optimal, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].axvline(x=k_optimal_silhouette, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'K óptimo ({k_optimal_silhouette})')
axes[1].set_xlabel('Número de Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score (Mayor es mejor)', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/04_optimo_clusters.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 04_optimo_clusters.png")
plt.close()

# ============================================================================
# 8. VISUALIZACIÓN: CLUSTERING EN PCA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 8: VISUALIZACIÓN DEL CLUSTERING")
print("=" * 80)

fig = plt.figure(figsize=(16, 5))

# Clustering en espacio PCA 2D
ax1 = fig.add_subplot(131)
scatter = ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                     c=labels_pca, cmap='viridis',
                     s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
centroids = kmeans_final_pca.cluster_centers_
ax1.scatter(centroids[:, 0], centroids[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroides')
ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax1.set_title('K-Means Clustering en PCA 2D', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Cluster')

# Clases reales vs Clusters predichos
ax2 = fig.add_subplot(132)
ax2.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
           c=y, cmap='Set1', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax2.set_title('Clases Reales (Iris)', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Comparación lado a lado
ax3 = fig.add_subplot(133)
comparison = (labels_pca == y).astype(int)
colors_comparison = ['green' if c == 1 else 'red' for c in comparison]
ax3.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
           c=colors_comparison, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax3.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax3.set_title('Concordancia: Clases vs Clusters\n(Verde=Acierto, Rojo=Error)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/05_clustering_pca.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 05_clustering_pca.png")
plt.close()

# ============================================================================
# 9. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PCA COMO ANÁLISIS MULTIVARIADO                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. REDUCCIÓN DE DIMENSIONALIDAD:
   • Datos originales: {X.shape[1]} dimensiones (características)
   • Datos PCA: 2-3 componentes principales
   • Varianza retenida en 2D: {pca_2d.explained_variance_ratio_.sum()*100:.2f}%
   • Varianza retenida en 3D: {pca_3d.explained_variance_ratio_.sum()*100:.2f}%

2. BENEFICIOS DEL PCA PREVIO AL CLUSTERING:
   ✓ Elimina ruido y características redundantes
   ✓ Mejora la separabilidad de clusters
   ✓ Reduce carga computacional
   ✓ Facilita visualización en 2D/3D
   ✓ Estabiliza algoritmos de clustering

3. ANÁLISIS MULTIVARIADO:
   • Se analizaron simultáneamente 4 características
   • Se identificaron correlaciones entre variables
   • PCA capturó patrones complejos de interacción

4. RESULTADOS DEL CLUSTERING:
   • Silhouette Score (PCA): {silhouette_scores_pca[k_optimal-2]:.4f}
   • Davies-Bouldin Index (PCA): {davies_bouldin_score(X_pca_3d, labels_pca):.4f}
   
5. COMPONENTES PRINCIPALES:
   • PC1 explica: {pca_2d.explained_variance_ratio_[0]*100:.2f}% de varianza
   • PC2 explica: {pca_2d.explained_variance_ratio_[1]*100:.2f}% de varianza
   • PC3 explica: {pca_3d.explained_variance_ratio_[2]*100:.2f}% de varianza
   
6. CARACTERÍSTICAS MÁS IMPORTANTES:
   • Para PC1: {feature_names[np.argmax(np.abs(pca_2d.components_[0]))]}
   • Para PC2: {feature_names[np.argmax(np.abs(pca_2d.components_[1]))]}

╔══════════════════════════════════════════════════════════════════════════════╗
║                            PARÁMETROS UTILIZADOS                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

StandardScaler():
  - Centra datos: media = 0
  - Escala datos: desv. estándar = 1

PCA(n_components=2/3):
  - svd_solver='auto': Elige automáticamente el mejor SVD
  - copy=True: No modifica datos originales

KMeans(n_clusters=3):
  - n_init=10: Ejecuta 10 veces con diferentes inicios
  - random_state=42: Reproducibilidad
  - Métrica: Distancia euclideana

Métricas de Evaluación:
  - Silhouette Score: [-1, 1], mayor es mejor
  - Davies-Bouldin Index: Menor es mejor
  - Método del codo: Busca punto de inflexión

╔══════════════════════════════════════════════════════════════════════════════╗
║                          ARCHIVOS GENERADOS                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_matriz_correlacion.png     - Correlaciones entre variables originales
02_varianza_explicada.png     - Varianza por componente (scree plot)
03_espacio_pca.png             - Datos en espacio PCA y cargas
04_optimo_clusters.png         - Método del codo y Silhouette scores
05_clustering_pca.png          - Resultados de K-Means en PCA

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/claude/")