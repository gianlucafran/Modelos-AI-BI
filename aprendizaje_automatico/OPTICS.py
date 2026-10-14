"""
OPTICS para Identificación de Puntos de Interés Urbanos
========================================================

Este script demuestra la aplicación del algoritmo OPTICS (Ordering Points To 
Identify the Clustering Structure) para detectar agrupaciones de puntos de 
interés (POI) en un área urbana simulada.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. GENERAR DATASET SIMULADO DE PUNTOS DE INTERÉS URBANOS
# ============================================================================

print("=" * 70)
print("OPTICS: Identificación de Puntos de Interés Urbanos")
print("=" * 70)

# Simular puntos de interés en una ciudad
# Generamos clusters de diferentes densidades (restaurantes, tiendas, parques, etc.)
np.random.seed(42)

# Crear puntos con diferentes densidades (zonas comerciales vs residenciales)
poi_coords = np.vstack([
    np.random.normal(loc=[2, 2], scale=0.3, size=(40, 2)),      # Zona Centro
    np.random.normal(loc=[8, 8], scale=0.5, size=(30, 2)),      # Zona Norte
    np.random.normal(loc=[0.5, 8], scale=0.4, size=(25, 2)),    # Zona Oeste
    np.random.normal(loc=[8, 0.5], scale=0.6, size=(20, 2)),    # Zona Este (dispersa)
    np.random.uniform(low=0, high=10, size=(15, 2))             # Ruido (puntos aislados)
])

X = poi_coords

# Normalizando los datos (simular coordenadas en km)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✓ Dataset generado: {X_scaled.shape[0]} puntos de interés urbanos")
print(f"  Dimensiones: {X_scaled.shape[1]} (latitud, longitud)")

# ============================================================================
# 2. APLICAR ALGORITMO OPTICS
# ============================================================================

print("\n" + "-" * 70)
print("CONFIGURACIÓN DE OPTICS")
print("-" * 70)

# Parámetros de OPTICS
min_samples = 5  # Mínimo de puntos para formar un cluster
max_eps = 3.0    # Máxima distancia a explorar
metric = 'euclidean'  # Métrica de distancia

print(f"\n📊 Parámetros utilizados:")
print(f"   • min_samples: {min_samples} (mínimo de puntos en un cluster)")
print(f"   • max_eps: {max_eps} (máxima distancia de vecindario)")
print(f"   • metric: {metric}")

# Crear y entrenar OPTICS
optics = OPTICS(
    min_samples=min_samples,
    max_eps=max_eps,
    metric=metric,
    xi=0.05,  # Parámetro de control de clusters en la jerarquía
    min_cluster_size=0.05  # Fracción mínima de puntos para formar cluster
)

# Ajustar el modelo
optics.fit(X_scaled)

# Obtener etiquetas de clusters
labels = optics.labels_

# Información sobre los clusters encontrados
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = list(labels).count(-1)

print(f"\n✅ Resultados de OPTICS:")
print(f"   • Número de clusters encontrados: {n_clusters}")
print(f"   • Puntos de ruido (no agrupados): {n_noise}")
print(f"   • Puntos en clusters: {len(labels) - n_noise}")

# Análisis por cluster
print(f"\n📍 Distribución de puntos por cluster:")
for cluster_id in sorted(unique_labels):
    if cluster_id == -1:
        continue
    count = list(labels).count(cluster_id)
    percentage = (count / len(labels)) * 100
    print(f"   Cluster {cluster_id}: {count:3d} puntos ({percentage:5.1f}%)")

if n_noise > 0:
    print(f"   Ruido (-1):    {n_noise:3d} puntos ({(n_noise/len(labels))*100:5.1f}%)")

# ============================================================================
# 3. ANÁLISIS DE REACHABILITY
# ============================================================================

print("\n" + "-" * 70)
print("ANÁLISIS DE ALCANZABILIDAD (REACHABILITY)")
print("-" * 70)

# Obtener la distancia de alcanzabilidad
reachability = optics.reachability_[optics.ordering_]

print(f"\n📈 Estadísticas de reachability:")
print(f"   • Mínimo: {np.min(reachability[np.isfinite(reachability)]):.4f}")
print(f"   • Máximo: {np.max(reachability[np.isfinite(reachability)]):.4f}")
print(f"   • Promedio: {np.mean(reachability[np.isfinite(reachability)]):.4f}")

# ============================================================================
# 4. VISUALIZACIÓN
# ============================================================================

print("\n" + "-" * 70)
print("GENERANDO VISUALIZACIONES")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('OPTICS: Identificación de Puntos de Interés Urbanos', 
             fontsize=16, fontweight='bold', y=1.00)

# --- Gráfico 1: Clusters encontrados ---
ax = axes[0, 0]
colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters + 1))

for cluster_id in sorted(unique_labels):
    if cluster_id == -1:
        col = 'red'
        marker = 'x'
        label = 'Ruido'
        size = 100
    else:
        col = colors[cluster_id % len(colors)]
        marker = 'o'
        label = f'Cluster {cluster_id}'
        size = 80

    class_member_mask = (labels == cluster_id)
    xy = X_scaled[class_member_mask]
    ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, s=size, 
              alpha=0.7, edgecolors='k', linewidth=0.5, label=label)

ax.set_xlabel('Latitud (normalizada)', fontweight='bold')
ax.set_ylabel('Longitud (normalizada)', fontweight='bold')
ax.set_title('Clusters de POI Identificados por OPTICS', fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# --- Gráfico 2: Gráfico de Reachability ---
ax = axes[0, 1]
reachability = optics.reachability_[optics.ordering_]
colors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(labels))))

for i, (cluster_id, color) in enumerate(zip(sorted(unique_labels), colors)):
    if cluster_id == -1:
        color = 'red'
    
    class_member_mask = labels[optics.ordering_] == cluster_id
    ax.plot(np.where(class_member_mask)[0], reachability[class_member_mask],
           'o', c=color, markersize=4, label=f'Cluster {cluster_id}' if cluster_id != -1 else 'Ruido')

ax.set_xlabel('Índice de Punto (ordenado)', fontweight='bold')
ax.set_ylabel('Distancia de Alcanzabilidad', fontweight='bold')
ax.set_title('Gráfico de Alcanzabilidad', fontweight='bold')
ax.axhline(y=max_eps, c='red', linestyle='--', linewidth=2, label=f'max_eps={max_eps}')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# --- Gráfico 3: Densidad de puntos ---
ax = axes[1, 0]
# Calcular k-distancia para visualizar densidad
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1], axis=0)

ax.plot(distances, linewidth=2, color='steelblue')
ax.axhline(y=max_eps, c='red', linestyle='--', linewidth=2, label=f'max_eps={max_eps}')
ax.set_xlabel('Puntos (ordenados)', fontweight='bold')
ax.set_ylabel(f'{min_samples}-ésima Distancia Vecino', fontweight='bold')
ax.set_title('Gráfico de Distancias k-NN', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# --- Gráfico 4: Información de clusters ---
ax = axes[1, 1]
ax.axis('off')

info_text = f"""
ESTADÍSTICAS DEL ANÁLISIS OPTICS

Parámetros:
  • min_samples: {min_samples}
  • max_eps: {max_eps}
  • metric: {metric}

Resultados:
  • Total de puntos: {len(labels)}
  • Clusters encontrados: {n_clusters}
  • Puntos en clusters: {len(labels) - n_noise}
  • Puntos de ruido: {n_noise}

Interpretación:
  Los clusters representan zonas de
  interés urbano (comercial, residencial,
  etc.) con densidades variables.
  
  El algoritmo OPTICS mantiene la
  jerarquía de clusters y es más
  flexible que DBSCAN para diferentes
  densidades de puntos.
"""

ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/claude/optics_urban_poi_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualización guardada: 'optics_urban_poi_visualization.png'")
plt.show()

# ============================================================================
# 5. ANÁLISIS DETALLADO POR CLUSTER
# ============================================================================

print("\n" + "-" * 70)
print("ANÁLISIS DETALLADO POR CLUSTER")
print("-" * 70)

for cluster_id in sorted(unique_labels):
    if cluster_id == -1:
        print(f"\n🔴 RUIDO (etiqueta: -1)")
        mask = labels == cluster_id
    else:
        print(f"\n🟢 CLUSTER {cluster_id}")
        mask = labels == cluster_id
    
    cluster_points = X_scaled[mask]
    
    # Calcular centroide
    centroid = np.mean(cluster_points, axis=0)
    
    # Calcular radio (máxima distancia al centroide)
    distances_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
    
    print(f"   Cantidad de puntos: {np.sum(mask)}")
    print(f"   Centroide: ({centroid[0]:.3f}, {centroid[1]:.3f})")
    print(f"   Radio del cluster: {np.max(distances_to_centroid):.3f}")
    print(f"   Densidad (puntos/unidad²): {np.sum(mask) / (np.pi * np.max(distances_to_centroid)**2):.2f}")

# ============================================================================
# 6. CÓDIGO DE EJEMPLO SIMPLE PARA USO RÁPIDO
# ============================================================================

print("\n" + "=" * 70)
print("CÓDIGO BÁSICO RÁPIDO")
print("=" * 70)

ejemplo_codigo = """
# Código mínimo para usar OPTICS

from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
import numpy as np

# 1. Cargar datos (puntos de interés urbanos)
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# 2. Crear modelo OPTICS
optics = OPTICS(min_samples=5, max_eps=3.0)

# 3. Entrenar y obtener etiquetas
labels = optics.fit_predict(X)

# 4. Acceder a resultados
print(f"Clusters encontrados: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Puntos de ruido: {list(labels).count(-1)}")
"""

print(ejemplo_codigo)

print("\n" + "=" * 70)
print("✅ Análisis completado exitosamente")
print("=" * 70)