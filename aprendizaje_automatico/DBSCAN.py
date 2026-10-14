"""
DBSCAN para Identificación de Anomalías Geoespaciales
=======================================================

Implementación de DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
para detectar anomalías en datos geoespaciales usando scikit-learn.

Características:
- Detección de outliers automática (puntos en baja densidad)
- No requiere especificar número de clusters
- Ideal para datos con coordenadas latitude-longitude
- Métrica haversine para distancias geográficas reales

Documentación: https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.DBSCAN.html
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================

print("=" * 80)
print("DBSCAN: Detección de Anomalías Geoespaciales")
print("=" * 80)

# Parámetros de DBSCAN
EPS = 0.05  # Radio de búsqueda en km (para haversine)
MIN_SAMPLES = 5  # Mínimo de puntos para formar un cluster
METRIC = 'haversine'  # Métrica para distancias geográficas
RANDOM_STATE = 42

print(f"\n⚙️ Configuración DBSCAN:")
print(f"   eps (radio): {EPS} km")
print(f"   min_samples: {MIN_SAMPLES}")
print(f"   métrica: {METRIC} (para lat-lon)")

# ============================================================================
# 2. GENERACIÓN DE DATOS GEOESPACIALES DE PRUEBA
# ============================================================================

print("\n" + "=" * 80)
print("📍 GENERANDO DATASET GEOESPACIAL")
print("=" * 80)

def generar_datos_geoespaciales(n_clusters: int = 4, n_samples: int = 300,
                               n_outliers: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos geoespaciales (lat-lon) con clusters y anomalías.
    
    Args:
        n_clusters: Número de clusters normales
        n_samples: Muestras por cluster
        n_outliers: Número de anomalías
    
    Returns:
        Coordenadas (lat, lon), etiquetas verdaderas
    """
    
    # Centros de clusters (México - Querétaro y alrededores)
    cluster_centers = np.array([
        [20.5937, -100.3161],  # Querétaro Centro
        [20.6296, -100.4865],  # San Juan del Río
        [20.7500, -99.9500],   # Celaya
        [21.0000, -101.2500],  # Guanajuato
    ])
    
    # Generar puntos en clusters
    X = []
    y = []
    
    for cluster_id, center in enumerate(cluster_centers):
        # Puntos normales en cluster (desviación ~0.05 grados)
        cluster_points = center + np.random.normal(0, 0.05, (n_samples, 2))
        X.append(cluster_points)
        y.extend([cluster_id] * n_samples)
    
    # Agregar anomalías (puntos aleatorios fuera de clusters)
    anomalies = np.random.uniform([18, -103], [23, -98], (n_outliers, 2))
    X.append(anomalies)
    y.extend([-1] * n_outliers)  # -1 indica anomalía real
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Mezclar
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    return X, y

# Generar datos
X, y_true = generar_datos_geoespaciales(n_clusters=4, n_samples=75, n_outliers=40)

print(f"\n✅ Dataset generado:")
print(f"   Puntos totales: {len(X)}")
print(f"   Clusters normales: 4")
print(f"   Anomalías reales: {sum(y_true == -1)}")
print(f"   Rango lat: [{X[:, 0].min():.4f}, {X[:, 0].max():.4f}]")
print(f"   Rango lon: [{X[:, 1].min():.4f}, {X[:, 1].max():.4f}]")

# ============================================================================
# 3. ESTIMACIÓN DE EPSILON (eps)
# ============================================================================

print("\n" + "=" * 80)
print("📏 ESTIMACIÓN DE PARÁMETRO eps")
print("=" * 80)

def estimar_eps(X: np.ndarray, k: int = 5, metric: str = 'haversine') -> float:
    """
    Estima valor óptimo de eps usando k-distance graph.
    
    Args:
        X: Datos de entrada
        k: Número de vecinos (generalmente min_samples)
        metric: Métrica de distancia
    
    Returns:
        Valor sugerido de eps
    """
    # Convertir a radianes para haversine
    X_rad = np.radians(X)
    
    # Calcular distancias a k-ésimo vecino más cercano
    neighbors = NearestNeighbors(n_neighbors=k, metric=metric)
    neighbors_fit = neighbors.fit(X_rad)
    distances, indices = neighbors_fit.kneighbors(X_rad)
    
    # Ordenar distancias
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Encontrar codo (sugerencia)
    # Para haversine, distancia en radianes ≈ distancia en km / 6371 (radio Tierra)
    distances_km = distances * 6371
    
    return distances_km

distances = estimar_eps(X, k=MIN_SAMPLES)

print(f"\n🔍 Análisis de k-distance graph:")
print(f"   k = {MIN_SAMPLES}")
print(f"   Distancia mínima: {distances[0]:.3f} km")
print(f"   Distancia máxima: {distances[-1]:.3f} km")
print(f"   Percentil 25: {np.percentile(distances, 25):.3f} km")
print(f"   Percentil 50 (mediana): {np.percentile(distances, 50):.3f} km")
print(f"   Percentil 75: {np.percentile(distances, 75):.3f} km")
print(f"   Percentil 90: {np.percentile(distances, 90):.3f} km")

print(f"\n💡 Valor de eps utilizado: {EPS} km")
print(f"   (Aproximadamente {EPS * 111:.2f} metros en línea recta)")

# ============================================================================
# 4. APLICAR DBSCAN
# ============================================================================

print("\n" + "=" * 80)
print("🎯 APLICANDO DBSCAN")
print("=" * 80)

# Convertir a radianes para métrica haversine
X_rad = np.radians(X)

# Crear y entrenar DBSCAN
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric=METRIC)
labels = db.fit_predict(X_rad)

# Estadísticas
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_anomalies = list(labels).count(-1)
n_noise_points = (labels == -1).sum()

print(f"\n✅ Resultados DBSCAN:")
print(f"   Clusters detectados: {n_clusters}")
print(f"   Puntos de ruido (anomalías): {n_anomalies}")
print(f"   Puntos normales: {len(X) - n_anomalies}")

# Distribución de clusters
print(f"\n📊 Distribución por cluster:")
for cluster_id in sorted(set(labels)):
    count = (labels == cluster_id).sum()
    if cluster_id == -1:
        print(f"   Ruido/Anomalías: {count}")
    else:
        print(f"   Cluster {cluster_id}: {count}")

# ============================================================================
# 5. ANÁLISIS DE ANOMALÍAS DETECTADAS
# ============================================================================

print("\n" + "=" * 80)
print("🔍 ANÁLISIS DE ANOMALÍAS")
print("=" * 80)

# Obtener índices de anomalías
anomaly_indices = np.where(labels == -1)[0]
normal_indices = np.where(labels != -1)[0]

# Crear dataframe con resultados
results_df = pd.DataFrame({
    'latitud': X[:, 0],
    'longitud': X[:, 1],
    'cluster': labels,
    'tipo': ['Anomalía' if l == -1 else 'Normal' for l in labels],
    'es_anomalia_real': [y == -1 for y in y_true]
})

print(f"\nPrimeras anomalías detectadas:")
anomalies_df = results_df[results_df['tipo'] == 'Anomalía'].head(5)
for idx, row in anomalies_df.iterrows():
    print(f"   Lat: {row['latitud']:.4f}, Lon: {row['longitud']:.4f} | Cluster: {row['cluster']}")

# ============================================================================
# 6. EVALUACIÓN DE DETECCIÓN
# ============================================================================

print("\n" + "=" * 80)
print("📈 EVALUACIÓN DE DETECCIÓN")
print("=" * 80)

# Comparar con anomalías reales
predicted_anomalies = (labels == -1)
true_anomalies = (y_true == -1)

# Métricas
tp = np.sum((predicted_anomalies) & (true_anomalies))  # Verdaderos positivos
fp = np.sum((predicted_anomalies) & (~true_anomalies))  # Falsos positivos
tn = np.sum((~predicted_anomalies) & (~true_anomalies))  # Verdaderos negativos
fn = np.sum((~predicted_anomalies) & (true_anomalies))  # Falsos negativos

# Cálculos
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\n✅ Métricas de desempeño:")
print(f"   Verdaderos Positivos (TP): {tp}")
print(f"   Falsos Positivos (FP): {fp}")
print(f"   Verdaderos Negativos (TN): {tn}")
print(f"   Falsos Negativos (FN): {fn}")

print(f"\n📊 Índices de rendimiento:")
print(f"   Precisión: {precision:.2%}")
print(f"   Recall (Sensibilidad): {recall:.2%}")
print(f"   F1-Score: {f1:.4f}")
print(f"   Exactitud: {accuracy:.2%}")

# ============================================================================
# 7. VISUALIZACIÓN
# ============================================================================

print("\n" + "=" * 80)
print("📊 GENERANDO VISUALIZACIONES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Clusters detectados por DBSCAN
ax1 = axes[0, 0]
scatter1 = ax1.scatter(X[labels != -1, 1], X[labels != -1, 0], 
                       c=labels[labels != -1], cmap='viridis', 
                       s=50, alpha=0.6, label='Clusters normales')
ax1.scatter(X[labels == -1, 1], X[labels == -1, 0], 
           c='red', marker='X', s=200, alpha=0.8, label='Anomalías detectadas')
ax1.set_xlabel('Longitud')
ax1.set_ylabel('Latitud')
ax1.set_title('DBSCAN: Clusters y Anomalías Detectadas')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Anomalías reales vs detectadas
ax2 = axes[0, 1]
ax2.scatter(X[y_true != -1, 1], X[y_true != -1, 0], 
           c='blue', s=50, alpha=0.6, label='Puntos normales (real)')
ax2.scatter(X[y_true == -1, 1], X[y_true == -1, 0], 
           c='red', marker='X', s=200, alpha=0.8, label='Anomalías reales')
ax2.set_xlabel('Longitud')
ax2.set_ylabel('Latitud')
ax2.set_title('Anomalías Reales en el Dataset')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Matriz de confusión
ax3 = axes[1, 0]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_anomalies, predicted_anomalies)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Normal', 'Anomalía'],
            yticklabels=['Normal', 'Anomalía'])
ax3.set_ylabel('Verdadero')
ax3.set_xlabel('Predicho')
ax3.set_title('Matriz de Confusión')

# Plot 4: K-distance graph
ax4 = axes[1, 1]
ax4.plot(distances, linewidth=0.5)
ax4.axhline(y=EPS, color='r', linestyle='--', label=f'eps={EPS}')
ax4.axhline(y=np.percentile(distances, 90), color='g', linestyle='--', 
           label=f'Percentil 90: {np.percentile(distances, 90):.3f}')
ax4.set_xlabel('Punto (ordenado)')
ax4.set_ylabel('Distancia a k-ésimo vecino (km)')
ax4.set_title('K-distance Graph (MIN_SAMPLES=5)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/dbscan_anomalias_geo.png', dpi=150, bbox_inches='tight')
print("\n✅ Gráficos guardados: dbscan_anomalias_geo.png")

# ============================================================================
# 8. EXPORTAR RESULTADOS PARA MAPBOX
# ============================================================================

print("\n" + "=" * 80)
print("📍 GENERANDO GeoJSON PARA MAPBOX")
print("=" * 80)

def generar_geojson(X: np.ndarray, labels: np.ndarray) -> Dict:
    """Genera GeoJSON compatible con Mapbox GL JS"""
    
    features = []
    for i, (lat, lon) in enumerate(X):
        label = labels[i]
        
        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "cluster": int(label) if label != -1 else "anomalia",
                "tipo": "Anomalía" if label == -1 else "Normal",
                "latitude": float(lat),
                "longitude": float(lon)
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)]
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

geojson_data = generar_geojson(X, labels)

# Guardar GeoJSON
with open('/home/claude/anomalias_geoespaciales.geojson', 'w') as f:
    json.dump(geojson_data, f, indent=2)

print(f"\n✅ GeoJSON generado: anomalias_geoespaciales.geojson")
print(f"   Total de features: {len(geojson_data['features'])}")
print(f"   Anomalías: {sum(1 for f in geojson_data['features'] if f['properties']['tipo'] == 'Anomalía')}")

# ============================================================================
# 9. EJEMPLO DE CÓDIGO MAPBOX GL JS
# ============================================================================

print("\n" + "=" * 80)
print("💻 CÓDIGO MAPBOX GL JS")
print("=" * 80)

mapbox_code = """
// Código para visualizar en Mapbox GL JS
const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: [-100.4, 20.7],  // Centro Querétaro
    zoom: 8
});

// Cargar datos GeoJSON
map.on('load', function() {
    map.addSource('anomalies', {
        type: 'geojson',
        data: 'anomalias_geoespaciales.geojson'
    });
    
    // Capa para puntos normales
    map.addLayer({
        id: 'normal-points',
        type: 'circle',
        source: 'anomalies',
        filter: ['!=', ['get', 'cluster'], 'anomalia'],
        paint: {
            'circle-radius': 6,
            'circle-color': [
                'interpolate',
                ['linear'],
                ['get', 'cluster'],
                0, '#1f77b4',
                1, '#ff7f0e',
                2, '#2ca02c',
                3, '#d62728'
            ],
            'circle-opacity': 0.7
        }
    });
    
    // Capa para anomalías
    map.addLayer({
        id: 'anomaly-points',
        type: 'circle',
        source: 'anomalies',
        filter: ['==', ['get', 'cluster'], 'anomalia'],
        paint: {
            'circle-radius': 8,
            'circle-color': '#FF0000',
            'circle-stroke-width': 2,
            'circle-stroke-color': '#FFF'
        }
    });
    
    // Popup al hacer clic
    map.on('click', 'anomaly-points', function(e) {
        const properties = e.features[0].properties;
        new mapboxgl.Popup()
            .setLngLat(e.lngLat)
            .setHTML(`<strong>ANOMALÍA</strong><br/>
                     Lat: ${properties.latitude.toFixed(4)}<br/>
                     Lon: ${properties.longitude.toFixed(4)}`)
            .addTo(map);
    });
});
"""

print(mapbox_code)

# ============================================================================
# 10. ANÁLISIS DE DENSIDAD
# ============================================================================

print("\n" + "=" * 80)
print("📊 ANÁLISIS DE DENSIDAD")
print("=" * 80)

# Calcular densidad local de cada punto
neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES, metric=METRIC)
X_rad = np.radians(X)
neighbors.fit(X_rad)
distances_to_neighbors, _ = neighbors.kneighbors(X_rad)

# Densidad = 1 / distancia promedio
max_distances = distances_to_neighbors[:, -1] * 6371  # convertir a km
densidad = 1.0 / (max_distances + 1e-10)

print(f"\n📈 Estadísticas de densidad local:")
print(f"   Densidad promedio (normal): {densidad[normal_indices].mean():.4f}")
print(f"   Densidad promedio (anomalías): {densidad[anomaly_indices].mean():.4f}")
print(f"   Densidad mínima: {densidad.min():.4f}")
print(f"   Densidad máxima: {densidad.max():.4f}")

# Puntos con densidad muy baja son anomalías
densidad_threshold = np.percentile(densidad, 10)
print(f"\n   Threshold de baja densidad (percentil 10): {densidad_threshold:.4f}")
print(f"   Puntos bajo threshold: {(densidad < densidad_threshold).sum()}")

# ============================================================================
# 11. TABLA DE RESULTADOS
# ============================================================================

print("\n" + "=" * 80)
print("📋 TABLA DE RESULTADOS PRINCIPALES")
print("=" * 80)

print(f"""
┌────────────────────────────────┬─────────────┐
│          Métrica               │   Valor     │
├────────────────────────────────┼─────────────┤
│ Total de puntos                │ {len(X):>11} │
│ Clusters detectados            │ {n_clusters:>11} │
│ Anomalías detectadas           │ {n_anomalies:>11} │
│ Tasa de anomalías             │ {100*n_anomalies/len(X):>10.1f}% │
├────────────────────────────────┼─────────────┤
│ Precisión en detección         │ {100*precision:>10.1f}% │
│ Recall (Sensibilidad)          │ {100*recall:>10.1f}% │
│ F1-Score                       │ {f1:>11.4f} │
│ Exactitud total                │ {100*accuracy:>10.1f}% │
├────────────────────────────────┼─────────────┤
│ eps (parámetro)                │ {EPS:>11.3f} km │
│ min_samples (parámetro)        │ {MIN_SAMPLES:>11} │
│ métrica (parámetro)            │ {METRIC:>11} │
└────────────────────────────────┴─────────────┘
""")

# ============================================================================
# 12. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("📝 RESUMEN")
print("=" * 80)

summary = f"""
✅ Implementación completada: DBSCAN para Anomalías Geoespaciales

📊 Características de DBSCAN:
   ✓ Detección automática de anomalías (outliers)
   ✓ No requiere número de clusters pre-especificado
   ✓ Ideal para datos con distribuciones no-convexas
   ✓ Métrica haversine para coordenadas lat-lon

🎯 Parámetros utilizados:
   • eps = {EPS} km (radio de búsqueda)
   • min_samples = {MIN_SAMPLES} (densidad mínima)
   • metric = '{METRIC}' (distancia geodésica)

📈 Rendimiento:
   • Clusters identificados: {n_clusters}
   • Anomalías detectadas: {n_anomalies} ({100*n_anomalies/len(X):.1f}%)
   • Precisión: {100*precision:.1f}%
   • Recall: {100*recall:.1f}%
   • F1-Score: {f1:.4f}

🗺️ Salidas generadas:
   1. dbscan_anomalias_geo.png - Visualizaciones
   2. anomalias_geoespaciales.geojson - Datos para Mapbox
   3. Código Mapbox GL JS para visualización interactiva

💡 Ventajas vs K-means:
   • Encuentra clusters de cualquier forma (no solo esféricos)
   • Detecta automáticamente outliers
   • No necesita especificar k
   • Mejor para datos geoespaciales con métrica haversine

🚀 Casos de uso:
   • Detección de movimientos anómalos en rutas GPS
   • Identificación de ubicaciones con baja actividad
   • Clustering de puntos de interés en ciudades
   • Análisis de flujos de tráfico y anomalías
   • Detección de fraude en transacciones geolocalización

⚙️ Ajustes recomendados:
   1. Aumentar eps si hay muchos falsos negativos
   2. Aumentar min_samples si hay muchos falsos positivos
   3. Usar k-distance graph para estimar eps óptimo
   4. Normalizar datos si tienen diferentes escalas
"""

print(summary)

print("=" * 80)
print("¡Proceso completado exitosamente!")
print("=" * 80)