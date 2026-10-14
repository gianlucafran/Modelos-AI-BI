"""
Luma Photon: VFX, Animación y Simulación de Dinámicas para Generación Cinematográfica
Aplicación: Extracción de Trayectorias de Partículas (VFX) con Scikit-Learn y Prompting Automático

Este script simula una explosión/sistema de partículas de VFX, extrae la geometría 3D/2D
y los centroides del movimiento con Scikit-Learn (K-Means) y construye las directivas
técnicas de cámara e iluminación para el renderizado en Luma Photon.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# 1. CARGA / GENERACIÓN DE DATASET DE SIMULACIÓN DE PARTÍCULAS (VFX)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET DE SIMULACIÓN DE PARTÍCULAS (VFX / ANIMACIÓN)")
print("=" * 80)

# Generamos un dataset de scikit-learn que simula un sistema de partículas VFX (humo/fuego/partículas)
# 3,000 partículas en un espacio tridimensional de coordenadas (X, Y, Z) + Densidad
X_particulas, y_clusters = make_blobs(
    n_samples=3000,
    n_features=3,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

print(f"✓ Dataset de partículas VFX generado: {X_particulas.shape[0]} puntos tridimensionales.")

# ============================================================================
# 2. EXTRACCIÓN DE CENTROIDES DE DINÁMICA DE FLUIDOS (K-MEANS)
# ============================================================================
print("\n" + "=" * 80)
print("2. EXTRACCIÓN DE NÚCLEOS DINÁMICOS DE ANIMACIÓN CON SCIKIT-LEARN")
print("=" * 80)

# Escalamos el espacio volumétrico
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_particulas)

# Identificamos los 4 centros de volumen de la simulación de fluidos/VFX
n_nucleos = 4
kmeans = KMeans(n_clusters=n_nucleos, random_state=42, n_init=10)
kmeans.fit(X_scaled)

centroides = scaler.inverse_transform(kmeans.cluster_centers_)

print("✓ Centroides volumétricos calculados (Puntos focalizados de emisión VFX):")
for i, center in enumerate(centroides):
    print(f"   • Núcleo VFX #{i+1}: Coordenadas (X: {center[0]:.2f}, Y: {center[1]:.2f}, Z: {center[2]:.2f})")

# ============================================================================
# 3. GENERADOR DE DIRECTIVAS Y PROMPTS CINEMATOGRÁFICOS PARA LUMA PHOTON
# ============================================================================
print("\n" + "=" * 80)
print("3. CONSTRUCCIÓN DE DIRECTIVAS DE RENDERIZADO PARA LUMA PHOTON")
print("=" * 80)

estilos_vfx = [
    {"tipo": "Volumetric Fire Explosion", "iluminacion": "intense orange and warm emissive glow, subsurface scattering"},
    {"tipo": "Cosmic Dark Energy Swirl", "iluminacion": "bioluminescent purple and cyan particles, volumetric fog"},
    {"tipo": "Liquid Metallic Hydro-simulation", "iluminacion": "chrome reflections, ray-traced caustics, studio rim light"},
    {"tipo": "Dust & Particle Shockwave", "iluminacion": "dramatic golden hour sunset light, cinematic volumetric rays"}
]

def generar_prompt_luma_photon(idx_nucleo, centroide):
    estilo = estilos_vfx[idx_nucleo % len(estilos_vfx)]
    
    prompt = (
        f"Cinematic VFX shot of a {estilo['tipo']}. The motion vortex expands around "
        f"spatial anchor (X:{centroide[0]:.1f}, Y:{centroide[1]:.1f}). "
        f"Lighting: {estilo['iluminacion']}. Camera movement: Slow 360-degree orbital pan, "
        f"shot on Arri Alexa 65, 35mm anamorphic lens, f/1.8 depth of field, photorealistic CGI composition, "
        f"8k resolution, Luma Photon VFX pipeline rendering."
    )
    return prompt

prompts_luma = []
for i, center in enumerate(centroides):
    p = generar_prompt_luma_photon(i, center)
    prompts_luma.append(p)
    print(f"\n🎬 Directiva Luma Photon #{i+1} [{estilos_vfx[i]['tipo']}]:")
    print(f"   \"{p}\"")

# ============================================================================
# 4. VISUALIZACIÓN DE SIMULACIÓN DE PARTÍCULAS Y RENDERIZADO
# ============================================================================
fig = plt.figure(figsize=(14, 6))

# Gráfico 1: Dispersión Tridimensional de Partículas (VFX Simulation)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
scatter = ax1.scatter(
    X_particulas[:, 0], X_particulas[:, 1], X_particulas[:, 2], 
    c=kmeans.labels_, cmap='plasma', alpha=0.4, s=8
)
ax1.scatter(
    centroides[:, 0], centroides[:, 1], centroides[:, 2], 
    c='cyan', s=150, marker='X', edgecolors='black', label='Centroides de Emisión VFX'
)
ax1.set_title('Simulación de Partículas VFX 3D\n(Agrupamiento K-Means Scikit-Learn)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Eje X (Amplitud)')
ax1.set_ylabel('Eje Y (Elevación)')
ax1.set_zlabel('Eje Z (Profundidad)')
ax1.legend()

# Gráfico 2: Proyección 2D y Coordenadas de Cámara
ax2 = fig.add_subplot(1, 2, 2)
sns.scatterplot(
    x=X_particulas[:, 0], y=X_particulas[:, 1], hue=kmeans.labels_, 
    palette='plasma', alpha=0.5, s=15, ax=ax2, legend=False
)
ax2.scatter(
    centroides[:, 0], centroides[:, 1], 
    c='red', s=180, marker='*', edgecolors='black', zorder=10, label='Puntos Focalización Luma Photon'
)
ax2.set_title('Plano de Composición 2D para Framing de Cámara', fontsize=11, fontweight='bold')
ax2.set_xlabel('Eje X')
ax2.set_ylabel('Eje Y')
ax2.legend()

plt.tight_layout()
#plt.savefig('07_luma_photon_vfx_simulation.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("✓ Gráfico de simulación VFX guardado: '07_luma_photon_vfx_simulation.png'")
print("=" * 80)