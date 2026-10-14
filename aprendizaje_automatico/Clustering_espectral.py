"""
Clustering Espectral: Segmentación de Redes Sociales y Detección de Comunidades
Demostración con el dataset REAL "Zachary's Karate Club" usando NetworkX y Scikit-Learn

Conceptos:
- Clustering Espectral: A diferencia de K-Means (que asume clusters esféricos basados en 
  distancia Euclidiana), el clustering espectral analiza la topología de los datos. 
  Es la técnica estándar del estado del arte para la detección de comunidades en grafos.
- Grafo y Matriz de Adyacencia: Una red social se representa como un grafo donde las 
  personas son Nodos y sus interacciones son Aristas. Esto se traduce a una matriz 
  cuadrada de NxN (Matriz de Adyacencia o Afinidad).
- El Laplaciano del Grafo: El corazón del algoritmo. Es una matriz (L = D - A, donde 
  D es la matriz de grado y A la de adyacencia). El algoritmo calcula los vectores 
  propios (eigenvectors) de esta matriz para reducir la dimensionalidad.
- Zachary's Karate Club: Un dataset clásico y REAL en sociología de redes (1977). 
  Documenta las interacciones de 34 miembros de un club de karate que, debido a un 
  conflicto, terminó fracturándose en dos facciones (lideradas por el "Instructor" y 
  el "Presidente" del club).
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. CARGAR EL DATASET (RED SOCIAL REAL)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR RED SOCIAL REAL (ZACHARY'S KARATE CLUB)")
print("=" * 80)

# Utilizamos NetworkX para cargar el dataset topológico
G = nx.karate_club_graph()

# Extraer información de la red
num_nodos = G.number_of_nodes()
num_aristas = G.number_of_edges()

print(f"Dataset: Club de Karate de Zachary (Red Social)")
print(f"Total de miembros (Nodos): {num_nodos}")
print(f"Interacciones / Amistades (Aristas): {num_aristas}")

# En este dataset histórico, sabemos exactamente cómo se dividió el club en la realidad.
# Extraemos las etiquetas reales ('Mr. Hi' = Instructor, 'Officer' = Presidente)
faccion_real = [G.nodes[i]['club'] for i in range(num_nodos)]
faccion_numerica_real = [0 if f == 'Mr. Hi' else 1 for f in faccion_real]

print("\nFacciones históricas documentadas (Ground Truth):")
print(f"  - Facción del Instructor (Mr. Hi): {faccion_real.count('Mr. Hi')} miembros")
print(f"  - Facción del Presidente (Officer): {faccion_real.count('Officer')} miembros")

# ============================================================================
# 2. PREPROCESAMIENTO: MATRIZ DE AFINIDAD
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: CONVERSIÓN A MATRIZ DE AFINIDAD (ADYACENCIA)")
print("=" * 80)

# Scikit-Learn requiere una matriz de afinidad para el Spectral Clustering.
# En el contexto de redes sociales, la matriz de adyacencia ES la matriz de afinidad.
# A_ij = 1 si hay conexión entre i y j, 0 de lo contrario.
matriz_adyacencia = nx.to_numpy_array(G)

print(f"✓ Grafo convertido a Matriz de Adyacencia de dimensiones {matriz_adyacencia.shape}")
print("Esta matriz será inyectada directamente al algoritmo de Scikit-Learn.")

# ============================================================================
# 3. ENTRENAMIENTO: CLUSTERING ESPECTRAL
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: APLICACIÓN DEL CLUSTERING ESPECTRAL (SCIKIT-LEARN)")
print("=" * 80)

# Configuramos el modelo de Scikit-Learn
# CRÍTICO: Usamos affinity='precomputed' porque nosotros le entregamos la red 
# directamente (la matriz de adyacencia). No queremos que Scikit-Learn intente 
# calcular distancias Euclidianas entre miembros.
spectral = SpectralClustering(
    n_clusters=2,             # Sabemos que el club se dividió en 2
    affinity='precomputed',   # Indica que la entrada ya es un grafo/matriz de afinidad
    n_init=100,               # Ejecuciones de K-Means en el espacio espectral
    assign_labels='kmeans',   # Cómo agrupar los vectores propios resultantes
    random_state=42
)

print("Calculando el Laplaciano del Grafo y extrayendo Vectores Propios...")
# Entrenamos y predecimos en un solo paso
etiquetas_predichas = spectral.fit_predict(matriz_adyacencia)

# Evaluar el rendimiento del algoritmo comparándolo con la historia real
# Adjusted Rand Index mide la similitud entre dos agrupamientos (1.0 es idéntico)
ari_score = adjusted_rand_score(faccion_numerica_real, etiquetas_predichas)

print(f"✓ Clustering Espectral completado.")
print(f"  - Precisión Topológica (Adjusted Rand Score): {ari_score:.4f}")
if ari_score > 0.8:
    print("  - ¡El algoritmo logró predecir casi a la perfección la fractura social del club\n    solo analizando las matemáticas de sus amistades!")

# ============================================================================
# 4. VISUALIZACIÓN: LA FRACTURA DE LA RED SOCIAL
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: VISUALIZACIÓN TOPOLÓGICA DE LA RED")
print("=" * 80)

# Generamos un layout (posición espacial de los nodos) para dibujar el grafo.
# El spring_layout simula resortes en las aristas para agrupar nodos conectados.
posiciones = nx.spring_layout(G, seed=42)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Colores para los clusters
color_map_real = ['#1f78b4' if f == 0 else '#e31a1c' for f in faccion_numerica_real]
# Asegurarnos de que los colores coincidan visualmente (el clustering no sabe qué es 0 o 1)
# Si están invertidos, los volteamos para la gráfica
if (etiquetas_predichas[0] != faccion_numerica_real[0]):
    etiquetas_predichas = 1 - etiquetas_predichas
color_map_predicho = ['#1f78b4' if f == 0 else '#e31a1c' for f in etiquetas_predichas]

# Gráfico 1: Lo que pasó en la realidad
axes[0].set_title('División Histórica (Ground Truth)\nFacción Azul (Instructor) vs Roja (Presidente)', 
                  fontsize=13, fontweight='bold')
nx.draw_networkx_edges(G, posiciones, alpha=0.5, ax=axes[0])
nx.draw_networkx_nodes(G, posiciones, node_color=color_map_real, node_size=500, 
                       edgecolors='black', linewidths=1.5, ax=axes[0])
nx.draw_networkx_labels(G, posiciones, font_size=10, font_color='white', font_weight='bold', ax=axes[0])
axes[0].axis('off')

# Gráfico 2: Lo que descubrió la Inteligencia Artificial
axes[1].set_title('Comunidades detectadas por Clustering Espectral\n(Inferencia matemática del Laplaciano)', 
                  fontsize=13, fontweight='bold')
nx.draw_networkx_edges(G, posiciones, alpha=0.5, ax=axes[1])
nx.draw_networkx_nodes(G, posiciones, node_color=color_map_predicho, node_size=500, 
                       edgecolors='black', linewidths=1.5, ax=axes[1])
nx.draw_networkx_labels(G, posiciones, font_size=10, font_color='white', font_weight='bold', ax=axes[1])
axes[1].axis('off')

# Encontrar los nodos donde el algoritmo se equivocó (si los hay)
diferencias = np.where(np.array(faccion_numerica_real) != np.array(etiquetas_predichas))[0]
if len(diferencias) > 0:
    nx.draw_networkx_nodes(G, posiciones, nodelist=diferencias.tolist(), 
                           node_color='yellow', node_size=600, edgecolors='red', 
                           linewidths=3, ax=axes[1], label='Error de Predicción')
    axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('/home/usuario/01_clustering_espectral_redes.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_clustering_espectral_redes.png")
plt.close()

# ============================================================================
# 5. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN TEÓRICO: CLUSTERING ESPECTRAL EN REDES SOCIALES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               CLUSTERING ESPECTRAL: MATEMÁTICAS DE LA SOCIOLOGÍA             ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. ¿POR QUÉ FALLARÍA K-MEANS AQUÍ?:
   • Si aplicaras K-Means sobre las coordenadas (x, y) de los nodos, cortaría 
     la red con una línea recta. 
   • Una red social es un concepto topológico, no euclidiano. La "distancia" 
     entre dos personas no es espacial, sino relacional (cuántos amigos en común 
     tienen). K-Means ignora la estructura de red; Spectral Clustering vive de ella.

2. LA SOLUCIÓN DEL ÁLGEBRA LINEAL (EL LAPLACIANO):
   • Al decirle a Scikit-Learn `affinity='precomputed'`, el algoritmo tomó 
     nuestra matriz de amistades (1s y 0s).
   • Calculó la Matriz Laplaciana (L = D - A), que describe cómo fluye la 
     información a través del grafo. 
   • Extrajo el "Fiedler Vector" (el segundo vector propio más pequeño de L). 
     El signo (+ o -) de los valores en este vector separa matemáticamente el 
     grafo en dos comunidades con la mínima cantidad de "cortes" (amistades rotas).

3. LA PREDICCIÓN TOPOLÓGICA (RESULTADOS):
   • El algoritmo analizó la estructura matemática sin saber nada de la historia 
     del conflicto del club de karate.
   • Logró identificar los clústeres reales casi con perfección, demostrando por 
     qué esta técnica es el estándar en Data Science para segmentación de 
     comunidades (Facebook, Twitter), análisis de clusters en grafos de 
     conocimiento y biología computacional.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_clustering_espectral_redes.png - Representación topológica de la red comparando 
                                    la realidad histórica vs el descubrimiento 
                                    no supervisado del algoritmo.
═══════════════════════════════════════════════════════════════════════════════
""")