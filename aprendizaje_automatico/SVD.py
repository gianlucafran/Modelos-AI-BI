"""
SVD: Filtrado Colaborativo y Sistemas de Recomendación Básicos
Demostración con dataset sintético (make_biclusters) de scikit-learn

Conceptos:
- SVD (Singular Value Decomposition): Reduce dimensionalidad y extrae características latentes.
- TruncatedSVD: Versión optimizada de SVD para matrices dispersas (esparcidas).
- Filtrado Colaborativo: Recomienda ítems basándose en patrones de usuarios similares.
- Matriz Usuario-Ítem: Estructura base donde filas son usuarios, columnas ítems y valores las calificaciones.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_biclusters
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. CARGAR Y EXPLORAR EL DATASET (SIMULACIÓN USUARIO-ÍTEM)
# ============================================================================
print("=" * 80)
print("PASO 1: GENERAR Y EXPLORAR EL DATASET")
print("=" * 80)

# Dado que sklearn no tiene un dataset de películas tipo Netflix, usamos 
# make_biclusters para generar una matriz con patrones latentes (grupos de 
# usuarios con gustos similares por ciertos grupos de películas).
n_users = 100
n_items = 20
n_clusters = 4  # Simularemos 4 "géneros" o perfiles de gusto

X, rows, cols = make_biclusters(
    shape=(n_users, n_items), 
    n_clusters=n_clusters, 
    noise=1.5, 
    random_state=42
)

# Escalar los valores simulados a un rango de calificaciones [1, 5]
X = np.interp(X, (X.min(), X.max()), (1, 5))

# En la realidad, las matrices de recomendación son "dispersas" (sparse).
# Los usuarios solo califican una pequeña fracción de los ítems disponibles.
# Simularemos esto dejando el 75% de la matriz como 0 (No calificado).
np.random.seed(42)
mask = np.random.rand(n_users, n_items) < 0.25
ratings_matrix = X * mask

# Crear DataFrames para mejor manejo
user_names = [f"Usuario_{i}" for i in range(n_users)]
item_names = [f"Item_{i}" for i in range(n_items)]
df_ratings = pd.DataFrame(ratings_matrix, index=user_names, columns=item_names)

print(f"\nDataset: Simulación de Calificaciones (make_biclusters)")
print(f"Número de Usuarios (filas): {df_ratings.shape[0]}")
print(f"Número de Ítems (columnas): {df_ratings.shape[1]}")
print(f"Grupos latentes ocultos inyectados: {n_clusters}")

print(f"\nPrimeras 5 filas de la Matriz Usuario-Ítem (0 = No calificado):")
print(df_ratings.iloc[:5, :8].round(1))  # Mostramos solo los primeros 8 ítems

# Calcular esparcidad
sparsity = 1.0 - (np.count_nonzero(ratings_matrix) / float(n_users * n_items))
print(f"\nEsparcidad de la matriz: {sparsity*100:.2f}% (Valores faltantes o ceros)")

# ============================================================================
# 2. DETERMINAR NÚMERO ÓPTIMO DE COMPONENTES (FACTORES LATENTES)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DETERMINAR COMPONENTES LATENTES (SVD)")
print("=" * 80)

# Exploramos la varianza explicada. TruncatedSVD es ideal para matrices con muchos ceros.
# Probaremos hasta 15 componentes.
svd_exploratory = TruncatedSVD(n_components=15, random_state=42)
svd_exploratory.fit(ratings_matrix)

explained_variance = svd_exploratory.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nVarianza explicada por cada factor latente (componente):")
for i, var in enumerate(explained_variance[:8], 1):
    print(f"  Factor {i}: {var:.4f} ({var*100:.2f}%)")

# Visualizar varianza explicada
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, 16), explained_variance, alpha=0.7, color='purple', edgecolor='black')
axes[0].set_xlabel('Factor Latente (Componente)', fontsize=12)
axes[0].set_ylabel('Varianza Explicada', fontsize=12)
axes[0].set_title('Varianza Explicada Individual', fontsize=13, fontweight='bold')
axes[0].set_xticks(range(1, 16))
axes[0].grid(axis='y', alpha=0.3)

axes[1].plot(range(1, 16), cumulative_variance, marker='o', color='darkorange', linewidth=2)
axes[1].axhline(y=0.85, color='red', linestyle='--', label='85% Varianza')
axes[1].set_xlabel('Número de Factores Latentes', fontsize=12)
axes[1].set_ylabel('Varianza Acumulada', fontsize=12)
axes[1].set_title('Varianza Acumulada', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(1, 16))
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/usuario/01_svd_varianza.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_svd_varianza.png")
plt.close()

# ============================================================================
# 3. APLICAR TRUNCATED SVD (REDUCCIÓN Y FACTORIZACIÓN)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: FACTORIZACIÓN DE MATRICES CON SVD")
print("=" * 80)

# Elegimos 4 componentes (ya que sabemos que inyectamos 4 clusters latentes)
# En la práctica, se elige el "codo" de la gráfica o un umbral (ej. 80% varianza)
n_latent_factors = 4
svd = TruncatedSVD(n_components=n_latent_factors, random_state=42)

# La matriz Original (R) se factoriza matemáticamente en SVD como: R ≈ U * Σ * V^T
# fit_transform nos da (U * Σ) que representa a los Usuarios en el espacio latente
user_factors = svd.fit_transform(ratings_matrix)

# components_ nos da (V^T) que representa a los Ítems en el espacio latente
item_factors = svd.components_

print(f"\nMatriz Original (Usuarios x Ítems): {ratings_matrix.shape}")
print(f"Matriz de Usuarios (Factores Latentes): {user_factors.shape}")
print(f"Matriz de Ítems (Factores Latentes): {item_factors.shape}")
print(f"Varianza total retenida: {np.sum(svd.explained_variance_ratio_)*100:.2f}%")

# ============================================================================
# 4. RECONSTRUCCIÓN: PREDICCIÓN DE CALIFICACIONES FALTANTES
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: RECONSTRUCCIÓN DE LA MATRIZ (PREDICCIONES)")
print("=" * 80)

# Al multiplicar las matrices factorizadas, reconstruimos la matriz original
# PERO rellenando los espacios vacíos (0) con las predicciones del modelo
predicted_ratings = np.dot(user_factors, item_factors)

# Escalamos un poco para mantener el rango de 1 a 5 estrellas
predicted_ratings = np.clip(predicted_ratings, 1, 5)

df_predictions = pd.DataFrame(predicted_ratings, index=user_names, columns=item_names)

print("\nMatriz Reconstruida (Predicciones para los primeros 5 usuarios e ítems):")
print(df_predictions.iloc[:5, :5].round(2))

# ============================================================================
# 5. VISUALIZACIÓN: MATRIZ ORIGINAL VS PREDICCIONES
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DE MATRICES")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Matriz Original (Dispersa)
sns.heatmap(df_ratings.iloc[:20, :], cmap='YlGnBu', ax=axes[0], 
            cbar_kws={'label': 'Calificación (0 = Vacío)'})
axes[0].set_title('Matriz Original (Esparsa / Con vacíos)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Ítems (Películas/Productos)')
axes[0].set_ylabel('Usuarios')

# Matriz Reconstruida (Densa)
sns.heatmap(df_predictions.iloc[:20, :], cmap='YlGnBu', ax=axes[1],
            cbar_kws={'label': 'Calificación Predicha'})
axes[1].set_title(f'Matriz Reconstruida por SVD (Predicciones)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Ítems (Películas/Productos)')
axes[1].set_ylabel('Usuarios')

plt.tight_layout()
plt.savefig('/home/usuario/02_reconstruccion_svd.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico guardado: 02_reconstruccion_svd.png")
plt.close()

# ============================================================================
# 6. SISTEMA DE RECOMENDACIÓN: EJECUCIÓN PRÁCTICA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: GENERADOR DE RECOMENDACIONES")
print("=" * 80)

def recomendar_items(usuario_idx, num_recomendaciones=3):
    """Recomienda ítems que el usuario no ha calificado aún."""
    usuario_nombre = df_ratings.index[usuario_idx]
    
    # Calificaciones originales del usuario
    calificaciones_reales = df_ratings.iloc[usuario_idx]
    
    # Predicciones hechas por el modelo SVD
    calificaciones_predichas = df_predictions.iloc[usuario_idx]
    
    # Identificar ítems que el usuario NO ha visto (donde la matriz original es 0)
    items_no_vistos = calificaciones_reales[calificaciones_reales == 0].index
    
    # Ordenar las predicciones solo para los ítems no vistos
    recomendaciones = calificaciones_predichas[items_no_vistos].sort_values(ascending=False)
    
    print(f"\n--- RECOMENDACIONES PARA: {usuario_nombre} ---")
    print("\nÍtems que el usuario calificó más alto originalmente:")
    vistos_top = calificaciones_reales[calificaciones_reales > 0].sort_values(ascending=False).head(3)
    for idx, (item, calif) in enumerate(vistos_top.items(), 1):
        print(f"  {idx}. {item}: {calif:.1f} estrellas")
        
    print(f"\nTop {num_recomendaciones} Nuevos Ítems Recomendados (Según SVD):")
    top_recs = recomendaciones.head(num_recomendaciones)
    for idx, (item, score) in enumerate(top_recs.items(), 1):
        print(f"  {idx}. {item}: Predicción de {score:.2f} estrellas")

# Probamos el sistema con un usuario aleatorio
recomendar_items(usuario_idx=15, num_recomendaciones=3)

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SVD EN SISTEMAS DE RECOMENDACIÓN                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. MANEJO DE ESPARCIDAD:
   • La matriz original tenía un {sparsity*100:.1f}% de valores vacíos (0s).
   • SVD logró encontrar patrones subyacentes a pesar de la gran falta de datos.
   • El resultado es una matriz completamente densa con predicciones.

2. FACTORES LATENTES (Varianza):
   • Dataset original: {n_items} dimensiones (ítems).
   • Modelo reducido: {n_latent_factors} dimensiones (factores latentes o "géneros").
   • Varianza retenida: {np.sum(svd.explained_variance_ratio_)*100:.2f}%.
   • Cada componente representa perfiles ocultos de gustos (ej. amantes de acción).

3. CÓMO FUNCIONA EL FILTRADO COLABORATIVO A TRAVÉS DE SVD:
   ✓ U (user_factors): Ubica a cada usuario en el mapa de gustos.
   ✓ V^T (item_factors): Ubica a cada producto en el mismo mapa.
   ✓ El Producto Punto (U • V^T) mide la afinidad entre un usuario y todos 
     los ítems, revelando su posible calificación para productos no vistos.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

make_biclusters(shape=(100, 20), n_clusters=4):
  - Inyecta clusters subyacentes. Representa perfectamente el problema de 
    filtrado colaborativo, simulando perfiles de usuarios.

TruncatedSVD(n_components=4):
  - Variante de PCA enfocada en matrices esparsas.
  - No centra los datos previamente (fundamental para preservar la 
    esparcidad, a diferencia del PCA tradicional).
  - random_state=42: Para reproducibilidad.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_svd_varianza.png        - Selección del número de factores latentes
02_reconstruccion_svd.png  - Heatmap comparativo (Original vs Predicciones)

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")