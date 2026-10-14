"""
Apriori: Análisis de la Canasta de Compra (Market Basket Analysis)
Generación de Recomendaciones para Carrito de Compras
Demostración con dataset sintético (make_multilabel_classification) de scikit-learn

Conceptos:
- Algoritmo Apriori: Extrae conjuntos de ítems frecuentes en grandes bases de datos transaccionales.
- Soporte (Support): Frecuencia con la que aparece un conjunto de ítems en todas las transacciones.
- Confianza (Confidence): Probabilidad de que un cliente compre el ítem B dado que ya compró el ítem A.
- Lift (Sustento/Elevación): Mide la fuerza de la asociación. Lift > 1 indica que los ítems se 
  compran juntos más de lo que se esperaría por puro azar.
- Reglas de Asociación: Expresiones del tipo "Si compra A, entonces recomendar B".

Nota: Scikit-learn no incluye implementaciones nativas de Apriori, por lo que usaremos
el estándar de la industria 'mlxtend' para el algoritmo, y scikit-learn para simular 
las transacciones correlacionadas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors  # <--- Solución al error
from sklearn.datasets import make_multilabel_classification
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')

# Nota: Requiere instalación previa de mlxtend (pip install mlxtend)
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    print("Por favor, instala mlxtend usando: pip install mlxtend")
    import sys
    sys.exit()

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. GENERAR Y EXPLORAR EL DATASET (SIMULACIÓN DE TRANSACCIONES)
# ============================================================================
print("=" * 80)
print("PASO 1: SIMULAR DATASET DE TRANSACCIONES (CARRITO DE COMPRA)")
print("=" * 80)

# Utilizamos make_multilabel_classification para simular "carritos" donde 
# un cliente puede tener múltiples productos (etiquetas) al mismo tiempo, 
# existiendo correlación entre algunas compras.
n_transacciones = 1000
n_productos = 8

# Generamos la matriz binaria Y (Transacciones x Productos)
_, Y = make_multilabel_classification(
    n_samples=n_transacciones, 
    n_classes=n_productos,
    n_labels=3,           # Promedio de 3 productos por carrito
    allow_unlabeled=False, 
    random_state=42
)

# Mapeamos las clases generadas a productos reales de supermercado
nombres_productos = [
    'Pan', 'Leche', 'Huevos', 'Mantequilla', 
    'Café', 'Azúcar', 'Cereal', 'Fruta'
]

df_compras = pd.DataFrame(Y, columns=nombres_productos)

# Convertimos a tipo booleano, formato requerido por mlxtend para mayor eficiencia
df_compras = df_compras.astype(bool)

print(f"\nDataset: Transacciones de Supermercado")
print(f"Total de Tickets (Transacciones): {df_compras.shape[0]}")
print(f"Catálogo de Productos: {df_compras.shape[1]}")

print(f"\nPrimeros 5 carritos de compra (True = Comprado, False = No Comprado):")
print(df_compras.head().astype(int))  # Mostramos como 1/0 para facilitar lectura

# Frecuencia de productos individuales
frecuencias = df_compras.mean().sort_values(ascending=False) * 100
print("\nFrecuencia de compra por producto (% de tickets):")
for prod, frec in frecuencias.items():
    print(f"  - {prod}: {frec:.1f}%")

# ============================================================================
# 2. ALGORITMO APRIORI (ITEMSETS FRECUENTES)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: APLICAR APRIORI (EXTRACCIÓN DE CONJUNTOS FRECUENTES)")
print("=" * 80)

# min_support = 0.15 significa que el conjunto de ítems debe aparecer en al menos
# el 15% de todas las transacciones para ser considerado "frecuente".
min_sup = 0.15
print(f"\nEjecutando Apriori buscando combinaciones con Soporte mínimo de {min_sup*100}%...")

itemsets_frecuentes = apriori(df_compras, min_support=min_sup, use_colnames=True)

# Añadimos la longitud del itemset para facilitar el filtrado
itemsets_frecuentes['longitud'] = itemsets_frecuentes['itemsets'].apply(lambda x: len(x))

print(f"✓ Se encontraron {len(itemsets_frecuentes)} conjuntos frecuentes.")

print("\nTop 5 Combinaciones de productos más compradas juntas (Tamaño >= 2):")
combinaciones = itemsets_frecuentes[itemsets_frecuentes['longitud'] >= 2].sort_values(by='support', ascending=False)
for _, row in combinaciones.head(5).iterrows():
    items = ", ".join(list(row['itemsets']))
    print(f"  - [{items}] (Soporte: {row['support']*100:.1f}%)")

# ============================================================================
# 3. GENERACIÓN DE REGLAS DE ASOCIACIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: GENERAR REGLAS DE ASOCIACIÓN Y MÉTRICAS")
print("=" * 80)

# A partir de los itemsets, generamos las reglas. 
# Usaremos 'lift' como métrica principal para filtrar. 
# Lift > 1.2 asegura que la asociación es significativamente mejor que el azar.
reglas = association_rules(itemsets_frecuentes, metric="lift", min_threshold=1.2)

# Ordenar las reglas por Lift y Confianza
reglas = reglas.sort_values(['lift', 'confidence'], ascending=[False, False])

print(f"\nSe generaron {len(reglas)} reglas de asociación fuertes (Lift > 1.2).")

print("\nTop 5 Reglas de Asociación más fuertes:")
for idx, row in reglas.head().iterrows():
    antecedente = ", ".join(list(row['antecedents']))
    consecuente = ", ".join(list(row['consequents']))
    print(f"  Si compra [{antecedente}] -> Recomendar [{consecuente}]")
    print(f"     | Confianza: {row['confidence']*100:.1f}% | Lift: {row['lift']:.2f}")

# ============================================================================
# 4. SISTEMA DE RECOMENDACIÓN PRÁCTICO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: SIMULADOR DE MOTOR DE RECOMENDACIÓN E-COMMERCE")
print("=" * 80)

def recomendar_productos(carrito_actual, dataframe_reglas, max_recomendaciones=3):
    """
    Función que simula el motor de recomendación en tiempo real.
    Busca reglas donde el 'antecedente' esté incluido en el carrito actual.
    """
    carrito_set = frozenset(carrito_actual)
    
    # Filtrar reglas donde el antecedente es un subconjunto del carrito actual
    recomendaciones = dataframe_reglas[
        dataframe_reglas['antecedents'].apply(lambda x: x.issubset(carrito_set))
    ]
    
    if recomendaciones.empty:
        return ["No hay recomendaciones cruzadas fuertes."]
    
    # Filtrar productos que ya están en el carrito
    recomendaciones = recomendaciones[
        ~recomendaciones['consequents'].apply(lambda x: x.issubset(carrito_set))
    ]
    
    # Extraer los productos recomendados ordenados por Lift
    productos_recom = []
    for consecuente in recomendaciones['consequents']:
        for item in consecuente:
            if item not in productos_recom and item not in carrito_actual:
                productos_recom.append(item)
                
    return productos_recom[:max_recomendaciones]

# Simulamos algunos carritos
carritos_prueba = [
    ['Pan', 'Mantequilla'],
    ['Café'],
    ['Leche', 'Cereal']
]

print("\nProbando el motor de recomendación:")
for carrito in carritos_prueba:
    recs = recomendar_productos(carrito, reglas)
    print(f"🛒 Carrito actual: {carrito}")
    print(f"💡 Se recomienda agregar: {recs}\n")

# ============================================================================
# 5. VISUALIZACIÓN: ESPACIO DE REGLAS DE ASOCIACIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DE MÉTRICAS (SUPPORT, CONFIDENCE, LIFT)")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Scatter plot de Support vs Confidence
scatter = axes[0].scatter(
    reglas['support'], 
    reglas['confidence'], 
    c=reglas['lift'], 
    cmap='viridis', 
    s=reglas['lift'] * 80,  # El tamaño del punto también indica el Lift
    alpha=0.8,
    edgecolor='black'
)
axes[0].set_title('Reglas de Asociación: Soporte vs Confianza\n(Color y tamaño = Lift)', 
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Soporte (Frecuencia de la regla)')
axes[0].set_ylabel('Confianza (Probabilidad condicional)')
axes[0].grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Lift (Fuerza de Asociación)')

# Gráfico 2: Heatmap de Co-ocurrencias (Correlación cruzada)
# Para esto calculamos la matriz de correlación directa (Phi coefficient)
matriz_corr = pd.DataFrame(Y, columns=nombres_productos).corr()

sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, ax=axes[1])
axes[1].set_title('Mapa de Calor: Correlación entre Productos\n(Visión Global del Catálogo)', 
                  fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/usuario/01_apriori_reglas.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_apriori_reglas.png")
plt.close()

# ============================================================================
# 6. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               APRIORI PARA RECOMENDACIONES DE CARRITO DE COMPRA              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. LÓGICA DEL ALGORITMO:
   • Apriori es un algoritmo de minería de reglas de asociación. 
   • Opera bajo el "Principio Apriori": Si un conjunto de ítems es frecuente, 
     todos sus subconjuntos también deben serlo. Esto reduce drásticamente 
     el tiempo de cómputo en catálogos gigantes.

2. LAS TRES MÉTRICAS FUNDAMENTALES (EJEMPLO: Si Pan -> Mantequilla):
   • SOPORTE (Support): ¿Qué porcentaje de los tickets totales incluyen 
     ambos productos? (Nos dice si vale la pena comercialmente).
   • CONFIANZA (Confidence): De todos los que compraron Pan, ¿qué porcentaje 
     también llevó Mantequilla? (Nos da la direccionalidad de la regla).
   • LIFT: ¿Es esta compra conjunta una casualidad? 
     - Lift = 1: Son independientes.
     - Lift > 1: Efecto sinérgico (Comprar Pan empuja a comprar Mantequilla).
     - Lift < 1: Productos sustitutos o mutuamente excluyentes.

3. IMPACTO EN EL NEGOCIO (MARKET BASKET ANALYSIS):
   • Motores de Recomendación (Cross-Selling): Como se demostró en el Paso 4, 
     al añadir 'Café', el e-commerce puede sugerir automáticamente ítems 
     altamente correlacionados para aumentar el Ticket Promedio.
   • Diseño de Tiendas (Layout): Productos con alto Lift deben colocarse cerca 
     en góndolas o páginas de aterrizaje.
   • Promociones Conjuntas: "Lleva la Leche con un 10% de descuento si 
     compras Cereal".

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

apriori(df, min_support=0.15):
  - Exige que una combinación aparezca en el 15% de las ventas para ser analizada.

association_rules(metric="lift", min_threshold=1.2):
  - Filtro severo: Solo queremos reglas cuya asociación sea un 20% más fuerte 
    que comprar los productos al azar.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_apriori_reglas.png  - Visualización del espacio Support-Confidence-Lift 
                         y mapa de correlación de compras.

═══════════════════════════════════════════════════════════════════════════════
"""

