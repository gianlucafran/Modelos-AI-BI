"""
Creación de Arte Conceptual (Estilo Midjourney) con Datasets de Scikit-Learn
Concepto: Extracción de Estilos Visuales (PCA) y Generación Automática de Prompts Estilizados

Este script analiza un dataset visual de Scikit-Learn (Labeled Faces in the Wild),
extrae componentes estéticos/texturales mediante PCA (Eigenfaces) y construye
prompts parametrizados para la creación de arte conceptual estilo Midjourney.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

# Set gráfico profesional
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# 1. CARGA DEL DATASET REAL DE SCIKIT-LEARN (LFW People)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET REAL DE SCIKIT-LEARN (Labeled Faces in the Wild)")
print("=" * 80)

# Cargar dataset de imágenes reales de personas/rostros
lfw_dataset = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
n_samples, h, w = lfw_dataset.images.shape
X = lfw_dataset.data
n_features = X.shape[1]

print(f"✓ Dataset cargado: {n_samples} imágenes con dimensión {h}x{w} píxeles ({n_features} características).")

# ============================================================================
# 2. EXTRAER COMPONENTES ESTÉTICAS / TEXTURALES (EIGENFACES) CON PCA
# ============================================================================
print("\n" + "=" * 80)
print("2. EXTRAENDO PATRONES ESTÉTICOS Y MAPAS LATENTES CON PCA")
print("=" * 80)

n_components = 6
pca = PCA(n_components=n_components, whiten=True, random_state=42).fit(X)
eigenfaces = pca.components_.reshape((n_components, h, w))

print(f"✓ Varianza explicada acumulada por los {n_components} vectores estéticos: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

# Mapeo de vectores latentes a conceptos de arte estético estilo Midjourney
estilos_conceptuales = [
    {"estilo": "Cyberpunk Neon Portrait", "paleta": "cyan and magenta neon lighting, volumetric atmosphere", "medio": "digital concept art, octane render 8k"},
    {"estilo": "Dark Fantasy Chiaroscuro", "paleta": "dramatic chiaroscuro lighting, deep shadows, amber tones", "medio": "oil painting style, masterpiece"},
    {"estilo": "Futuristic Minimalist Sci-Fi", "paleta": "clean white background, soft ambient occlusion, metallic accents", "medio": "3d character design, unreal engine 5"},
    {"estilo": "Surrealist Impressionism", "paleta": "vibrant pastel colors, brush stroke textures, ethereal light", "medio": "concept art, detailed matte painting"},
    {"estilo": "Vintage Noir Charcoal", "paleta": "monochrome high contrast, gritty texture, atmospheric fog", "medio": "charcoal sketch, cinematic framing"},
    {"estilo": "Biomorphic Organic Art", "paleta": "emerald green and iridescent gold, bio-luminescent glow", "medio": "fantasy concept art, trending on artstation"}
]

# ============================================================================
# 3. GENERADOR DE PROMPTS PARAMETRIZADOS PARA MIDJOURNEY v6
# ============================================================================
def generar_midjourney_prompt(idx_estilo, sujeto_base="cybernetic portrait of a guardian"):
    """
    Combina características latentes del dataset de scikit-learn con parámetros
    avanzados de Midjourney (--v 6.0, --ar 16:9, --stylize, --chaos).
    """
    meta_estilo = estilos_conceptuales[idx_estilo % len(estilos_conceptuales)]
    var_exp = pca.explained_variance_ratio_[idx_estilo] * 100
    
    # Construcción del prompt refinado de Midjourney
    prompt_texto = (
        f"/imagine prompt: {sujeto_base}, {meta_estilo['estilo']}, "
        f"{meta_estilo['paleta']}, {meta_estilo['medio']}, highly detailed, "
        f"epic composition, cinematic lighting --ar 16:9 --v 6.0 --stylize {int(200 + var_exp * 10)} --chaos 15"
    )
    return prompt_texto, meta_estilo

print("\n" + "=" * 80)
print("3. GENERACIÓN DE PROMPTS DE ARTE CONCEPTUAL (ESTILO MIDJOURNEY)")
print("=" * 80)

prompts_generados = []
for i in range(n_components):
    prompt_mj, meta = generar_midjourney_prompt(i, sujeto_base="concept art of a space explorer")
    prompts_generados.append(prompt_mj)
    print(f"\n🎨 Concepto #{i+1} [{meta['estilo']}]:")
    print(f"   {prompt_mj}")

# ============================================================================
# 4. VISUALIZACIÓN DE COMPONENTES DE ESTILO Y MAPAS CONCEPTUALES
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Análisis Estético Latente (Scikit-Learn PCA) para Generación de Arte Conceptual Midjourney", fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    im = ax.imshow(eigenfaces[i], cmap='magma')
    ax.set_title(f"Mapa de Estilo #{i+1}\n{estilos_conceptuales[i]['estilo']}", fontsize=10, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('midjourney_concept_art_style_map.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("✓ Gráfico de mapas latentes de estilo guardado: 'midjourney_concept_art_style_map.png'")
print("=" * 80)