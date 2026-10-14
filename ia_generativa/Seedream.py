"""
Pipeline de Concept Art Creativo: Clustering Temático con Scikit-Learn
y Generación de Conceptos Visuales con Seedream 5.0 Pro / Seedream 4.0 API.
"""

import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt

# 1. SCIKIT-LEARN: CARGA DE DATASET REAL Y PROCESAMIENTO DE CONCEPTOS
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# ==========================================================
# 1. EXTRACCIÓN Y CLUSTERING DE CONCEPTOS CON SCIKIT-LEARN
# ==========================================================
def extraer_conceptos_creativos():
    """
    Carga el dataset '20 newsgroups' de scikit-learn, extrae las características 
    semánticas clave con TF-IDF y agrupa los temas con KMeans para generar 
    direcciones de arte diferenciadas.
    """
    print("=== 1. Cargando Dataset Real '20 Newsgroups' desde Scikit-Learn ===")
    
    # Seleccionar categorías de Ciencia y Tecnología para direcciones visuales
    categorias = ['sci.space', 'comp.graphics', 'sci.electronics']
    dataset = fetch_20newsgroups(subset='train', categories=categorias, remove=('headers', 'footers', 'quotes'))
    
    print(f"Documentos cargados: {len(dataset.data)}")

    # Vectorización de texto (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(dataset.data)

    # Reducción de dimensionalidad (Latent Semantic Analysis - LSA)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    # Clustering de conceptos con KMeans
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_reduced)

    # Identificar palabras clave principales por cada Cluster
    terms = vectorizer.get_feature_names_out()
    cluster_concepts = []
    
    # Mapeo de estilos visuales sugeridos según el cluster
    estilos_artisticos = [
        "Sci-Fi Cinematic Concept Art, Cyberpunk aesthetic, volumetric lighting, Octane Render",
        "Minimalist Architectural Render, Studio lighting, isometric view, soft shadows, 8k",
        "Retro-Futurism Graphic Art, vibrant neon colors, detailed textures, cinematic depth"
    ]

    for idx, cluster_id in enumerate(range(n_clusters)):
        # Obtener los términos principales del espacio SVD / TF-IDF
        original_space_centroids = svd.inverse_transform(kmeans.cluster_centers_[cluster_id])
        top_term_indices = original_space_centroids.argsort()[-6:][::-1]
        top_keywords = [terms[i] for i in top_term_indices]
        
        keywords_str = ", ".join(top_keywords)
        cluster_concepts.append({
            "cluster_id": cluster_id,
            "keywords": keywords_str,
            "style_direction": estilos_artisticos[idx]
        })

    return pd.DataFrame(cluster_concepts)


# ==========================================================
# 2. CLIENTE DE SEEDREAM AI API (ByteDance Seedream Model)
# ==========================================================
class SeedreamAPIClient:
    """
    Cliente Python para la API de Seedream AI (ByteDance).
    Soporta configuraciones de resolución (1K/2K/4K), relación de aspecto (16:9, 1:1)
    y generación guiada de arte conceptual.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("SEEDREAM_API_KEY") or os.getenv("SEGMIND_API_KEY")
        # Endpoint de integración estándar REST de Seedream
        self.endpoint_url = "https://api.segmind.com/v1/seedream-4"

    def generar_concepto_art(self, prompt, aspect_ratio="16:9", resolution="2K"):
        """
        Envía el prompt creativo a la API de Seedream para renderizar el concept art.
        """
        if not self.api_key:
            # Modo Simulación cuando no hay llaves de API activas
            return {
                "status": "Simulated Success (Seedream API)",
                "image_url": "https://seedream-cdn.bytedance.com/samples/concept_art_render.png",
                "prompt_used": prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution
            }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "size": resolution,
            "samples": 1
        }

        try:
            response = requests.post(self.endpoint_url, json=payload, headers=headers)
            response.raise_for_status()
            res_data = response.json()
            
            return {
                "status": "Success",
                "image_url": res_data.get("image_url") or res_data.get("data", [{}])[0].get("url"),
                "prompt_used": prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution
            }
        except Exception as e:
            print(f"Error llamando a Seedream API: {e}")
            return None


# ==========================================================
# 3. PIPELINE DE EJECUCIÓN DEL FLUJO CREATIVO
# ==========================================================
if __name__ == "__main__":
    # Step 1: Agrupar conceptos del dataset real con Scikit-Learn
    df_conceptos = extraer_conceptos_creativos()
    print("\n=== Conceptos Creativos Extraídos ===")
    print(df_conceptos[['cluster_id', 'keywords', 'style_direction']])

    # Step 2: Inicializar cliente de Seedream
    seedream_client = SeedreamAPIClient()

    # Step 3: Construir prompts y generar el Concept Art
    print("\n=== 2. Generando Concept Art con Seedream AI ===")
    
    for _, row in df_conceptos.iterrows():
        # Ensamblar la fórmula de Prompt según la documentación de Seedream
        prompt_completo = (
            f"Concept art visual depicting '{row['keywords']}'. "
            f"Art direction: {row['style_direction']}. "
            f"High level of detail, photorealistic textures, master composition, no watermarks."
        )

        print(f"\n[Cluster {row['cluster_id']}] Prompt para Seedream:")
        print(f"  --> \"{prompt_completo}\"")

        # Generar imagen mediante Seedream API
        resultado = seedream_client.generar_concepto_art(
            prompt=prompt_completo,
            aspect_ratio="16:9",
            resolution="2K"
        )

        print("  Respuesta Seedream AI:")
        print(f"  URL del Render: {resultado['image_url']}")
        print(f"  Estado: {resultado['status']}")