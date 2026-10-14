"""
Pipeline de Educación Online: Clustering de Temas Educativos con Scikit-Learn
y Generación de Cursos en Video con DeepBrain AI (AI Studios API v2).
"""

import os
import json
import time
import requests
import pandas as pd

# Importación de Scikit-Learn para Clustering y NLP sobre el Dataset Real
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ==========================================================
# 1. EXTRACCIÓN Y GENERACIÓN DE GUIONES CON SCIKIT-LEARN
# ==========================================================
def preparar_lecciones_educativas():
    """
    Carga el dataset real '20 newsgroups' de scikit-learn, agrupa los conceptos
    técnicos mediante KMeans y genera un guión pedagógico para la lección online.
    """
    print("=== 1. Procesando Dataset Académico con Scikit-Learn ===")
    
    # Cargar categorías orientadas a tecnología y educación en computación
    categorias = ['comp.graphics', 'comp.os.ms-windows.misc', 'sci.electronics']
    dataset = fetch_20newsgroups(subset='train', categories=categorias, remove=('headers', 'footers', 'quotes'))
    
    # Vectorización de texto con TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_tfidf = vectorizer.fit_transform(dataset.data)
    
    # Agrupamiento no supervisado con KMeans
    n_lecciones = 2
    kmeans = KMeans(n_clusters=n_lecciones, random_state=42)
    kmeans.fit(X_tfidf)
    
    terms = vectorizer.get_feature_names_out()
    lecciones = []
    
    # Plantillas de lecciones académicas para el avatar de DeepBrain AI
    plantillas_script = [
        "Hola a todos y bienvenidos a la lección de Gráficos y Procesamiento Visual. Hoy abordaremos temas fundamentales como: {keywords}. Al finalizar la clase, revisa el material complementario.",
        "Bienvenidos al módulo de Fundamentos de Hardware y Sistemas Operativos. En esta sesión analizaremos conceptos clave sobre: {keywords}. Asegúrate de tomar notas durante la lección."
    ]
    
    for cluster_id in range(n_lecciones):
        top_indices = kmeans.cluster_centers_[cluster_id].argsort()[-5:][::-1]
        top_keywords = [terms[i] for i in top_indices]
        keywords_str = ", ".join(top_keywords)
        
        guion_texto = plantillas_script[cluster_id].format(keywords=keywords_str)
        
        lecciones.append({
            "leccion_id": f"LEC-{cluster_id + 1:02d}",
            "titulo": f"Lección {cluster_id + 1}: Fundamentos Técnicos",
            "keywords": keywords_str,
            "script": guion_texto
        })
        
    return pd.DataFrame(lecciones)


# ==========================================================
# 2. CLIENTE DE DEEPBRAIN AI (AI Studios API v2)
# ==========================================================
class DeepBrainAIClient:
    """
    Cliente Python para la API de DeepBrain AI (AI Studios v2).
    Documentación oficial: https://docs.aistudios.com/
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPBRAIN_API_KEY") or os.getenv("AISTUDIOS_API_KEY")
        self.base_url = "https://open-api.aistudios.com/v2"

    def crear_video_educativo(self, titulo, guion, model_avatar="M00001", idioma="es"):
        """
        Envía la solicitud de generación de video a la API de AI Studios de DeepBrain AI.
        """
        if not self.api_key:
            # Modo simulación cuando no hay API Key configurada
            return {
                "status": "Simulated Success",
                "project_id": "db_proj_edu_2026_98765",
                "title": titulo,
                "video_url": "https://aistudios.com/share/sample_educational_video.mp4"
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Estructura del payload según la API de AI Studios
        payload = {
            "title": titulo,
            "scenes": [
                {
                    "avatar": {
                        "model": model_avatar,
                        "speech": guion,
                        "language": idioma
                    },
                    "background": {
                        "type": "color",
                        "value": "#1E1E2E"
                    }
                }
            ]
        }

        try:
            url = f"{self.base_url}/video/generate"
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            
            return {
                "status": "Success",
                "project_id": res_json.get("data", {}).get("projectId"),
                "title": titulo,
                "video_url": res_json.get("data", {}).get("videoUrl")
            }
        except Exception as e:
            print(f"Error al llamar a DeepBrain AI API: {e}")
            return None


# ==========================================================
# 3. EJECUCIÓN DEL PIPELINE
# ==========================================================
if __name__ == "__main__":
    # Paso 1: Procesar dataset y generar guiones con Scikit-Learn
    df_lecciones = preparar_lecciones_educativas()
    print("\n=== Lecciones Educativas Generadas ===")
    print(df_lecciones[['leccion_id', 'titulo', 'script']])

    # Paso 2: Inicializar cliente de DeepBrain AI
    deepbrain_client = DeepBrainAIClient()

    # Paso 3: Enviar solicitudes de renderizado de video
    print("\n=== 2. Generando Videos de Educación Online con DeepBrain AI ===")
    
    for _, row in df_lecciones.iterrows():
        print(f"\nGenerando Video para: {row['titulo']}...")
        print(f"Guión para el Avatar: \"{row['script']}\"")
        
        resultado = deepbrain_client.crear_video_educativo(
            titulo=row['titulo'],
            guion=row['script'],
            idioma="es"
        )
        
        print("  Respuesta de DeepBrain AI:")
        print(f"  ID de Proyecto: {resultado['project_id']}")
        print(f"  URL del Video: {resultado['video_url']}")
        print(f"  Estado: {resultado['status']}")