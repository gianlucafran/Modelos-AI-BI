"""
Pipeline de Generación de Videos de Capacitación:
1. Agrupamiento de Temas de Capacitación con Scikit-Learn (KMeans & TF-IDF)
2. Renderizado de Videos Educativos con Avatares mediante Synthesia API v2
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
# 1. EXTRACCIÓN Y RESUMEN DE TEMAS CON SCIKIT-LEARN
# ==========================================================
def preparar_guiones_capacitacion():
    """
    Carga el dataset real '20 newsgroups' de Scikit-Learn, agrupa temas de 
    capacitación técnica y genera un guión pedagógico para cada módulo.
    """
    print("=== 1. Procesando Dataset de Capacitación Corporativa con Scikit-Learn ===")
    
    # Cargar categorías orientadas a soporte técnico y capacitación de sistemas
    categories = ['comp.sys.mac.hardware', 'comp.os.ms-windows.misc', 'sci.electronics']
    dataset = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    # Extracción de características con TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_tfidf = vectorizer.fit_transform(dataset.data)
    
    # Agrupamiento no supervisado para definir módulos de capacitación
    n_modulos = 2
    kmeans = KMeans(n_clusters=n_modulos, random_state=42)
    kmeans.fit(X_tfidf)
    
    terms = vectorizer.get_feature_names_out()
    modulos = []
    
    # Generación de guiones estructurados para el avatar de Synthesia
    plantillas_guion = [
        "Bienvenido al módulo de Capacitación en Infraestructura y Hardware. En esta sesión aprenderás conceptos clave sobre: {keywords}. Por favor completa la evaluación al finalizar este video.",
        "Hola a todos, bienvenidos al entrenamiento de Seguridad y Configuración de Sistemas. Hoy revisaremos los protocolos para: {keywords}. Asegúrate de seguir las buenas prácticas presentadas."
    ]
    
    for cluster_id in range(n_modulos):
        top_indices = kmeans.cluster_centers_[cluster_id].argsort()[-5:][::-1]
        top_keywords = [terms[i] for i in top_indices]
        keywords_str = ", ".join(top_keywords)
        
        guion_texto = plantillas_guion[cluster_id].format(keywords=keywords_str)
        
        modulos.append({
            "modulo_id": f"MOD-{cluster_id + 1:02d}",
            "titulo": f"Capacitación Corporativa Nivel {cluster_id + 1}",
            "keywords": keywords_str,
            "script": guion_texto
        })
        
    return pd.DataFrame(modulos)


# ==========================================================
# 2. CLIENTE API DE SYNTHESIA (Video Generation API v2)
# ==========================================================
class SynthesiaApiClient:
    """
    Cliente Python para interactuar con la API REST v2 de Synthesia.
    Documentación API: https://docs.synthesia.io/reference/
    """
    def __init__(self, api_key=None):
        # La API Key de Synthesia se envía directamente en el header Authorization
        self.api_key = api_key or os.getenv("SYNTHESIA_API_KEY")
        self.base_url = "https://api.synthesia.io/v2"

    def crear_video_capacitacion(self, titulo, guion, avatar_id="anna_costume1_cameraa", voice_id="spanish_female_1"):
        """
        Envía la solicitud de renderizado de video con avatar AI a Synthesia.
        """
        if not self.api_key:
            # Modo Simulación cuando no hay llave API en variables de entorno
            return {
                "id": "syn_mock_vid_987654321",
                "status": "in_progress (Simulación)",
                "title": titulo,
                "created_at": time.time(),
                "download_url": "https://share.synthesia.io/embeds/videos/sample_training.mp4"
            }

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        # Estructura oficial del payload v2 para generación de videos de Synthesia
        payload = {
            "test": True,  # Cambiar a False en producción para eliminar marca de agua de prueba
            "title": titulo,
            "description": "Video automático de capacitación generado vía Synthesia API",
            "visibility": "private",
            "input": [
                {
                    "scriptText": guion,
                    "avatar": avatar_id,
                    "avatarSettings": {
                        "horizontalAlign": "center",
                        "scale": 1.0
                    },
                    "background": "off_white"
                }
            ]
        }

        try:
            url = f"{self.base_url}/videos"
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error en Synthesia API: {e}")
            return None

    def consultar_estado_video(self, video_id):
        """
        Realiza polling sobre la API para verificar si el renderizado fue completado.
        """
        if not self.api_key:
            return {"status": "complete", "download": "https://share.synthesia.io/sample.mp4"}

        headers = {"Authorization": self.api_key}
        url = f"{self.base_url}/videos/{video_id}"
        
        response = requests.get(url, headers=headers)
        return response.json()


# ==========================================================
# 3. PIPELINE DE EJECUCIÓN
# ==========================================================
if __name__ == "__main__":
    # Paso 1: Generar guiones desde el dataset de Scikit-Learn
    df_modulos = preparar_guiones_capacitacion()
    print("\n=== Guiones de Capacitación Generados ===")
    print(df_modulos[['modulo_id', 'titulo', 'script']])

    # Paso 2: Inicializar cliente Synthesia API
    synthesia = SynthesiaApiClient()

    # Paso 3: Enviar solicitudes de renderizado de video para cada módulo
    print("\n=== 2. Solicitando Renderizado de Videos a Synthesia API ===")
    
    for _, row in df_modulos.iterrows():
        print(f"\nProcesando Módulo: {row['titulo']}...")
        print(f"Guión enviado al Avatar: \"{row['script']}\"")
        
        # Crear solicitud de video
        res_video = synthesia.crear_video_capacitacion(
            titulo=row['titulo'],
            guion=row['script'],
            avatar_id="santa_costume1_cameraa" # ID de avatar estándar
        )
        
        if res_video:
            video_id = res_video.get("id")
            estado = res_video.get("status")
            print(f"  -> Solicitud aceptada exitosamente. Video ID: {video_id}")
            print(f"  -> Estado inicial de renderizado: {estado}")
            
            if "download_url" in res_video:
                print(f"  -> URL de visualización: {res_video['download_url']}")