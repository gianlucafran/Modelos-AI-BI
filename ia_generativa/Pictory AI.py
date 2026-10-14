"""
Pictory AI: Generación Automática de Shorts y Reels desde Textos Largos
Aplicación: Extracción de Frases Clave (LSA / TF-IDF) con Scikit-Learn (Dataset Real 20 Newsgroups)
y Generación de Storyboard / Solicitudes a la API REST de Pictory.

Usa el dataset real 20 Newsgroups de Scikit-Learn para extraer resúmenes automáticos
de artículos de tecnología/espacio y transformarlos en guiones con formato vertical (9:16).
"""

import json
import re
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ============================================================================
# 1. CARGA DE DATASET REAL DE SCIKIT-LEARN (20 Newsgroups)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET REAL DE SCIKIT-LEARN: 20 Newsgroups (Artículos y Noticias)")
print("=" * 80)

# Cargamos artículos reales sobre exploración espacial y tecnología
categorias = ['sci.space', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categorias, remove=('headers', 'footers', 'quotes'))

articulo_raw = newsgroups.data[0]  # Tomamos el primer texto representativo
# Limpieza básica de saltos de línea múltiples
articulo_limpio = re.sub(r'\s+', ' ', articulo_raw).strip()

print(f"✓ Dataset cargado. Longitud del artículo seleccionado: {len(articulo_limpio)} caracteres.")
print(f"   Extracto inicial: \"{articulo_limpio[:150]}...\"\n")

# ============================================================================
# 2. EXTRACCIÓN DE HIGHLIGHTS (LSA / TF-IDF) PARA GUION DE REELS
# ============================================================================
print("=" * 80)
print("2. EXTRAENDO HIGHLIGHTS DE IMPACTO PARA EL SHORT CON SCIKIT-LEARN (LSA)")
print("=" * 80)

# Segmentamos el texto en oraciones individuales
oraciones = [s.strip() for s in re.split(r'(?<=[.!?]) +', articulo_limpio) if len(s.strip()) > 20]

# Representación vectorial con TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(oraciones)

# Latent Semantic Analysis (TruncatedSVD) para encontrar las oraciones más representativas
svd = TruncatedSVD(n_components=1, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

# Ordenamos las oraciones según su importancia conceptual en el texto
scores = svd_matrix.ravel()
top_indices = np.argsort(scores)[::-1][:4]  # Top 4 oraciones clave para un Reel de 30-45 seg
oraciones_seleccionadas = [oraciones[i] for i in sorted(top_indices)]

print("✓ Frases clave extraídas por Machine Learning para el guion del Reel:")
for idx, o in enumerate(oraciones_seleccionadas, 1):
    print(f"   {idx}. \"{o}\"")

# ============================================================================
# 3. GENERADOR DE STORYBOARD Y PAYLOAD API PARA PICTORY
# ============================================================================
print("\n" + "=" * 80)
print("3. CONSTRUCCIÓN DEL PAYLOAD JSON PARA PICTORY API (SCRIPT-TO-VIDEO)")
print("=" * 80)

def construir_payload_pictory_reel(oraciones_script, titulo_reel="Tech News Highlight"):
    """
    Construye el JSON formal bajo la especificación Script-to-Video de la API de Pictory v2.
    Configura el formato vertical (9:16) para TikTok, Instagram Reels y YouTube Shorts.
    """
    escenas = []
    for text in oraciones_script:
        escenas.append({
            "text": text,
            "voiceOver": True,
            "splitText": True
        })
    
    payload = {
        "videoName": titulo_reel,
        "videoDescription": "Reel automático generado mediante NLP y Pictory API",
        "language": "en",
        "videoWidth": 1080,
        "videoHeight": 1920,  # Formato vertical 9:16 (Shorts / Reels)
        "audio": {
            "aiVoiceover": {
                "speaker": "Jackson",  # Voz de IA predeterminada
                "speed": "100",
                "volume": "100"
            },
            "autoBackgroundMusic": True,
            "backGroundMusicVolume": 0.15
        },
        "scenes": escenas,
        "settings": {
            "subtitles": {
                "enabled": True,
                "style": "Highlight-Yellow-Bold"  # Subtítulos dinámicos estilo Hormozi/Reels
            }
        }
    }
    return payload

payload_pictory = construir_payload_pictory_reel(oraciones_seleccionadas, "Space Exploration News Reel")

print("✓ JSON Payload listo para POST a https://api.pictory.ai/v1/jobs/scriptToVideo:")
print(json.dumps(payload_pictory, indent=2))

# ============================================================================
# 4. SIMULACIÓN DE PETICIÓN A LA API DE PICTORY
# ============================================================================
print("\n" + "=" * 80)
print("4. CÓDIGO DE INTEGRACIÓN HTTP (CLIENTE API PICTORY)")
print("=" * 80)

codigo_solicitud_http = """import requests

# Endpoints oficiales de Pictory API
URL_PICTORY = "https://api.pictory.ai/v1/jobs/scriptToVideo"
HEADERS = {
    "X-Pictory-User-Id": "TU_USER_ID",
    "Authorization": "Bearer TU_PICTORY_API_KEY",
    "Content-Type": "application/json"
}

# Envío del job de renderizado
response = requests.post(URL_PICTORY, headers=HEADERS, json=payload_pictory)

if response.status_code == 200:
    job_data = response.json()
    print(f"Job enviado con éxito. ID de Job: {job_data.get('jobId')}")
else:
    print(f"Error {response.status_code}: {response.text}")
"""

print(codigo_solicitud_http)