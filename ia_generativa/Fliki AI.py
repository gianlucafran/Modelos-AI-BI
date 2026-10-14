"""
Fliki AI: Creación de Contenido Rápido y Automatizado (Text-to-Video / TTS)
Aplicación: Análisis de Datos Reales (California Housing Dataset de Scikit-Learn) 
y Generación Automática de Guiones en Vídeo a través de Fliki API.

Este script analiza métricas del mercado inmobiliario real mediante Scikit-Learn,
extrae insights clave y construye el flujo de escenas/audio para la API de Fliki.
"""

import json
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. CARGA DE DATASET REAL DE SCIKIT-LEARN (California Housing)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET REAL DE SCIKIT-LEARN: California Housing (Datos Inmobiliarios)")
print("=" * 80)

# Cargar dataset real de mercado inmobiliario
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"✓ Dataset real cargado con éxito: {df.shape[0]} registros de viviendas.")
print(f"   • Variables disponibles: {list(df.columns)}")

# ============================================================================
# 2. AGRUPAMIENTO Y EXTRACCIÓN DE INSIGHTS CLAVE CON SCIKIT-LEARN
# ============================================================================
print("\n" + "=" * 80)
print("2. ANÁLISIS DE MERCADO Y SEGMENTACIÓN MEDIANTE K-MEANS")
print("=" * 80)

# Seleccionar características clave para segmentar áreas inmobiliarias
features = ['MedInc', 'AveRooms', 'MedHouseVal']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Agrupamiento en 3 segmentos de mercado (Económico, Promedio, Premium)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Calcular promedios por segmento
resumen_clusters = df.groupby('Cluster')[features].mean()
resumen_clusters['MedHouseVal'] = resumen_clusters['MedHouseVal'] * 100000  # Convertir a USD reales

print("✓ Insights extraídos por Machine Learning para el guion de vídeo:")
for cluster_id, row in resumen_clusters.iterrows():
    print(f"   • Segmento #{cluster_id + 1}: Ingreso Medio = ${row['MedInc']*10000:.0f} | Valor Propiedad = ${row['MedHouseVal']:,.0f}")

# ============================================================================
# 3. GENERADOR DE GUION Y ESTRUCTURA DE ESCENAS PARA FLIKI AI
# ============================================================================
print("\n" + "=" * 80)
print("3. CONSTRUCCIÓN DEL GUION AUTOMÁTICO Y ESTRUCTURA PARA FLIKI")
print("=" * 80)

def generar_escenas_fliki(resumen_df):
    """
    Convierte los insights de datos en un arreglo de escenas con texto para locución (TTS)
    y sugerencias de medios de fondo para la creación rápida en Fliki.
    """
    escenas = [
        {
            "content": "Welcome to today's real estate market breakdown based on machine learning analysis.",
            "mediaQuery": "real estate luxury house modern exterior"
        }
    ]
    
    for idx, row in resumen_df.iterrows():
        texto_escena = (
            f"Market Segment {idx + 1}: Areas with an average income of ${row['MedInc']*10000:,.0f} "
            f"show median home prices around ${row['MedHouseVal']:,.0f}."
        )
        media_kw = "luxury mansion" if row['MedHouseVal'] > 250000 else "suburban neighborhood houses"
        
        escenas.append({
            "content": texto_escena,
            "mediaQuery": media_kw
        })
        
    escenas.append({
        "content": "Subscribe for more data-driven market insights generated automatically.",
        "mediaQuery": "real estate agent handshake investment"
    })
    
    return escenas

escenas_script = generar_escenas_fliki(resumen_clusters)

print(f"✓ Guion de {len(escenas_script)} escenas generado dinámicamente:")
for i, sc in enumerate(escenas_script, 1):
    print(f"   • Escena {i}: \"{sc['content']}\" (B-Roll: '{sc['mediaQuery']}')")

# ============================================================================
# 4. CONSTRUCCIÓN DE PAYLOAD Y CLIENTE HTTP PARA FLIKI API
# ============================================================================
print("\n" + "=" * 80)
print("4. PAYLOAD Y CÓDIGO DE INTEGRACIÓN HTTP PARA FLIKI API v1")
print("=" * 80)

def construir_payload_fliki(title, escenas_list, aspect_ratio="16:9"):
    """
    Genera la estructura de carga útil especificada por la API REST oficial de Fliki
    para la creación automatizada de contenidos.
    """
    payload = {
        "title": title,
        "aspectRatio": aspect_ratio,  # Options: "16:9", "9:16", "1:1"
        "voice": {
            "language": "English",
            "name": "Sara",
            "gender": "Female"
        },
        "scenes": [
            {
                "text": scene["content"],
                "mediaSearchQuery": scene["mediaQuery"]
            }
            for scene in escenas_list
        ]
    }
    return payload

payload_fliki = construir_payload_fliki("California Real Estate Market Analysis", escenas_script)

print("✓ JSON Payload generado listo para enviar a la API de Fliki:")
print(json.dumps(payload_fliki, indent=2))

codigo_peticion_fliki = """
import requests

# Configuración de credenciales de la API de Fliki
API_URL = "https://api.fliki.ai/v1/content/generate/text-to-video"
HEADERS = {
    "Authorization": "Bearer TU_FLIKI_API_KEY",
    "Content-Type": "application/json"
}

# Envío de la solicitud de creación rápida de vídeo
response = requests.post(API_URL, headers=HEADERS, json=payload_fliki)

if response.status_code == 200 or response.status_code == 201:
    data = response.json()
    print(f"Vídeo creado exitosamente. ID del contenido: {data.get('id')}")
    print(f"Enlace de previsualización / render: {data.get('url')}")
else:
    print(f"Error en la creación ({response.status_code}): {response.text}")
"""

print("\n✓ Código de petición HTTP cliente:")
print(codigo_peticion_fliki)