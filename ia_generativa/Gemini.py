"""
Agente Conversacional Fluido y Multiturno (Gemini)
Aplicación: Consulta Dinámica sobre Datasets Reales de Scikit-Learn

Este script demuestra cómo un agente conversacional mantiene el hilo de la
conversación (memoria de contexto) mientras invoca herramientas en tiempo real
para consultar estadísticas de un dataset real de Scikit-Learn.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# SDK Oficial de Google GenAI
# Instalación: pip install google-genai pandas scikit-learn
from google import genai
from google.genai import types

# ============================================================================
# 1. CARGA DEL DATASET REAL DE SCIKIT-LEARN (California Housing)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET REAL DE SCIKIT-LEARN (California Housing)")
print("=" * 80)

data = fetch_california_housing(as_frame=True)
df_housing = data.frame
df_housing['MedHouseVal'] = df_housing['MedHouseVal'] * 100000  # Convertir a valor real en USD

print(f"✓ Dataset cargado correctamente: {df_housing.shape[0]} filas y {df_housing.shape[1]} columnas.\n")

# ============================================================================
# 2. DEFINICIÓN DE HERRAMIENTAS DE CONSULTA PARA EL AGENTE (TOOL USE)
# ============================================================================
def consultar_estadisticas_vivienda(zona_habitaciones: float = None) -> str:
    """
    Obtiene estadísticas descriptivas del precio de viviendas en California.
    Permite filtrar por número promedio de habitaciones si se especifica.
    """
    df = df_housing
    if zona_habitaciones:
        df = df[df['AveRooms'] >= zona_habitaciones]
    
    stats = {
        "total_registros": int(len(df)),
        "precio_promedio_usd": float(df['MedHouseVal'].mean()),
        "precio_mediano_usd": float(df['MedHouseVal'].median()),
        "ingreso_promedio_bloque": float(df['MedInc'].mean() * 10000)
    }
    return str(stats)

# ============================================================================
# 3. INICIALIZACIÓN DEL AGENTE CONVERSACIONAL Y SESIÓN CHAT MULTITURNO
# ============================================================================
print("=" * 80)
print("2. CONFIGURANDO EL AGENTE CONVERSACIONAL FLUIDO (GEMINI)")
print("=" * 80)

# Inicialización del cliente oficial (requiere variable de entorno GEMINI_API_KEY)
# os.environ["GEMINI_API_KEY"] = "tu-api-key-aqui"
client = genai.Client()

system_instruction = """
Eres un agente conversacional fluido, empático y altamente analítico experto en el mercado inmobiliario de California.
Tu objetivo es dialogar de forma natural con el usuario, mantener la fluidez de la conversación
y utilizar la herramienta `consultar_estadisticas_vivienda` cuando el usuario pregunte sobre precios o métricas de viviendas.
Responde siempre con tono conciso, directo y profesional en español.
"""

# Configuración del modelo con herramientas activadas
config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.3,
    tools=[consultar_estadisticas_vivienda]
)

# Inicializamos la sesión de chat con memoria fluida (automática)
chat = client.chats.create(
    model="gemini-2.5-flash", # Modelo optimizado para baja latencia y fluidez
    config=config
)

# ============================================================================
# 4. SIMULACIÓN DE DIÁLOGO FLUIDO MULTITURNO (CON MEMORIA Y HERRAMIENTAS)
# ============================================================================
mensajes_usuario = [
    "Hola, estoy interesado en analizar precios de casas en California para una inversión.",
    "¿Podrías darme el precio promedio de las viviendas según el dataset de California?",
    "¿Y cómo cambia esa cifra si busco casas más grandes, digamos con al menos 6 habitaciones promedio?",
    "Perfecto, muchas gracias. ¿Qué recomendación general me das según lo que conversamos?"
]

print("\n--- INICIO DE LA CONVERSACIÓN CON EL AGENTE FLUIDO ---\n")

for i, msj in enumerate(mensajes_usuario, 1):
    print(f"👤 Usuario (Turno {i}): {msj}")
    try:
        # El método chat.send_message gestiona automáticamente el historial multiturno
        response = chat.send_message(msj)
        print(f"🤖 Agente Gemini: {response.text}\n")
    except Exception as e:
        print(f"Nota (requiere GEMINI_API_KEY activa): {e}\n")
        break