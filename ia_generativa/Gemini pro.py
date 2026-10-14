import os
from google import genai
from google.genai import types
from sklearn.datasets import fetch_20newsgroups

# ==========================================
# 1. Carga del Dataset Real de Scikit-Learn
# ==========================================
# Seleccionamos subcategorías reales de soporte técnico
categories = [
    'comp.sys.hardware',   # Problemas de hardware
    'comp.graphics',       # Problemas gráficos/drivers
    'comp.os.ms-windows.misc' # Problemas de sistema operativo
]

print("Cargando dataset real de soporte técnico desde Scikit-Learn...")
tech_dataset = fetch_20newsgroups(
    subset='test', 
    categories=categories, 
    remove=('headers', 'footers', 'quotes')
)

# Seleccionar un ticket de soporte técnico real
ticket_sample = tech_dataset.data[12]
category_real = tech_dataset.target_names[tech_dataset.target[12]]

print(f"\n[Ticket de Soporte Seleccionado - Categoría: {category_real}]")
print("-" * 60)
print(ticket_sample[:300] + "...\n")
print("-" * 60)


# ==========================================
# 2. Definición de Herramientas (Function Calling)
# ==========================================
def buscar_solucion_en_base_de_datos(componente: str, fallo: str) -> str:
    """
    Busca soluciones técnicas en la base de conocimientos del sistema.
    
    Args:
        componente: El componente afectado (ej. 'GPU', 'RAM', 'Monitor').
        fallo: Descripción breve del problema.
    """
    # Simulación de respuesta de base de datos técnica
    return (
        f"[Base de Datos Técnica] Para el componente '{componente}' con fallo '{fallo}': "
        f"Se recomienda actualizar los drivers oficiales, revisar la fuente de alimentación "
        f"y verificar la temperatura del componente mediante diagnósticos de BIOS."
    )


# ==========================================
# 3. Configuración del Cliente Gemini 3 Pro
# ==========================================
# Asegúrate de configurar la variable de entorno: export GEMINI_API_KEY="tu_api_key"
client = genai.Client()

# Definir las instrucciones del sistema para la Asistencia Técnica
system_instruction = (
    "Eres un Agente Experto de Asistencia Técnica de Nivel 3. "
    "Tu objetivo es analizar los reportes o tickets de problemas técnicos enviados por los usuarios. "
    "Debes identificar el problema raíz, categorizarlo, y utilizar las herramientas disponibles "
    "para consultar soluciones o proponer una guía paso a paso detallada para resolver la falla."
)

prompt = f"""
Por favor, analiza el siguiente ticket de soporte técnico real enviado por un usuario:

--- TICKET DE USUARIO ---
{ticket_sample}
-------------------------

Realiza un diagnóstico del problema y proporciona una respuesta técnica estructurada.
"""

# ==========================================
# 4. Inferencia con Gemini 3 Pro
# ==========================================
print("\nEnviando consulta a Gemini 3 Pro...")

response = client.models.generate_content(
    model='gemini-3-pro-preview', # O 'gemini-2.5-pro'
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2, # Baja temperatura para respuestas precisas y técnicas
        tools=[buscar_solucion_en_base_de_datos] # Habilitar herramientas
    )
)

print("\n=== RESPUESTA DEL AGENTE DE ASISTENCIA TÉCNICA (GEMINI 3 PRO) ===")
print(response.text)