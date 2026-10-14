import os
import time
import requests
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# ==========================================
# 1. Carga del Dataset Real (Scikit-Learn)
# ==========================================
print("Cargando dataset real de informes y documentos desde Scikit-Learn...")

# Cargamos el dataset real '20newsgroups' filtrando discusiones/informes comerciales y legales
dataset = fetch_20newsgroups(
    subset='train',
    categories=['talk.politics.misc', 'misc.forsale', 'sci.electronics'],
    remove=('headers', 'footers', 'quotes')
)

# Seleccionamos un documento extenso real que actuará como informe/contrato
raw_doc_text = dataset.data[15]

# Formateamos el texto para simular un informe de contrato corporativo real
contract_text = f"""
================================================================================
CONTRATO DE PRESTACIÓN DE SERVICIOS E INFORME FINANCIERO - SECTOR TECNOLÓGICO
================================================================================

1. CLÁUSULA DE CONFIDENCIALIDAD Y OBLIGACIONES:
Las partes se comprometen a mantener estricta confidencialidad sobre la información técnica y comercial.
Detalles y reporte del caso analizado:
{raw_doc_text}

2. CLÁUSULA DE PAGO Y PENALIZACIONES:
El pago se realizará dentro de los 30 días posteriores a la entrega del informe técnico.
Cualquier retraso generará un interés moratorio del 2.5% mensual.

3. JURISDICCIÓN Y RESOLUCIÓN DE CONFLICTOS:
El presente acuerdo se rige por la legislación aplicable en materia comercial e internacional.
================================================================================
"""

# Guardar el contrato localmente para enviarlo a Humata AI
file_path = "contrato_informe_real.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(contract_text)

print(f"✔ Documento real preparado y guardado localmente como '{file_path}' ({len(contract_text)} caracteres).")


# ==========================================
# 2. Cliente para la API REST de Humata AI
# ==========================================
class HumataAIClient:
    """
    Cliente para la API de Humata AI (RAG para Contratos e Informes).
    Documentación oficial de endpoints: https://api.humata.ai/
    """
    def __init__(self, api_key: str):
        self.base_url = "https://api.humata.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }

    def upload_document(self, file_path: str) -> str:
        """
        Sube un documento (PDF/TXT) al espacio de trabajo de Humata AI.
        """
        endpoint = f"{self.base_url}/file/upload"
        
        with open(file_path, "rb") as file_data:
            files = {"file": (os.path.basename(file_path), file_data, "text/plain")}
            response = requests.post(endpoint, headers=self.headers, files=files)
        
        response_json = response.json()
        return response_json.get("document_id") or response_json.get("id")

    def ask_question(self, document_id: str, question: str) -> dict:
        """
        Realiza consultas en lenguaje natural sobre un documento indexado en Humata AI.
        """
        endpoint = f"{self.base_url}/chat/query"
        payload = {
            "document_id": document_id,
            "question": question,
            "include_citations": True  # Devuelve referencias/citas exactas
        }
        
        headers_json = {**self.headers, "Content-Type": "application/json"}
        response = requests.post(endpoint, json=payload, headers=headers_json)
        return response.json()


# ==========================================
# 3. Pipeline de Análisis de Contratos
# ==========================================
HUMATA_API_KEY = os.getenv("HUMATA_API_KEY", "tu_humata_api_key_aqui")
humata_client = HumataAIClient(api_key=HUMATA_API_KEY)

# Preguntas clave sobre el contrato e informe
questions = [
    "¿Cuáles son las obligaciones financieras y penalizaciones mencionadas en el contrato?",
    "¿Qué se establece con respecto a la confidencialidad?",
    "Resume en 3 puntos clave el contenido del informe."
]

print("\n--- INICIANDO ANÁLISIS AUTOMATIZADO CON HUMATA AI ---")

try:
    # 1. Subir el documento a Humata AI
    print("1. Subiendo el informe/contrato a Humata AI...")
    doc_id = humata_client.upload_document(file_path)
    print(f" ✔ Documento subido e indexado con éxito. ID: {doc_id}")

    # 2. Consultar cláusulas del documento mediante RAG
    print("\n2. Realizando preguntas analíticas sobre el contrato...")
    for q in questions:
        print(f"\nPregunta: \"{q}\"")
        res = humata_client.ask_question(doc_id, q)
        
        answer = res.get("answer", "Sin respuesta disponible.")
        citations = res.get("citations", [])
        
        print(f"Respuesta de Humata AI:\n{answer}")
        if citations:
            print(f"Citas del documento: {citations}")

except Exception as e:
    print("\n[Simulación de Pipeline de API]: Para ejecutar la llamada en vivo configura `HUMATA_API_KEY`.")
    print(f"Estructura de la solicitud enviada:")
    print(f" Endpoint Objetivo: https://api.humata.ai/v1/chat/query")
    print(f" Preguntas preparadas para el análisis: {questions}")
    print(f" Detalle técnico: {e}")