"""
Resolución de Problemas Complejos con ChatGPT/GPT-5 y Datasets de Scikit-Learn
Patrón: Chain-of-Thought (CoT) & Agente Analítico Supervisado

Este script carga un dataset tabular real de Scikit-Learn, formatea casos multivariable
complejos y utiliza la API de OpenAI (GPT-4o/GPT-5) para realizar diagnósticos
explicables paso a paso.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Asegúrate de instalar openai: pip install openai
from openai import OpenAI

# 1. CARGA Y PREPARACIÓN DEL DATASET REAL DE SCIKIT-LEARN
print("=" * 80)
print("1. CARGANDO DATASET REAL DE SCIKIT-LEARN (Breast Cancer Wisconsin)")
print("=" * 80)

data = load_breast_cancer(as_frame=True)
df = data.frame
target_names = data.target_names

# Seleccionar una muestra compleja para análisis
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['target']), 
    df['target'], 
    test_size=0.2, 
    random_state=42
)

# Tomamos un caso de prueba
sample_idx = 0
sample_data = X_test.iloc[sample_idx].to_dict()
actual_label = target_names[y_test.iloc[sample_idx]]

print(f"Caso de prueba seleccionado (Índice {sample_idx}):")
print(f"Resultado Real en Dataset: {actual_label.upper()}\n")

# 2. DEFINICIÓN DEL PROMPT DE RAZONAMIENTO COMPLEJO (CHAIN-OF-THOUGHT)
def construir_prompt_cot(caracteristicas_caso):
    """
    Construye una estructura Prompt CoT que obliga al modelo a descomponer 
    el problema en pasos lógicos antes de dar una respuesta final.
    """
    features_formatted = "\n".join([f"- {k}: {v:.4f}" for k, v in caracteristicas_caso.items()])
    
    prompt = f"""
Eres un sistema experto en diagnóstico médico y análisis estadístico multivariable.
Tu tarea es analizar las siguientes mediciones celulares provenientes de una biopsia y determinar si la muestra es Benigna (benign) o Maligna (malignant).

MEDICIONES DE LA MUESTRA:
{features_formatted}

Sigue estrictamente el método de **Cadena de Pensamiento (Chain-of-Thought)** para resolver este problema complejo:

PASO 1: Análisis de Características Principales (Radio, Textura y Perímetro Promedio).
PASO 2: Evaluación de Atipicidad y Dispersión (Valores de 'worst' y 'error').
PASO 3: Ponderación de Riesgo y Contrastación de Indicadores Clave.
PASO 4: Conclusión Final y Diagnóstico Estimado.

Responde obligatoriamente en formato JSON válido con la siguiente estructura:
{{
    "paso_1_analisis_base": "...",
    "paso_2_evaluacion_atipicidad": "...",
    "paso_3_ponderacion_riesgo": "...",
    "diagnostico_final": "benign" o "malignant",
    "nivel_confianza": "Alto/Medio/Bajo"
}}
"""
    return prompt

# 3. EJECUCIÓN DEL MODELO DE RAZONAMIENTO COMPLEJO
print("=" * 80)
print("2. CONSULTANDO A CHATGPT / GPT PARA RESOLUCIÓN DEL PROBLEMA")
print("=" * 80)

# Inicializar cliente de OpenAI (Requiere la variable de entorno OPENAI_API_KEY)
# os.environ["OPENAI_API_KEY"] = "tu-api-key-aqui"

client = OpenAI()

prompt_evaluacion = construir_prompt_cot(sample_data)

try:
    response = client.chat.completions.create(
        model="gpt-4o",  # Usar "gpt-4o", "o1" o "gpt-5" según disponibilidad en API
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Eres un asistente analítico experto en resolución de problemas complejos mediante razonamiento paso a paso."},
            {"role": "user", "content": prompt_evaluacion}
        ],
        temperature=0.2
    )

    resultado_json = json.loads(response.choices[0].message.content)

    print("\n--- PROCESO DE RAZONAMIENTO PASO A PASO (CoT) ---")
    print(f"\n[Paso 1 - Análisis Base]:\n{resultado_json.get('paso_1_analisis_base')}")
    print(f"\n[Paso 2 - Atipicidad]:\n{resultado_json.get('paso_2_evaluacion_atipicidad')}")
    print(f"\n[Paso 3 - Riesgo]:\n{resultado_json.get('paso_3_ponderacion_riesgo')}")
    print("\n" + "=" * 50)
    print(f"DIAGNÓSTICO PREDICHO: {resultado_json.get('diagnostico_final').upper()}")
    print(f"CONFIANZA: {resultado_json.get('nivel_confianza')}")
    print(f"VALOR REAL EN DATASET: {actual_label.upper()}")
    print("=" * 50)

except Exception as e:
    print(f"\nNota: Para ejecutar la llamada en vivo a la API se requiere una API Key activa.")
    print(f"Detalle del error o falta de credenciales: {e}")