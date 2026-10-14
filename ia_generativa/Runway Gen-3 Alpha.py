"""
Runway Gen-3 Alpha: Pipeline de Producción Cinematográfica y Generación de Vlogs/Tomas
Aplicación: Clasificación de Expresión/Escenas con Scikit-Learn (Dataset Real LFW People) 
y Orquestación de Prompts / API para Gen-3 Alpha.

Usa el dataset real Labeled Faces in the Wild (LFW) de Scikit-Learn para extraer 
parámetros visuales de personajes y generar las directivas de cámara de Gen-3 Alpha.
"""

import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. CARGA DE DATASET REAL DE SCIKIT-LEARN (LFW People)
# ============================================================================
print("=" * 80)
print("1. CARGANDO DATASET REAL DE SCIKIT-LEARN: LFW People (Personajes de Cine)")
print("=" * 80)

# Cargar dataset real de rostros (personas etiquetadas en la web)
# Filtramos clases para tener una muestra representativa de casting/personajes
lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.4)

n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

print(f"✓ Dataset LFW cargado exitosamente:")
print(f"   • Total de tomas/muestras: {n_samples}")
print(f"   • Dimensión de imagen: {h}x{w} píxeles")
print(f"   • Personajes identificados en casting: {len(target_names)}")

# ============================================================================
# 2. MODELADO DE ATRIBUTOS CINEMATOGRÁFICOS (PCA + SVM)
# ============================================================================
print("\n" + "=" * 80)
print("2. CLASIFICACIÓN DE PERSONAJES Y EXTRACTION DE VECTOR ESTÍMULO (EFECTOS/ILUMINACIÓN)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Pipeline de Eigenfaces para análisis cinematográfico
n_components = 50
model = make_pipeline(
    StandardScaler(),
    PCA(n_components=n_components, whiten=True, random_state=42),
    SVC(kernel='rbf', class_weight='balanced', C=1000.0, gamma=0.005)
)

model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f"✓ Modelo de clasificación de personajes entrenado | Precisión: {acc*100:.2f}%")

# ============================================================================
# 3. GENERADOR DE PROMPTS CINEMATOGRÁFICOS (GEN-3 ALPHA SINTAXIS ESTRUCTURADA)
# ============================================================================
print("\n" + "=" * 80)
print("3. GENERADOR DE PROMPTS Y MOVIMIENTO DE CÁMARA PARA RUNWAY GEN-3 ALPHA")
print("=" * 80)

# Gen-3 Alpha utiliza una estructura recomendada: [Estilo/Entorno] + [Sujeto/Acción] + [Cámara/Iluminación]
movimientos_camara = [
    "Slow push-in zoom with anamorphic lens flare, 35mm lens, f/1.8",
    "Dynamic FPV drone tracking shot moving left to right at high speed",
    "Static cinematic medium close-up, dramatic Rembrandt lighting with volumetric fog",
    "Low angle pedestal up movement, soft golden hour sunlight, hyperrealistic 8k"
]

def crear_prompt_gen3(personaje_id, nombre_actor, intensidad_luz):
    movimiento = movimientos_camara[personaje_id % len(movimientos_camara)]
    
    # Determinamos tono basado en intensidad de iluminación simulada
    estilo_luz = "moody low-key chiaroscuro lighting" if intensidad_luz < 0.5 else "high-key bright natural studio light"
    
    prompt_estructurado = (
        f"[Camera: {movimiento}] [Subject: A cinematic shot of character played by {nombre_actor}, "
        f"expressing intense emotion, detailed skin texture, photorealistic movie still] "
        f"[Environment & Lighting: Cinematic film set, {estilo_luz}, shot on ARRI Alexa 35, 8k resolution]"
    )
    return prompt_estructurado

prompts_generados = []
for i in range(min(4, len(target_names))):
    actor = target_names[i]
    intensidad = float(np.mean(X[y == i]) / 255.0)
    p = crear_prompt_gen3(i, actor, intensidad)
    prompts_generados.append({"actor": actor, "prompt": p})
    print(f"\n🎬 Toma Cinematográfica #{i+1} [{actor}]:")
    print(f"   \"{p}\"")

# ============================================================================
# 4. INTEGRAción CON LA API DE RUNWAY (SDK / REQUESTS DE GEN-3 ALPHA)
# ============================================================================
print("\n" + "=" * 80)
print("4. ESTRUCTURA DE LLAMADA A LA API DE RUNWAY GEN-3 ALPHA")
print("=" * 80)

def payload_runway_gen3(prompt_text, duration_sec=5, ratio="16:9"):
    """
    Construye el JSON especificado en la documentación técnica de la API de Runway
    para el modelo Gen-3 Alpha (Text-to-Video / Image-to-Video).
    """
    return {
        "model": "gen3a_turbo",
        "promptText": prompt_text,
        "watermark": False,
        "duration": duration_sec,
        "ratio": ratio,
        "cameraControl": {
            "pan": 0.2,
            "tilt": -0.1,
            "zoom": 0.5
        }
    }

ejemplo_payload = payload_runway_gen3(prompts_generados[0]["prompt"])
print("✓ JSON Payload generado listo para POST https://api.dev.runwayml.com/v1/tasks:")
print(json.dumps(ejemplo_payload, indent=2))