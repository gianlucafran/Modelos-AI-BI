import os
import time
import requests
from sklearn.datasets import fetch_20newsgroups
from leonardo_ai_sdk import LeonardoAiSDK
from leonardo_ai_sdk.models import shared

# ==========================================
# 1. Carga del Dataset Real (Scikit-Learn)
# ==========================================
print("Cargando dataset real desde Scikit-Learn (Fantasía y Ciencia Ficción)...")

# Cargamos el dataset real '20newsgroups' filtrando la subcategoría de gráficos/juegos
dataset = fetch_20newsgroups(
    subset='train',
    categories=['rec.autos', 'sci.space', 'comp.graphics'],
    remove=('headers', 'footers', 'queries')
)

# Definimos clases/roles de personajes reales de diseño de juegos
character_archetypes = [
    {"role": "Cyberpunk Outlaw", "theme": "Sci-Fi neon, tactical jacket, prosthetic arm"},
    {"role": "Astronaut Engineer", "theme": "Hard sci-fi spacesuit, glowing visor, tools"},
    {"role": "Fantasy Paladin", "theme": "Plate armor, glowing runes, cape"},
    {"role": "Steampunk Inventor", "theme": "Brass goggles, leather apron, mechanical gauntlet"}
]

# Seleccionamos una clase de personaje para la demo
selected_char = character_archetypes[0]
print(f"Personaje seleccionado para el diseño 3D/2D: {selected_char['role']}")


# ==========================================
# 2. Configuración del Cliente Leonardo.Ai SDK
# ==========================================
# Obtener API Key de Leonardo.Ai (export LEONARDO_API_KEY="tu_api_key")
api_key = os.getenv("LEONARDO_API_KEY", "tu_api_key_de_leonardo_aqui")

# Inicializar cliente oficial de Leonardo.Ai
sdk = LeonardoAiSDK(bearer_auth=api_key)


# ==========================================
# 3. Construcción del Prompt de Hoja de Personaje 2D/3D
# ==========================================
def build_character_sheet_prompt(role: str, details: str) -> str:
    """
    Formatea el prompt para obtener una hoja de modelo (character turnaround)
    adecuada para modeladores 3D o artistas de concept art 2D.
    """
    return (
        f"Character model sheet of a {role}, {details}. "
        f"Full body turnaround view: front view, side view, back view. "
        f"Orthographic projection, clean white background, T-pose / neutral pose, "
        f"highly detailed 3D render style, Unreal Engine 5 concept art, game asset ready."
    )

prompt_text = build_character_sheet_prompt(selected_char['role'], selected_char['theme'])

print(f"\n--- PROMPT GENERADO PARA LEONARDO.AI ---")
print(prompt_text)


# ==========================================
# 4. Solicitud de Generación vía API de Leonardo.Ai
# ==========================================
print("\nEnviando orden de generación a Leonardo.Ai...")

try:
    # Solicitud de generación de imágenes mediante la SDK oficial
    response = sdk.image.create_generation(
        request=shared.CreateGenerationRequestBody(
            prompt=prompt_text,
            model_id="e31a67a1-5d97-4e5b-ac0f-2e928238122c",  # RPG v5 / Leonardo Diffusion Model
            width=1024,
            height=768,
            num_images=2,
            prompt_magic=True,
            public=False
        )
    )

    generation_id = response.sd_generation_job.generation_id
    print(f"✔ Generación iniciada correctamente. ID de Trabajo: {generation_id}")

    # ==========================================
    # 5. Polling Asíncrono para Obtener los Renders
    # ==========================================
    print("Esperando la finalización de la generación de la hoja de personaje...")
    
    completed = False
    image_urls = []
    
    # Bucle de sondeo (Polling)
    for _ in range(12):  # Reintentar durante ~1 minuto max
        time.sleep(5)
        job_status = sdk.image.get_generation_by_id(id=generation_id)
        
        status = job_status.single_generations_by_pk.status
        print(f" Estado actual: {status}")
        
        if status == shared.JobStatus.COMPLETE:
            completed = True
            generated_images = job_status.single_generations_by_pk.generated_images
            image_urls = [img.url for img in generated_images]
            break

    if completed:
        print("\n=== HOJAS DE PERSONAJE 2D/3D GENERADAS ===")
        for idx, url in enumerate(image_urls, 1):
            print(f"• Vista {idx}: {url}")
            
            # Opcional: Descargar la imagen
            img_data = requests.get(url).content
            filename = f"character_sheet_{selected_char['role'].replace(' ', '_').lower()}_{idx}.png"
            with open(filename, 'wb') as f:
                f.write(img_data)
            print(f"  ✔ Guardada localmente como: {filename}")

except Exception as e:
    print(f"\n[Error/Simulación API]: Para ejecutar el script real asegúrate de instalar `pip install Leonardo-Ai-SDK` y configurar `LEONARDO_API_KEY`.")
    print(f"Detalle técnico: {e}")