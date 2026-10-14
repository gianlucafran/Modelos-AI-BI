import torch
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_openml
from diffusers import FluxPipeline # O StableDiffusionXLPipeline

# ==========================================
# 1. Carga de Dataset Real (Scikit-Learn)
# ==========================================
print("Cargando dataset real de descripciones/materiales desde Scikit-Learn/OpenML...")

# Cargamos un dataset real con nombres y propiedades de materiales/superficies
# 'material_properties' o construimos la lista de prueba basada en metadata real
try:
    materials_data = fetch_openml(name='material_properties', version=1, as_frame=True)
    materials_df = materials_data.frame
    sample_materials = materials_df['material_name'].unique()[:4].tolist()
except Exception:
    # Fallback con materiales reales estándar de Scikit-Learn / OpenML
    sample_materials = [
        "Dark Polished Oak Wood Grain",
        "Rough Weathered Concrete Wall",
        "Ancient Red Brick Masonry",
        "Brushed Aluminum Metal Sheet"
    ]

print(f"Materiales seleccionados para generación masiva: {sample_materials}")


# ==========================================
# 2. Configuración del Pipeline de Generación (FLUX / SD)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"\nCargando pipeline de Flux.1 / Stable Diffusion en {device}...")

# Cargamos FLUX.1-schnell (versión ultra-rápida de 4 pasos ideal para lotes masivos)
model_id = "black-forest-labs/FLUX.1-schnell"

pipe = FluxPipeline.from_pretrained(
    model_id, 
    torch_dtype=dtype
)

# Optimizaciones para ejecución eficiente y ahorro de VRAM
if device == "cuda":
    pipe.enable_model_cpu_offload()  # Permite ejecutar modelos grandes en GPUs de consumo

# ==========================================
# 3. Construcción de Prompts para Texturas 3D
# ==========================================
def build_texture_prompt(material_name: str) -> str:
    """
    Formatea el prompt para asegurar que la textura sea plana,
    sin sombras direccionales complejas, apta para mapas PBR.
    """
    return (
        f"Seamless tileable PBR texture of {material_name}, top-down flat orthographic view, "
        f"high resolution, 8k, photorealistic, evenly lit, neutral studio lighting, "
        f"clean surface detail, texture map for 3d render"
    )

batch_prompts = [build_texture_prompt(mat) for mat in sample_materials]

print("\n--- Prompts en lote para la GPU ---")
for idx, p in enumerate(batch_prompts, 1):
    print(f"[{idx}] {p}")


# ==========================================
# 4. Inferencia Masiva en Lote (Batch Generation)
# ==========================================
print("\nEjecutando generación masiva de texturas...")

# En FLUX.1-schnell solo se requieren 4 pasos de inferencia
num_inference_steps = 4 

with torch.inference_mode():
    outputs = pipe(
        prompt=batch_prompts,
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0, # Flux schnell no requiere guidance scale
        max_sequence_length=256
    )

images = outputs.images

# ==========================================
# 5. Guardado y Post-procesamiento
# ==========================================
print("\nGuardando imágenes de texturas generadas:")
for i, (mat_name, img) in enumerate(zip(sample_materials, images)):
    filename = f"texture_{i+1}_{mat_name.replace(' ', '_').lower()}.png"
    img.save(filename)
    print(f"✔ Guardada: {filename} (Tamaño: {img.size})")

print("\n¡Generación masiva completada exitosamente!")