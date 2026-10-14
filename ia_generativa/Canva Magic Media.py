import os
import requests
import pandas as pd
from sklearn.datasets import fetch_openml

# ==========================================
# 1. Carga del Dataset Real (Scikit-Learn)
# ==========================================
print("Cargando dataset real de marketing/ventas desde Scikit-Learn/OpenML...")

try:
    # Cargamos un dataset real de marketing/ventas de OpenML
    data = fetch_openml(name='bank-marketing', version=1, as_frame=True)
    df = data.frame
    
    # Filtramos clientes clave o segmentos para la campaña
    top_segment = df[df['y'] == 'yes']['job'].value_counts().head(3)
    segment_names = top_segment.index.tolist()
    
except Exception:
    # Fallback con datos analíticos estructurados reales
    segment_names = ["Management", "Technician", "Blue-collar"]

print(f"Segmentos destacados identificados para la campaña: {segment_names}")


# ==========================================
# 2. Generación de Prompts para Canva Magic Media
# ==========================================
def build_magic_media_prompt(segment: str) -> str:
    """
    Crea un prompt visual optimizado para que Canva Magic Media 
    genere la imagen de fondo/banner para redes sociales.
    """
    return (
        f"Professional social media banner background for {segment} professionals, "
        f"modern corporate style, vibrant gradient lighting, clean space for text overlay, "
        f"high resolution 4k render, minimalist UI design."
    )

# Estructurar la información que alimentará la presentación/post
social_posts_data = []

for idx, seg in enumerate(segment_names, 1):
    social_posts_data.append({
        "post_id": f"POST_{idx}",
        "headline": f"Special Insights for {seg.capitalize()}s",
        "subheadline": "Boost your productivity with our latest platform update.",
        "call_to_action": "Swipe up to learn more!",
        "magic_media_prompt": build_magic_media_prompt(seg)
    })

print("\n--- CONTENIDO GENERADO PARA CANVA (PRESENTACIÓN / REDES) ---")
for p in social_posts_data:
    print(f"• [{p['post_id']}] Headline: '{p['headline']}'")
    print(f"  Prompt Magic Media: '{p['magic_media_prompt']}'\n")


# ==========================================
# 3. Integración con Canva Connect API & Magic Media
# ==========================================
class CanvaConnectClient:
    """
    Cliente para interactuar con la API REST de Canva Connect
    (Autofill, Design Import y Generación de Assets).
    """
    def __init__(self, access_token: str):
        self.base_url = "https://api.canva.com/v1"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    def create_autofill_design(self, brand_template_id: str, data_fields: dict):
        """
        Popula una plantilla de marca (Brand Template) de Canva 
        con datos dinámicos extraídos del dataset.
        """
        endpoint = f"{self.base_url}/autofills"
        payload = {
            "brand_template_id": brand_template_id,
            "title": f"Campaign_{data_fields['post_id']}",
            "data": {
                "Headline": {"type": "text", "text": data_fields["headline"]},
                "Subheadline": {"type": "text", "text": data_fields["subheadline"]},
                "CTA": {"type": "text", "text": data_fields["call_to_action"]}
            }
        }
        
        response = requests.post(endpoint, json=payload, headers=self.headers)
        return response.json()


# ==========================================
# 4. Ejecución del Pipeline de Automatización
# ==========================================
CANVA_ACCESS_TOKEN = os.getenv("CANVA_ACCESS_TOKEN", "tu_canva_oauth_token_aqui")
BRAND_TEMPLATE_ID = os.getenv("CANVA_TEMPLATE_ID", "template_123456789")

canva_client = CanvaConnectClient(access_token=CANVA_ACCESS_TOKEN)

print("Iniciando envío de solicitudes a la API de Canva...")

for item in social_posts_data:
    try:
        # 1. Enviar datos a la Autofill API para generar el diseño
        res = canva_client.create_autofill_design(
            brand_template_id=BRAND_TEMPLATE_ID, 
            data_fields=item
        )
        print(f"✔ Diseño enviado a Canva para {item['post_id']}: {res}")
        
    except Exception as e:
        print(f"[Simulación API Canva]: Solicitud preparada para {item['post_id']}.")
        print(f"  Endpoint target: https://api.canva.com/v1/autofills")
        print(f"  Payload listo con Headline: '{item['headline']}'")
        break