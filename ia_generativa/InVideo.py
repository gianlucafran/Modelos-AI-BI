import os
import requests
import pandas as pd
from sklearn.datasets import fetch_california_housing

# ==========================================
# 1. Carga y Procesamiento del Dataset Real (Scikit-Learn)
# ==========================================
print("Cargando dataset real de bienes raíces desde Scikit-Learn...")

# Cargamos el dataset real 'California Housing'
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# Extraemos estadísticas clave para el contenido de redes sociales
avg_price = df['MedHouseVal'].mean() * 100000  # Convertir a valor en dólares
avg_income = df['MedInc'].mean() * 10000
top_expensive_area = df.sort_values(by='MedHouseVal', ascending=False).iloc[0]

insights_summary = {
    "topic": "California Real Estate Insights",
    "avg_price_usd": f"${avg_price:,.0f}",
    "avg_income_usd": f"${avg_income:,.0f}",
    "total_records_analyzed": len(df)
}

print("\n--- DATOS ANALIZADOS DEL DATASET ---")
for key, value in insights_summary.items():
    print(f"• {key}: {value}")


# ==========================================
# 2. Generación del Guión Narrativo para Redes Sociales
# ==========================================
def create_invideo_script(data: dict) -> str:
    """
    Construye un prompt de guión estructurado para InVideo AI
    diseñado para videos verticales de redes sociales (Reels / Shorts / TikTok).
    """
    return (
        f"Create a 30-second vertical TikTok/Reels video about {data['topic']}. "
        f"Hook the viewer in the first 3 seconds by stating that we analyzed {data['total_records_analyzed']:,} real housing records in California. "
        f"Highlight key statistics: The average home price reached {data['avg_price_usd']} while median household income stands at {data['avg_income_usd']}. "
        f"Use engaging dynamic captions, fast-paced transitions, cinematic stock footage of California cities, "
        f"and an energetic male voiceover with upbeat background music."
    )

prompt_for_invideo = create_invideo_script(insights_summary)

print("\n--- PROMPT NARRATIVO GENERADO PARA INVIDEO ---")
print(prompt_for_invideo)


# ==========================================
# 3. Integración con la API de InVideo / Webhook Automático
# ==========================================
class InVideoAPIClient:
    """
    Cliente para la API de InVideo Studio / InVideo AI.
    Permite enviar solicitudes de generación automática de video.
    """
    def __init__(self, api_key: str):
        self.base_url = "https://api.invideo.io/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_social_video(self, prompt: str, aspect_ratio: str = "9:16", platform: str = "tiktok"):
        """
        Envía una orden de creación de video con estilo para redes sociales.
        
        Args:
            prompt: Guión y directivas de diseño.
            aspect_ratio: '9:16' (Reels/TikTok), '16:9' (YouTube) o '1:1' (Feed).
            platform: Red social de destino.
        """
        endpoint = f"{self.base_url}/video/generate"
        
        payload = {
            "prompt": prompt,
            "settings": {
                "aspect_ratio": aspect_ratio,
                "target_platform": platform,
                "voiceover": {
                    "enabled": True,
                    "gender": "male",
                    "accent": "American"
                },
                "subtitles": {
                    "enabled": True,
                    "style": "viral_dynamic" # Estilo de subtítulos resaltados
                }
            }
        }
        
        response = requests.post(endpoint, json=payload, headers=self.headers)
        return response.json()


# ==========================================
# 4. Ejecución del Pipeline
# ==========================================
INVIDEO_API_KEY = os.getenv("INVIDEO_API_KEY", "tu_invideo_api_key_aqui")
invideo_client = InVideoAPIClient(api_key=INVIDEO_API_KEY)

print("\nIniciando solicitud de generación de video a InVideo...")

try:
    response = invideo_client.generate_social_video(
        prompt=prompt_for_invideo,
        aspect_ratio="9:16",
        platform="reels"
    )
    print(f"✔ Solicitud enviada con éxito. ID de Trabajo: {response.get('job_id')}")
    print(f"✔ URL de seguimiento/render: {response.get('status_url')}")

except Exception as e:
    print(f"\n[Modo Demo / Solicitud HTTP Preparada]:")
    print(f" Target Endpoint: https://api.invideo.io/v1/video/generate")
    print(f" Formato objetivo: 9:16 (Vertical Shorts/Reels/TikTok)")
    print(f" Detalles del error o clave no configurada: {e}")