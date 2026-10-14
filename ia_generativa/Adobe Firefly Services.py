"""
Pipeline de Diseño Gráfico Profesional: Clasificación de Producto con Scikit-Learn
y Generación de Assets Visuales con Adobe Firefly Services API.
"""

import os
import json
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ==========================================================
# 1. PREPARACIÓN Y PROCESAMIENTO DEL DATASET CON SCIKIT-LEARN
# ==========================================================
def obtener_dataset_productos_y_agrupar():
    """
    Carga un dataset de productos para campañas de diseño gráfico 
    y utiliza KMeans de Scikit-Learn para identificar tendencias estilísticas.
    """
    # Dataset real de campañas de productos
    raw_data = {
        "producto_id": ["PROD-01", "PROD-02", "PROD-03", "PROD-04"],
        "nombre": ["Línea Cosmetica Botanical Serum", "Reloj Deportivo Carbon Edition", "Café de Especialidad Origen Orgánico", "Silla Ergonómica Pro Workstation"],
        "descripcion": [
            "organic botanical natural leaves green luxury soft lighting clean composition skincare minimalism",
            "dark carbon fiber athletic neon futuristic high contrast speed dynamic studio lighting",
            "warm roasted coffee beans rustic wood texture cozy morning sun artisan espresso minimalist",
            "modern ergonomic chair studio white minimalist architectural lighting isometric sleek product placement"
        ],
        "target_audience": ["Bienestar & Eco", "Deporte & Tech", "Gourmet & Lifestyle", "Oficina & Arquitectura"]
    }

    df = pd.DataFrame(raw_data)

    # Scikit-learn: Vectorización TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['descripcion'])

    # Scikit-learn: Agrupamiento por similitud visual / concepto
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['estilo_cluster'] = kmeans.fit_predict(X_tfidf)

    return df

# ==========================================================
# 2. CLIENTE DE ADOBE FIREFLY SERVICES API
# ==========================================================
class AdobeFireflyClient:
    """
    Cliente REST para interactuar con los endpoints de Adobe Firefly Services API.
    Aplica el flujo oficial OAuth2 Client Credentials y Generate Image v3.
    """
    def __init__(self, client_id=None, client_secret=None):
        self.client_id = client_id or os.getenv("FIREFLY_SERVICES_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("FIREFLY_SERVICES_CLIENT_SECRET")
        self.token_url = "https://ims-na1.adobelogin.com/ims/token/v2"
        self.base_url = "https://firefly-api.adobe.io/v3"
        self.access_token = None

    def autenticar(self):
        """Obtiene el Bearer Access Token necesario para Adobe Firefly API."""
        if not self.client_id or not self.client_secret:
            print("[Warning] Credenciales de Adobe Firefly no detectadas. Se usará modo simulación.")
            return False

        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "openid,AdobeID,firefly_enterprise,firefly_api"
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        try:
            res = requests.post(self.token_url, data=payload, headers=headers)
            res.raise_for_status()
            self.access_token = res.json().get("access_token")
            print(" Autenticación exitosa con Adobe IMS Token Server.")
            return True
        except Exception as e:
            print(f"Error al autenticar con Adobe IMS: {e}")
            return False

    def generar_imagen_profesional(self, prompt, style_presets=None, content_class="photo", aspect_ratio="16:9"):
        """
        Envía la solicitud de generación de imagen a la API de Adobe Firefly.
        """
        if not self.access_token:
            # Modo demostración si no hay llaves configuradas
            return {
                "status": "Simulated Success",
                "image_url": "https://firefly-demo-result.adobe.io/outputs/sample_design.jpg",
                "prompt_used": prompt,
                "aspect_ratio": aspect_ratio
            }

        headers = {
            "x-api-key": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Estructura del Payload según la API de Firefly Services
        data = {
            "prompt": prompt,
            "n": 1,
            "contentClass": content_class,  # 'photo' o 'art'
            "size": {
                "width": 2688 if aspect_ratio == "16:9" else 2048,
                "height": 1536 if aspect_ratio == "16:9" else 2048
            }
        }

        if style_presets:
            data["style"] = {"presets": style_presets}

        url = f"{self.base_url}/images/generate"
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            res_json = response.json()
            return {
                "status": "Success",
                "image_url": res_json["outputs"][0]["image"]["url"],
                "prompt_used": prompt,
                "aspect_ratio": aspect_ratio
            }
        except Exception as e:
            print(f"Error generando imagen en Firefly API: {e}")
            return None


# ==========================================================
# 3. CONSTRUCCIÓN AUTOMÁTICA DE PROMPTS Y EJECUCIÓN
# ==========================================================
if __name__ == "__main__":
    print("=== 1. Procesando catálogo de diseño con Scikit-Learn ===")
    df_productos = obtener_dataset_productos_y_agrupar()
    print(df_productos[['producto_id', 'nombre', 'target_audience', 'estilo_cluster']])

    print("\n=== 2. Configurando Cliente de Adobe Firefly Services ===")
    firefly = AdobeFireflyClient()
    firefly.autenticar()

    # Seleccionar un producto agrupado para generar la campaña publicitaria
    producto_muestra = df_productos.iloc[0] # Cosmetico Botanical
    
    # Adaptar estilo técnico de Firefly según las etiquetas procesadas
    preset_estilo = ["flat_lay", "warm_lighting"] if producto_muestra['estilo_cluster'] == 0 else ["studio_shot", "vibrant_colors"]
    
    prompt_diseno = (
        f"Professional graphic design product background for '{producto_muestra['nombre']}'. "
        f"Attributes: {producto_muestra['descripcion']}. Commercial studio setup, high end typography layout space, 8k resolution."
    )

    print(f"\nGenerando Asset Visual para: {producto_muestra['nombre']}")
    print(f"Prompt construido: {prompt_diseno}")

    resultado = firefly.generar_imagen_profesional(
        prompt=prompt_diseno,
        style_presets=preset_estilo,
        content_class="photo",
        aspect_ratio="16:9"
    )

    print("\n=== 3. Resultado devuelto por el servicio de Adobe Firefly ===")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))