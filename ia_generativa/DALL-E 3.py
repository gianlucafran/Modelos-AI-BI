"""
Automatización de Ilustraciones de Marketing con DALL-E 3 y Scikit-Learn
========================================================================
Aplicación: Generación automatizada de recursos gráficos para campañas 
de marketing digital personalizadas basadas en catálogos de producto.
"""

import os
import json
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. CLIENTE DE OPENAI (Usando la versión oficial más reciente)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "tu-api-key-aqui"))
except ImportError:
    raise ImportError("Por favor instala la librería oficial: pip install openai pillow pandas scikit-learn")


# 2. CARGA Y PROCESAMIENTO DEL DATASET CON SCIKIT-LEARN
def cargar_y_procesar_dataset():
    """
    Simula o carga un dataset de catálogo de productos para marketing
    y utiliza KMeans de scikit-learn para agrupar los productos por tipo de audiencia.
    """
    data = {
        "producto_id": [101, 102, 103, 104],
        "nombre": ["Auriculares Wireless Noise-Canceling", "Bebida Energética Orgánica Citrus", "Sillón Ergonómico Minimalista", "Zapatillas Running Ultraligeras"],
        "categoria": ["Tecnología", "Alimentos & Bebidas", "Mobiliario & Hogar", "Calzado Deportivo"],
        "descripcion": [
            "auriculares premium sonido de alta fidelidad diseño futurista elegante mate",
            "bebida natural energia limpia citricos color vibrante estilo fresco juvenil",
            "silla de oficina diseno nordico madera natural tonos neutros iluminacion suave",
            "tenis de carrera dinamismo movimiento neon velocidad atletismo moderno"
        ],
        "estilo_marketing": ["Futurista / Tech", "Vibrante / Pop Art", "Minimalista / Nórdico", "Dinámico / Neón"]
    }
    
    df = pd.DataFrame(data)
    
    # Aplicar scikit-learn: Vectorización TF-IDF y Clustering
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['descripcion'])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['segmento_cluster'] = kmeans.fit_predict(X)
    
    return df


# 3. CONSTRUCTOR DE PROMPTS DE MARKETING PARA DALL-E 3
def construir_prompt_marketing(row):
    """
    Construye un prompt detallado para DALL-E 3 optimizado para ilustración comercial.
    """
    prompt = (
        f"Una ilustración comercial de alta calidad para marketing digital del producto '{row['nombre']}'. "
        f"Estilo visual: {row['estilo_marketing']}. "
        f"Detalles de la escena: {row['descripcion']}. "
        f"Composición limpia orientada a anuncio publicitario, iluminación profesional, render 3D estilizado, "
        f"sin texto superpuesto, estética limpia y atractiva para redes sociales."
    )
    return prompt


# 4. GENERACIÓN DE IMAGEN CON DALL-E 3 API
def generar_ilustracion_dalle3(prompt, calidad="standard", tamaño="1024x1024", estilo="vivid"):
    """
    Realiza la llamada a la API de OpenAI DALL-E 3.
    Parametros DALL-E 3:
    - quality: 'standard' o 'hd'
    - style: 'vivid' (colores hiperrealistas/dramáticos) o 'natural'
    - size: '1024x1024', '1792x1024' (horizontal), '1024x1792' (vertical)
    """
    print(f"\nGenerando ilustración con DALL-E 3...")
    print(f"Prompt enviado: {prompt[:100]}...\n")

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=tamaño,
            quality=calidad,
            style=estilo,
            n=1
        )

        url_imagen = response.data[0].url
        prompt_revisado = response.data[0].revised_prompt
        
        return url_imagen, prompt_revisado

    except Exception as e:
        print(f"Error en la llamada a la API (asegúrate de configurar OPENAI_API_KEY): {e}")
        return None, None


# 5. EJECUCIÓN DEL FLUJO DE TRABAJO
if __name__ == "__main__":
    # Cargar datos procesados con scikit-learn
    df_productos = cargar_y_procesar_dataset()
    print("Dataset de Productos Procesado:")
    print(df_productos[['producto_id', 'nombre', 'estilo_marketing', 'segmento_cluster']])

    # Seleccionar un producto del dataset para la campaña de marketing
    producto_campaña = df_productos.iloc[0] # Auriculares Wireless
    
    prompt_final = construir_prompt_marketing(producto_campaña)
    
    # Generar la imagen (requiere API key activa)
    if os.getenv("OPENAI_API_KEY"):
        url_img, prompt_opt = generar_ilustracion_dalle3(
            prompt=prompt_final,
            calidad="hd",
            tamaño="1024x1024",
            estilo="vivid"
        )
        
        if url_img:
            print(f"URL de la ilustración generada: {url_img}")
            print(f"Prompt optimizado automáticamente por DALL-E 3: {prompt_opt}")
            
            # Descargar y mostrar la ilustración
            res = requests.get(url_img)
            img = Image.open(BytesIO(res.content))
            img.show()
    else:
        print("\n[Modo Demostración]: Configura la variable de entorno 'OPENAI_API_KEY' para ejecutar la generación real.")
        print(f"Prompt listo para producción: {prompt_final}")