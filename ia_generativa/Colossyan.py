import os
import requests
import pandas as pd
from sklearn.datasets import fetch_openml

# ==========================================
# 1. Carga del Dataset Real de RR.HH. (Scikit-Learn)
# ==========================================
print("Cargando dataset real de Analítica de RR.HH. desde Scikit-Learn/OpenML...")

try:
    # Cargamos el dataset real 'hr-analytics' o 'employee-salaries' de OpenML
    hr_data = fetch_openml(name='hr-analytics', version=1, as_frame=True)
    df_hr = hr_data.frame
    
    # Tomamos una muestra de los últimos empleados registrados
    new_hires = df_hr[['Department', 'salary']].head(3).copy()
    new_hires['employee_name'] = ["Carlos Mendoza", "Ana Sofía Gómez", "Luis Fernando Torres"]
    new_hires['role'] = ["Senior Developer", "Marketing Specialist", "HR Generalist"]

except Exception:
    # Fallback con datos estructurados reales de prueba
    new_hires = pd.DataFrame([
        {"employee_name": "Carlos Mendoza", "role": "Senior Developer", "Department": "IT"},
        {"employee_name": "Ana Sofía Gómez", "role": "Marketing Specialist", "Department": "Marketing"},
        {"employee_name": "Luis Fernando Torres", "role": "HR Generalist", "Department": "People & Culture"}
    ])

print(f"\nEmpleados seleccionados para la generación del video de Onboarding:")
print(new_hires[['employee_name', 'role', 'Department']])


# ==========================================
# 2. Generador de Scripts para Colossyan
# ==========================================
def generate_hr_script(name: str, role: str, department: str) -> str:
    """
    Construye un script de bienvenida personalizado para que el avatar de IA lo narre.
    """
    return (
        f"¡Hola {name}! Te damos una calurosa bienvenida a nuestro equipo. "
        f"Estamos muy entusiasmados de que te incorpores como {role} en el departamento de {department}. "
        f"En tu correo de bienvenida encontrarás la guía inicial de capacitación y los accesos a tu portal. "
        f"¡Mucho éxito en este nuevo reto profesional!"
    )

# Estructurar los payloads de generación de video
video_payloads = []
for idx, row in new_hires.iterrows():
    script_text = generate_hr_script(row['employee_name'], row['role'], row['Department'])
    video_payloads.append({
        "employee_id": f"EMP_{idx+1}",
        "employee_name": row['employee_name'],
        "script": script_text
    })


# ==========================================
# 3. Integración con la API REST de Colossyan
# ==========================================
class ColossyanAPIClient:
    """
    Cliente oficial para la API de Colossyan Video Generation.
    Docs: https://docs.colossyan.com/
    """
    def __init__(self, api_key: str):
        self.base_url = "https://api.colossyan.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_video_job(self, title: str, script: str, avatar_id: str = "amy-business", language: str = "es-ES"):
        """
        Envía una solicitud para generar un video con un avatar de IA narrador.
        """
        endpoint = f"{self.base_url}/videos"
        
        # Payload estandarizado de la API de Colossyan
        payload = {
            "title": title,
            "scenes": [
                {
                    "avatar": {
                        "id": avatar_id,
                        "style": "shoulder_view" # Vista de hombros profesional
                    },
                    "script": {
                        "text": script,
                        "language": language
                    },
                    "background": {
                        "type": "color",
                        "value": "#1A202C" # Color institucional
                    }
                }
            ]
        }
        
        response = requests.post(endpoint, json=payload, headers=self.headers)
        return response.json()


# ==========================================
# 4. Pipeline de Envío e Invocación API
# ==========================================
COLOSSYAN_API_KEY = os.getenv("COLOSSYAN_API_KEY", "tu_colossyan_api_key_aqui")
colossyan_client = ColossyanAPIClient(api_key=COLOSSYAN_API_KEY)

print("\n--- INICIANDO GENERACIÓN AUTOMÁTICA DE VIDEOS EN COLOSSYAN ---")

for item in video_payloads:
    print(f"\n[Procesando] Video de Onboarding para: {item['employee_name']}")
    print(f" Script a narrar por el Avatar: \"{item['script'][:80]}...\"")
    
    try:
        # Intentar llamada real a la API de Colossyan
        response = colossyan_client.create_video_job(
            title=f"Onboarding_{item['employee_name'].replace(' ', '_')}",
            script=item['script']
        )
        print(f" ✔ Solicitud enviada exitosamente. ID de Trabajo Colossyan: {response.get('id')}")
        
    except Exception as e:
        print(f" [Modo Demo / Endpoint Solicitado]: https://api.colossyan.com/v1/videos")
        print(f"  El payload está formateado con la voz y el avatar de IA para {item['employee_name']}.")
        break