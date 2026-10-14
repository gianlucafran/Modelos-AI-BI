import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from xai_sdk import Client
from xai_sdk.chat import system, user

# ==========================================
# 1. Carga de Dataset Real (S&P 500 Stocks via Scikit-Learn)
# ==========================================
print("Cargando dataset real de datos bursátiles (S&P 500) desde Scikit-Learn/OpenML...")

# Cargamos el dataset real de datos de mercado 'sp500_values'
sp500_data = fetch_openml(name='sp500_values', version=1, as_frame=True)
df = sp500_data.frame

# Aseguramos el orden cronológico y limpiamos tipos de datos
df['date'] = pd.to_datetime(df['date'])
df['close'] = df['close'].astype(float)
df = df.sort_values('date').reset_index(drop=True)

# ==========================================
# 2. Preprocesamiento y Cálculo de Tendencias
# ==========================================
# Calculamos indicadores técnicos de tendencia
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['Daily_Return'] = df['close'].pct_change()

# Extraemos una ventana reciente de datos para el análisis
recent_window = df.tail(60)

price_start = recent_window['close'].iloc[0]
price_end = recent_window['close'].iloc[-1]
total_change_pct = ((price_end - price_start) / price_start) * 100
volatility = recent_window['Daily_Return'].std() * np.sqrt(252) * 100

# Determinación cuantitativa previa de la tendencia
trend_direction = "Alcista" if total_change_pct > 0 else "Bajista"

metrics_summary = {
    "Fecha Inicio Ventana": str(recent_window['date'].iloc[0].date()),
    "Fecha Fin Ventana": str(recent_window['date'].iloc[-1].date()),
    "Precio Inicial ($)": round(price_start, 2),
    "Precio Final ($)": round(price_end, 2),
    "Variación Porcentual (%)": round(total_change_pct, 2),
    "Dirección Cuantitativa": trend_direction,
    "Volatilidad Anualizada Est. (%)": round(volatility, 2),
    "Última Promedio Móvil 20D": round(recent_window['SMA_20'].iloc[-1], 2),
    "Última Promedio Móvil 50D": round(recent_window['SMA_50'].iloc[-1], 2)
}

print("\n--- MÉTIRCAS EXTRAÍDAS PARA EL ANÁLISIS ---")
for k, v in metrics_summary.items():
    print(f"• {k}: {v}")

# ==========================================
# 3. Integración con Grok 4.1 para Análisis Estratégico
# ==========================================
# Nota: La API requiere configurar la variable de entorno: export XAI_API_KEY="tu_api_key"
api_key = os.getenv("XAI_API_KEY", "tu_xai_api_key_aqui")

# Inicializar el cliente oficial xAI SDK
client = Client(api_key=api_key)

# Definición del Prompt con contexto de Tendencias
prompt_analisis = f"""
Actúa como un Analista Financiero Senior. Analiza el siguiente resumen cuantitativo de la tendencia de mercado de la acción/índice S&P 500 extraído de los datos:

{metrics_summary}

Proporciona:
1. **Evaluación de la Tendencia Actual:** Explicación técnica de la dirección alcista/bajista y las promedios móviles.
2. **Evaluación de Riesgo y Volatilidad:** Interpretación de la volatilidad anualizada calculada.
3. **Perspectiva y Recomendación Estratégica:** Sugerencia analítica para inversores basada en la tendencia observada.
"""

print("\nEnviando datos procesados a Grok 4.1 para generación del reporte de tendencias...")

try:
    # Usando el modelo Grok 4.1 (vía SDK oficial)
    chat = client.chat.create(
        model="grok-4.1-fast-reasoning", # O 'grok-4.1-fast-non-reasoning' / 'grok-4.3'
        messages=[
            system("Eres un analista cuantitativo experto en inteligencia de mercados y detección de tendencias financiearas."),
            user(prompt_analisis)
        ]
    )
    
    response = chat.sample()
    
    print("\n=== REPORTE DE ANÁLISIS DE TENDENCIAS (GROK 4.1) ===")
    print(response.content)

except Exception as e:
    print(f"\n[Nota API]: Para ejecutar la inferencia directa con Grok 4.1 asegura tener instalada la librería `xai-sdk` y tu `XAI_API_KEY` activa.")
    print(f"Detalle técnico de la llamada: {e}")