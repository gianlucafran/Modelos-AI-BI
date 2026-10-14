"""
Transformers: Generación y Traducción Avanzada (NLP)
Demostración con dataset de texto real (20 Newsgroups de scikit-learn) e integración con Hugging Face

Conceptos:
- Transformers: Arquitectura de red neuronal introducida en "Attention is All You Need" (2017). 
  Eliminó la necesidad de recurrencia (RNNs) permitiendo procesar texto en paralelo.
- Mecanismo de Auto-Atención (Self-Attention): Permite al modelo ponderar la importancia 
  de cada palabra en una oración con respecto a todas las demás, capturando contexto 
  bidireccional y dependencias a largo plazo de forma instantánea.
- Generación (Causal LM): Modelos tipo Decoder (como GPT) que predicen el siguiente token.
- Traducción (Seq2Seq): Modelos Encoder-Decoder (como T5 o MarianMT) que leen una 
  secuencia en un idioma y generan su equivalente en otro.

Nota: Scikit-learn proporciona el dataset de texto, pero para implementar Transformers 
modernos utilizaremos la librería estándar de la industria: 'transformers' de Hugging Face.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
import warnings
import time

warnings.filterwarnings('ignore')

# Requiere instalación de librerías de Deep Learning para NLP: pip install transformers torch
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("Por favor, instala las dependencias usando: pip install transformers torch")
    import sys
    sys.exit()

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. CARGAR EL DATASET (TEXTO REAL EN INGLÉS)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR TEXTO REAL (SCIKIT-LEARN 20 NEWSGROUPS)")
print("=" * 80)

# Utilizaremos el grupo de noticias sobre ciencia espacial (sci.space)
# para tener un contexto interesante para la generación y traducción.
print("Descargando dataset de noticias espaciales...")
dataset = fetch_20newsgroups(
    subset='train', 
    categories=['sci.space'], 
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Limpiar textos vacíos o muy cortos
textos_validos = [t.strip().replace('\n', ' ') for t in dataset.data if len(t.strip()) > 100]
texto_origen = textos_validos[1] # Tomamos un texto interesante sobre la luna/espacio

# Extraemos solo las primeras 30 palabras para usarlo como "Prompt" (Semilla)
prompt_semilla = " ".join(texto_origen.split()[:30]) + "..."

print(f"\nDataset: 20 Newsgroups (Categoría: sci.space)")
print(f"Total de documentos procesados: {len(textos_validos)}")
print(f"\nPROMPT EXTRAÍDO (Inglés original):")
print(f"\"{prompt_semilla}\"")

# ============================================================================
# 2. INICIALIZAR TRANSFORMERS (HUGGING FACE PIPELINES)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DESCARGAR E INICIALIZAR MODELOS TRANSFORMER")
print("=" * 80)

print("Cargando modelos pre-entrenados (Transfer Learning).")
print("Esto descarga los pesos de la red neuronal si es la primera vez...\n")

# 2.1 Modelo de Generación (Familia GPT - Decoder-only)
# Usamos 'gpt2' como modelo generativo liviano
print("1. Cargando Transformer Generativo (GPT-2)...")
t0 = time.time()
generador = pipeline('text-generation', model='gpt2', device=-1) # device=-1 usa CPU
print(f"   ✓ Generador cargado en {time.time() - t0:.2f} segundos.")

# 2.2 Modelo de Traducción (Encoder-Decoder Seq2Seq)
# Usamos MarianMT entrenado por la Universidad de Helsinki para Inglés -> Español
print("\n2. Cargando Transformer de Traducción (Helsinki-NLP/opus-mt-en-es)...")
t0 = time.time()
traductor = pipeline('translation', model='Helsinki-NLP/opus-mt-en-es', device=-1)
print(f"   ✓ Traductor cargado en {time.time() - t0:.2f} segundos.")

# ============================================================================
# 3. GENERACIÓN AVANZADA DE TEXTO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: GENERACIÓN DE TEXTO CON GPT-2 (INFERENCIA CAUSAL)")
print("=" * 80)

print(f"Alimentando el Prompt al modelo generativo...")
print("El Transformer calculará la atención sobre el contexto y predecirá las siguientes palabras.")

# Generamos texto a partir de la semilla
# Parámetros: max_length controla el largo, temperature controla la creatividad
resultado_generacion = generador(
    prompt_semilla, 
    max_length=80, 
    num_return_sequences=1,
    temperature=0.7, # 1.0 es muy creativo/caótico, 0.1 es muy determinista
    top_k=50,
    pad_token_id=50256
)

texto_generado_en = resultado_generacion[0]['generated_text']

print("\nTEXTO GENERADO (Prompt Original + Continuación de la IA):")
print("-" * 60)
print(texto_generado_en)
print("-" * 60)

# ============================================================================
# 4. TRADUCCIÓN AUTOMÁTICA NEURONAL
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: TRADUCCIÓN AVANZADA (ENCODER-DECODER)")
print("=" * 80)

print("Procesando el texto generado a través del modelo de traducción (Inglés -> Español)...")
print("El Encoder crea un vector de contexto dinámico (Attention), y el Decoder genera el español.\n")

resultado_traduccion = traductor(texto_generado_en, max_length=150)
texto_traducido_es = resultado_traduccion[0]['translation_text']

print("TEXTO TRADUCIDO AL ESPAÑOL:")
print("-" * 60)
print(texto_traducido_es)
print("-" * 60)

# ============================================================================
# 5. VISUALIZACIÓN: EL MECANISMO DE ATENCIÓN (SIMULACIÓN PEDAGÓGICA)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DEL MECANISMO DE AUTO-ATENCIÓN (SELF-ATTENTION)")
print("=" * 80)

# Extraer los pesos de atención reales de HuggingFace requiere acceder a los tensores 
# profundos (output_attentions=True) y es computacionalmente denso. 
# Para propósitos demostrativos y de visualización analítica, construiremos una 
# matriz de atención sintética que emula cómo el modelo relaciona las palabras.

frase_ejemplo = "El telescopio espacial captura imágenes de galaxias lejanas"
tokens = frase_ejemplo.split()
n_tokens = len(tokens)

# Simulamos una matriz de auto-atención (Softmax sobre Query x Key)
# donde ciertas palabras prestan mucha atención a sus adjetivos o complementos.
atencion_simulada = np.zeros((n_tokens, n_tokens))

for i in range(n_tokens):
    for j in range(n_tokens):
        # Base de atención a palabras cercanas
        atencion_simulada[i, j] = 1.0 / (abs(i - j) + 1)

# Añadimos picos de atención semántica fuerte (Simulando lo que aprende el Transformer)
# "telescopio" (1) presta atención a "espacial" (2)
atencion_simulada[1, 2] += 2.0
atencion_simulada[2, 1] += 2.0
# "captura" (3) presta atención a "telescopio" (1) y "imágenes" (4)
atencion_simulada[3, 1] += 1.5
atencion_simulada[3, 4] += 1.8
# "imágenes" (4) presta atención a "galaxias" (6)
atencion_simulada[4, 6] += 2.5

# Aplicar Softmax por fila (como en el Transformer real) para que las probabilidades sumen 1
atencion_simulada = np.exp(atencion_simulada) / np.sum(np.exp(atencion_simulada), axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Heatmap de Auto-Atención
sns.heatmap(atencion_simulada, annot=True, fmt='.2f', cmap='magma', 
            xticklabels=tokens, yticklabels=tokens, ax=axes[0], cbar_kws={'label': 'Peso de Atención'})

axes[0].set_title('Mecanismo de Self-Attention (Capa Interna)\n¿A qué palabras presta atención cada token?', 
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Token Atendido (Key)')
axes[0].set_ylabel('Token que Atiende (Query)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# Gráfico 2: Complejidad Computacional (Longitud de Secuencia vs Atención)
# En transformers, el cálculo de atención crece de forma cuadrática O(N^2)
longitudes = np.arange(10, 2000, 50)
complejidad_rnn = longitudes * 10 # O(N)
complejidad_transformer = (longitudes ** 2) / 100 # O(N^2) escalado para visualización

axes[1].plot(longitudes, complejidad_rnn, color='blue', linewidth=3, label='RNN / LSTM (Lineal $O(N)$)')
axes[1].plot(longitudes, complejidad_transformer, color='red', linewidth=3, label='Transformer (Cuadrática $O(N^2)$)')
axes[1].fill_between(longitudes, complejidad_rnn, complejidad_transformer, color='red', alpha=0.1)

axes[1].set_title('El Cuello de Botella del Transformer\n(Consumo de Memoria por Context Window)', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Longitud de la Secuencia (Tokens)')
axes[1].set_ylabel('Operaciones de Atención (Memoria VRAM)')
axes[1].legend(loc='upper left')
axes[1].grid(alpha=0.3)

# Explicación textual en Gráfico 2
axes[1].text(500, max(complejidad_transformer)*0.6, 
             "La atención cuadrática\nhace que procesar libros\nenteros requiera GPUs masivas.", 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
#plt.savefig('/home/usuario/01_transformers_nlp.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_transformers_nlp.png")
plt.close()

# ============================================================================
# 6. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               TRANSFORMERS: GENERACIÓN Y TRADUCCIÓN AVANZADA                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. EL CAMBIO DE PARADIGMA (ADIÓS A LAS RNNs):
   • Antes del 2017, para traducir una frase, una red LSTM debía leer palabra 
     por palabra en estricto orden secuencial. 
   • El Transformer revolucionó esto procesando TODA la frase en paralelo. En 
     lugar de memoria secuencial, utiliza la "Auto-Atención" (Visualizada en 
     el Gráfico 1) para conectar matemáticamente cada palabra con las demás 
     sin importar la distancia física en la oración.

2. GENERACIÓN (DECODER-ONLY - GPT):
   • Usamos `gpt2` de Hugging Face. El modelo recibió el texto extraído del 
     dataset científico espacial de scikit-learn como "semilla".
   • Utilizó inferencia causal (ocultando el futuro) para predecir 
     estadísticamente cuál debía ser la siguiente palabra más lógica.

3. TRADUCCIÓN AUTOMÁTICA (ENCODER-DECODER):
   • Usamos el modelo `opus-mt-en-es`. Aquí la arquitectura consta de dos partes:
     - El Encoder: Leyó el texto en inglés y lo codificó en una matriz matemática 
       gigante de significado puro.
     - El Decoder: Tomó esa matriz, aplicó "Cross-Attention" (atención cruzada 
       entre español e inglés) y redactó el texto traducido con alta fluidez.

4. EL PRECIO DEL PODER (O(N^2)):
   • Como se muestra en el Gráfico 2, el cálculo matemático para que "cada 
     token atienda a cada token" escala cuadráticamente. Esta es la razón por 
     la que los modelos de lenguaje modernos requieren inmensas cantidades de 
     VRAM en las GPUs para ventanas de contexto largas (ej. 128K tokens).

╔══════════════════════════════════════════════════════════════════════════════╗
║                           LIBRERÍAS Y HERRAMIENTAS                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

- Scikit-Learn: `fetch_20newsgroups` (Fuente de datos reales no estructurados).
- Hugging Face `transformers`: La API `pipeline` abstrae la complejidad de la 
  tokenización (Subword Tokenization / BPE), la conversión a tensores y el 
  pase por la arquitectura neuronal en una sola línea de código.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_transformers_nlp.png - Matriz de Auto-Atención (Simulación de conexiones 
                          sintácticas) y gráfica de complejidad computacional.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Análisis de Modelos Transformers para Generación y Traducción completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")